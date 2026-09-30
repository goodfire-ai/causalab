//! The calibration figures the planner reads instead of hard-coded constants.
//!
//! A [`Profile`] says what each storage class (and, overriding it, each mount
//! point) can do: how fast one file reads, how the aggregate grows with files
//! in flight, whether splitting a file across readers helps, whether cuFile
//! can register handles there; and what the device path costs. The planner
//! is still a pure function — [`crate::plan::plan_read`] takes the profile as
//! a value and never measures anything — but the numbers behind its rules now
//! arrive from outside, so a calibration tool can produce them for a machine
//! up front and the same library plans correctly on any of them.
//!
//! [`Profile::default_measured`] is the built-in profile: the measurements in
//! `docs/fastersafetensors.md` §Measurements, serialized verbatim at
//! `docs/profiles/h100-nfs-2026-09-08.json`. The JSON schema is
//! documented in `docs/fastersafetensors.md` §Calibration profile.

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::env::{Env, StorageClass};
use crate::error::ProfileError;

/// The schema version this build reads and writes.
pub const SCHEMA_VERSION: u32 = 1;

/// A storage entry's key: the [`StorageClass`] variant's name, without the
/// file-system name `OtherNetwork` and `Other` carry (a profile speaks for a
/// class; a particular mount is overridden through [`Profile::mounts`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum StorageClassKey {
    /// [`StorageClass::LocalBlock`].
    LocalBlock,
    /// [`StorageClass::Nfs`].
    Nfs,
    /// [`StorageClass::Fuse`].
    Fuse,
    /// [`StorageClass::Ram`].
    Ram,
    /// [`StorageClass::OtherNetwork`].
    OtherNetwork,
    /// [`StorageClass::Other`].
    Other,
}

impl StorageClassKey {
    /// Every key, in declaration order.
    pub const ALL: [StorageClassKey; 6] = [
        StorageClassKey::LocalBlock,
        StorageClassKey::Nfs,
        StorageClassKey::Fuse,
        StorageClassKey::Ram,
        StorageClassKey::OtherNetwork,
        StorageClassKey::Other,
    ];

    /// The key as it appears in JSON.
    pub fn name(self) -> &'static str {
        match self {
            StorageClassKey::LocalBlock => "LocalBlock",
            StorageClassKey::Nfs => "Nfs",
            StorageClassKey::Fuse => "Fuse",
            StorageClassKey::Ram => "Ram",
            StorageClassKey::OtherNetwork => "OtherNetwork",
            StorageClassKey::Other => "Other",
        }
    }

    /// Parse a JSON key.
    pub fn parse(key: &str) -> Result<StorageClassKey, ProfileError> {
        StorageClassKey::ALL
            .into_iter()
            .find(|k| k.name() == key)
            .ok_or_else(|| ProfileError::UnknownStorageClass {
                key: key.to_owned(),
            })
    }
}

impl fmt::Display for StorageClassKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl From<&StorageClass> for StorageClassKey {
    fn from(class: &StorageClass) -> StorageClassKey {
        match class {
            StorageClass::LocalBlock => StorageClassKey::LocalBlock,
            StorageClass::Nfs => StorageClassKey::Nfs,
            StorageClass::Fuse => StorageClassKey::Fuse,
            StorageClass::Ram => StorageClassKey::Ram,
            StorageClass::OtherNetwork(_) => StorageClassKey::OtherNetwork,
            StorageClass::Other(_) => StorageClassKey::Other,
        }
    }
}

/// One row of an aggregate-throughput table: what the mount delivered with
/// this many files being read at once.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AggregatePoint {
    /// Files read concurrently.
    pub files_in_flight: u32,
    /// Aggregate throughput, GB/s (decimal), cold cache.
    pub gbps: f64,
}

/// Whether and how well cuFile works on a storage entry.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GdsProfile {
    /// `cuFileHandleRegister` succeeds for files here. False on mounts
    /// libcufile cannot build a file object for (NFSv3/TCP; md RAID without
    /// udev `ID_FS_USAGE`).
    pub registers: bool,
    /// `cuFileRead` throughput into device memory, GB/s, when measured.
    #[serde(default)]
    pub read_gbps: Option<f64>,
}

/// What one storage class or mount can do.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StorageProfile {
    /// Throughput of one file read alone, GB/s, cold cache — however many
    /// readers or whatever block size; the ceiling the mount puts on a file.
    pub single_file_gbps: f64,
    /// Aggregate throughput against files in flight, cold cache, strictly
    /// increasing in `files_in_flight`. The planner keeps in flight the
    /// smallest count within [`PLATEAU_TOLERANCE`] of the table's maximum.
    pub aggregate_gbps: Vec<AggregatePoint>,
    /// Whether splitting one file across concurrent readers raises its
    /// throughput. False where `single_file_gbps` is a hard per-file cap.
    pub split_helps: bool,
    /// Aggregate throughput from page cache (warm), GB/s, when measured.
    #[serde(default)]
    pub page_cache_gbps: Option<f64>,
    /// Cost of `open(2)` on this mount, milliseconds, when measured.
    #[serde(default)]
    pub open_cost_ms: Option<f64>,
    /// Fixed cost of one read call beyond its bytes — the round trip a
    /// `pread` pays however small it is — microseconds, when measured. With
    /// `single_file_gbps` it says how large a gap between two wanted runs
    /// is cheaper to read through than to skip with a second call
    /// ([`crate::plan::coalesce_policy`]). `None` is unmeasured: the
    /// planner coalesces only up to its floor.
    #[serde(default)]
    pub io_cost_us: Option<f64>,
    /// cuFile on this entry; `None` means unmeasured, which the planner
    /// treats as "does not register".
    #[serde(default)]
    pub gds: Option<GdsProfile>,
}

/// The `io_cost_us` the built-in NFS entry carries until it is measured:
/// **a placeholder**. One small `pread` on NFSv3 over TCP is a request and
/// a reply on the wire plus the server's own lookup, a few hundred
/// microseconds on a datacenter LAN; 200 µs at 3 GB/s makes the coalescing
/// gap 600 KB, well inside the planner's 64 KiB–8 MiB clamp either way. The
/// multi-rank bench measures it; until then the source string says
/// "unmeasured".
pub const NFS_IO_COST_US_PLACEHOLDER: f64 = 200.0;

/// Fraction below the aggregate table's maximum that still counts as the
/// plateau. The measured NFS table reaches 8.3 GB/s at 16 files and 9.7 at
/// 32 (`docs/fastersafetensors.md` §Measurements): 16 is within 15% and is the count the
/// warm runs also favour (25.9 GB/s at 16 against 17.7 at 32), so 16 is the
/// plateau. At 10% the same table would pick 32.
pub const PLATEAU_TOLERANCE: f64 = 0.15;

impl StorageProfile {
    /// The smallest `files_in_flight` in the aggregate table whose throughput
    /// is within [`PLATEAU_TOLERANCE`] of the table's maximum, with the row
    /// picked and the maximum row, for the reason line. `None` only for an
    /// empty table, which validation rejects.
    pub fn plateau(&self) -> Option<Plateau> {
        let max = self
            .aggregate_gbps
            .iter()
            .copied()
            .max_by(|a, b| a.gbps.total_cmp(&b.gbps))?;
        let floor = max.gbps * (1.0 - PLATEAU_TOLERANCE);
        let picked = self
            .aggregate_gbps
            .iter()
            .copied()
            .find(|p| p.gbps >= floor)?;
        Some(Plateau { picked, max })
    }

    /// Whether cuFile registers handles here; unmeasured counts as no.
    pub fn gds_registers(&self) -> bool {
        self.gds.is_some_and(|g| g.registers)
    }

    /// The NFS entry of [`Profile::default_measured`]: what the planner
    /// assumes for a storage class the profile does not describe. It is the
    /// least optimistic shape measured — a hard per-file cap, throughput only
    /// from files in flight, no cuFile.
    pub fn conservative() -> StorageProfile {
        StorageProfile {
            single_file_gbps: 3.0,
            aggregate_gbps: vec![
                AggregatePoint {
                    files_in_flight: 1,
                    gbps: 2.6,
                },
                AggregatePoint {
                    files_in_flight: 4,
                    gbps: 7.4,
                },
                AggregatePoint {
                    files_in_flight: 16,
                    gbps: 8.3,
                },
                AggregatePoint {
                    files_in_flight: 32,
                    gbps: 9.7,
                },
            ],
            split_helps: false,
            page_cache_gbps: Some(25.9),
            open_cost_ms: None,
            // unmeasured placeholder: a small NFS read over TCP is a
            // round trip of a few hundred microseconds; the multi-rank
            // bench replaces this with a figure
            io_cost_us: Some(NFS_IO_COST_US_PLACEHOLDER),
            gds: Some(GdsProfile {
                registers: false,
                read_gbps: Some(2.5),
            }),
        }
    }

    fn validate(&self, entry: &str) -> Result<(), ProfileError> {
        positive(entry, "single_file_gbps", self.single_file_gbps)?;
        if self.aggregate_gbps.is_empty() {
            return Err(ProfileError::EmptyTable {
                entry: entry.to_owned(),
                table: "aggregate_gbps",
            });
        }
        let mut previous: Option<u32> = None;
        for (index, point) in self.aggregate_gbps.iter().enumerate() {
            positive(entry, "aggregate_gbps.gbps", point.gbps)?;
            if point.files_in_flight == 0 || previous.is_some_and(|p| p >= point.files_in_flight) {
                return Err(ProfileError::AggregateNotSorted {
                    entry: entry.to_owned(),
                    index,
                    files_in_flight: point.files_in_flight,
                });
            }
            previous = Some(point.files_in_flight);
        }
        if let Some(v) = self.page_cache_gbps {
            positive(entry, "page_cache_gbps", v)?;
        }
        if let Some(v) = self.open_cost_ms {
            non_negative(entry, "open_cost_ms", v)?;
        }
        if let Some(v) = self.io_cost_us {
            non_negative(entry, "io_cost_us", v)?;
        }
        if let Some(v) = self.gds.and_then(|g| g.read_gbps) {
            positive(entry, "gds.read_gbps", v)?;
        }
        Ok(())
    }
}

/// The row [`StorageProfile::plateau`] picked and the table's maximum.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Plateau {
    /// The smallest row within tolerance of the maximum.
    pub picked: AggregatePoint,
    /// The row with the highest throughput.
    pub max: AggregatePoint,
}

/// What the host-to-device path costs.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeviceProfile {
    /// `cudaMemcpyAsync` host-to-device from pinned memory, GB/s.
    pub h2d_pinned_gbps: f64,
    /// The same from pageable memory, GB/s.
    pub h2d_pageable_gbps: f64,
    /// `cudaHostAlloc` cost, milliseconds per MiB allocated.
    pub pinned_alloc_ms_per_mib: f64,
}

impl DeviceProfile {
    fn validate(&self) -> Result<(), ProfileError> {
        positive("device", "h2d_pinned_gbps", self.h2d_pinned_gbps)?;
        positive("device", "h2d_pageable_gbps", self.h2d_pageable_gbps)?;
        non_negative(
            "device",
            "pinned_alloc_ms_per_mib",
            self.pinned_alloc_ms_per_mib,
        )
    }
}

fn positive(entry: &str, field: &'static str, value: f64) -> Result<(), ProfileError> {
    if value > 0.0 {
        Ok(())
    } else {
        Err(ProfileError::NonPositiveRate {
            entry: entry.to_owned(),
            field,
            value,
        })
    }
}

fn non_negative(entry: &str, field: &'static str, value: f64) -> Result<(), ProfileError> {
    if value >= 0.0 {
        Ok(())
    } else {
        Err(ProfileError::NonPositiveRate {
            entry: entry.to_owned(),
            field,
            value,
        })
    }
}

/// The calibration of one machine, or a family of them.
///
/// Deserializing validates: an unknown storage-class key, a non-positive
/// rate, an aggregate table that is empty or not strictly increasing, or an
/// unsupported `schema_version` is a [`ProfileError`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "RawProfile")]
pub struct Profile {
    /// The schema this document follows; [`SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Where and when these numbers were measured: the hardware type, the
    /// storage and the date, plus notes on unmeasured entries. It names
    /// hardware by type, not by host name.
    /// Printed in every reason line the planner writes from this profile.
    pub source: String,
    /// One entry per storage class described.
    pub storage: BTreeMap<StorageClassKey, StorageProfile>,
    /// Entries for particular mount points, overriding the class entry for
    /// every path under them; the longest mount point that is a prefix of
    /// the path wins, as in [`Env::storage_class`].
    pub mounts: BTreeMap<PathBuf, StorageProfile>,
    /// The device path, when measured.
    pub device: Option<DeviceProfile>,
}

/// The document as written, before the keys are typed and the numbers
/// checked.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawProfile {
    schema_version: u32,
    source: String,
    #[serde(default)]
    storage: BTreeMap<String, StorageProfile>,
    #[serde(default)]
    mounts: BTreeMap<PathBuf, StorageProfile>,
    #[serde(default)]
    device: Option<DeviceProfile>,
}

impl TryFrom<RawProfile> for Profile {
    type Error = ProfileError;

    fn try_from(raw: RawProfile) -> Result<Profile, ProfileError> {
        if raw.schema_version != SCHEMA_VERSION {
            return Err(ProfileError::UnsupportedSchemaVersion {
                found: raw.schema_version,
                supported: SCHEMA_VERSION,
            });
        }
        let storage = raw
            .storage
            .into_iter()
            .map(|(key, entry)| StorageClassKey::parse(&key).map(|k| (k, entry)))
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        let profile = Profile {
            schema_version: raw.schema_version,
            source: raw.source,
            storage,
            mounts: raw.mounts,
            device: raw.device,
        };
        profile.validate()?;
        Ok(profile)
    }
}

/// Which entry the planner used for a path, and what to call it in a reason.
#[derive(Debug, Clone, PartialEq)]
pub struct StorageLookup {
    /// The class the environment gave the path.
    pub class: StorageClass,
    /// The entry the profile provides — or the conservative shape, when it
    /// provides none.
    pub entry: StorageProfile,
    /// How the entry is named in reasons: `Nfs`, `mount "/mnt/nfs"`, or
    /// `OtherNetwork("lustre") (no profile entry; conservative NFS-shaped
    /// defaults)`.
    pub label: String,
    /// Whether the entry came from the profile or is the fallback.
    pub from_profile: bool,
}

impl Profile {
    /// The measurements in `docs/fastersafetensors.md` §Measurements and
    /// `docs/profiles/h100-nfs-2026-09-08.json`: one H100 80 GB host, 16
    /// cores, weights on NFSv3 `nconnect=16`, 2026-09-08; plus the CUDA
    /// runtime notes from the same host (CUDA 13.0, cuFile 1.15.1) of the
    /// same day.
    ///
    /// * `Nfs`: one file caps near 3 GB/s however split (fastsafetensors one
    ///   file at a time, 16 threads, 64–256 MB blocks: 2.9–3.1). Aggregate
    ///   cold: 2.6 GB/s sequential, 7.4 at 4 files, 8.3 at 16, 9.7 at 32
    ///   (the 32 row is fastsafetensors nogds with 32 threads). Warm at 16
    ///   files: 25.9 GB/s. cuFile in compat mode reads an NFSv3 file at
    ///   2.5 GB/s through bounce buffers and no `nvidia_fs` is present:
    ///   `registers: false`. `io_cost_us` is
    ///   [`NFS_IO_COST_US_PLACEHOLDER`], **unmeasured**.
    /// * `LocalBlock`: **unmeasured**. The table is the NFS table cut at 4
    ///   files, which reproduces the first planner's rule (4 files in flight,
    ///   split across readers) until a local mount is measured.
    ///   `split_helps: true` is the property that defines the class.
    ///   `gds.registers: true` is the first planner's assumption that a block
    ///   device is GDS-capable when `nvidia_fs` is loaded; the H100 host's
    ///   `/tmp` (xfs over md RAID) refutes it for that mount, which a
    ///   per-host profile expresses through `mounts`.
    /// * `Ram`: reads are memcpy from page cache; the warm figures stand in:
    ///   7.2 GB/s one stream, 25.9 at 16. Splitting helps; no cuFile.
    /// * `device`: 1 GiB H2D from pinned at 55 GB/s (PCIe Gen5);
    ///   `cudaHostAlloc` 386 ms per GiB ≈ 0.38 ms/MiB, rounded to 0.4;
    ///   pageable H2D not measured, set to the single-stream page-cache
    ///   figure of 7.2 GB/s as a floor.
    pub fn default_measured() -> Profile {
        let nfs = StorageProfile::conservative();
        let local = StorageProfile {
            single_file_gbps: 3.0,
            aggregate_gbps: vec![
                AggregatePoint {
                    files_in_flight: 1,
                    gbps: 2.6,
                },
                AggregatePoint {
                    files_in_flight: 4,
                    gbps: 7.4,
                },
            ],
            split_helps: true,
            page_cache_gbps: None,
            open_cost_ms: None,
            io_cost_us: None,
            gds: Some(GdsProfile {
                registers: true,
                read_gbps: None,
            }),
        };
        let ram = StorageProfile {
            single_file_gbps: 7.2,
            aggregate_gbps: vec![
                AggregatePoint {
                    files_in_flight: 1,
                    gbps: 7.2,
                },
                AggregatePoint {
                    files_in_flight: 4,
                    gbps: 20.0,
                },
                AggregatePoint {
                    files_in_flight: 16,
                    gbps: 25.9,
                },
            ],
            split_helps: true,
            page_cache_gbps: Some(25.9),
            open_cost_ms: None,
            io_cost_us: None,
            gds: None,
        };
        let storage = [
            (StorageClassKey::LocalBlock, local),
            (StorageClassKey::Nfs, nfs),
            (StorageClassKey::Ram, ram),
        ]
        .into_iter()
        .collect();
        Profile {
            schema_version: SCHEMA_VERSION,
            source:
                "h100-nfs-2026-09-08 (docs/fastersafetensors.md; LocalBlock and Nfs.io_cost_us \
                     unmeasured placeholders)"
                    .to_owned(),
            storage,
            mounts: BTreeMap::new(),
            device: Some(DeviceProfile {
                h2d_pinned_gbps: 55.0,
                h2d_pageable_gbps: 7.2,
                pinned_alloc_ms_per_mib: 0.4,
            }),
        }
    }

    /// Parse and validate a JSON document.
    pub fn from_json_str(text: &str) -> Result<Profile, ProfileError> {
        let raw: RawProfile = serde_json::from_str(text)?;
        Profile::try_from(raw)
    }

    /// Read, parse and validate a JSON file. The one place this module
    /// touches the file system.
    pub fn from_path(path: &Path) -> Result<Profile, ProfileError> {
        let text = std::fs::read_to_string(path).map_err(|source| ProfileError::Io {
            path: path.to_owned(),
            source,
        })?;
        Profile::from_json_str(&text)
    }

    /// Pretty-printed JSON, the form `docs/profiles/*.json` is kept in.
    pub fn to_json_string(&self) -> Result<String, ProfileError> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// `over` on top of `base`, section by section: a storage class or mount
    /// present in `over` replaces the base's entry for it, the others stay;
    /// `device` is `over`'s when it has one; `source` names both.
    pub fn merge(base: &Profile, over: &Profile) -> Profile {
        let mut storage = base.storage.clone();
        storage.extend(over.storage.iter().map(|(k, v)| (*k, v.clone())));
        let mut mounts = base.mounts.clone();
        mounts.extend(over.mounts.iter().map(|(k, v)| (k.clone(), v.clone())));
        Profile {
            schema_version: SCHEMA_VERSION,
            source: format!("{} over {}", over.source, base.source),
            storage,
            mounts,
            device: over.device.or(base.device),
        }
    }

    /// Check every number; what deserialization runs, for a profile built in
    /// code.
    pub fn validate(&self) -> Result<(), ProfileError> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(ProfileError::UnsupportedSchemaVersion {
                found: self.schema_version,
                supported: SCHEMA_VERSION,
            });
        }
        for (key, entry) in &self.storage {
            entry.validate(key.name())?;
        }
        for (point, entry) in &self.mounts {
            entry.validate(&format!("mount {:?}", point.display().to_string()))?;
        }
        if let Some(device) = &self.device {
            device.validate()?;
        }
        Ok(())
    }

    /// The entry for `path`: the longest mount-point override that is a
    /// prefix of it, else the entry for its [`StorageClass`] under `env`,
    /// else [`StorageProfile::conservative`] with a label that says so.
    pub fn lookup(&self, env: &Env, path: &Path) -> StorageLookup {
        let class = env.storage_class(path);
        let mount = self
            .mounts
            .iter()
            .filter(|(point, _)| path.starts_with(point))
            .max_by_key(|(point, _)| point.as_os_str().len());
        if let Some((point, entry)) = mount {
            return StorageLookup {
                class,
                entry: entry.clone(),
                label: format!("mount {:?}", point.display().to_string()),
                from_profile: true,
            };
        }
        let key = StorageClassKey::from(&class);
        match self.storage.get(&key) {
            Some(entry) => StorageLookup {
                class,
                entry: entry.clone(),
                label: key.name().to_owned(),
                from_profile: true,
            },
            None => StorageLookup {
                entry: StorageProfile::conservative(),
                label: format!("{class:?} (no profile entry; conservative NFS-shaped defaults)"),
                class,
                from_profile: false,
            },
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::env::{Gds, Mount};
    use proptest::prelude::*;

    fn point(files_in_flight: u32, gbps: f64) -> AggregatePoint {
        AggregatePoint {
            files_in_flight,
            gbps,
        }
    }

    fn entry(single: f64, table: &[(u32, f64)], split_helps: bool) -> StorageProfile {
        StorageProfile {
            single_file_gbps: single,
            aggregate_gbps: table.iter().map(|&(n, g)| point(n, g)).collect(),
            split_helps,
            page_cache_gbps: None,
            open_cost_ms: None,
            io_cost_us: None,
            gds: None,
        }
    }

    fn profile_with(storage: &[(StorageClassKey, StorageProfile)]) -> Profile {
        Profile {
            schema_version: SCHEMA_VERSION,
            source: "test".into(),
            storage: storage.iter().cloned().collect(),
            mounts: BTreeMap::new(),
            device: None,
        }
    }

    const DEFAULTS_FILE: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../docs/profiles/h100-nfs-2026-09-08.json"
    );

    #[test]
    fn default_measured_is_valid_and_round_trips() {
        let p = Profile::default_measured();
        p.validate().unwrap();
        let text = p.to_json_string().unwrap();
        assert_eq!(Profile::from_json_str(&text).unwrap(), p);
        let generic: Profile = serde_json::from_str(&text).unwrap();
        assert_eq!(generic, p);
    }

    #[test]
    fn defaults_file_is_default_measured_serialized() {
        let text = std::fs::read_to_string(DEFAULTS_FILE).unwrap();
        assert_eq!(
            text.trim_end(),
            Profile::default_measured().to_json_string().unwrap(),
            "regenerate {DEFAULTS_FILE} from Profile::default_measured()"
        );
        assert_eq!(
            Profile::from_path(Path::new(DEFAULTS_FILE)).unwrap(),
            Profile::default_measured()
        );
    }

    const B200_FILE: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../docs/profiles/b200-nfs-2026-09-09.json"
    );

    /// The 8x B200 host's profile (`docs/profiles/b200-nfs-2026-09-09.json`,
    /// measured 2026-09-09) loads, is the measured shape — NFS
    /// plateaus at 16 files in flight (2.0 / 9.26 / 12.34 GB/s at 1 / 8 /
    /// 16) and one file scales with readers there, no cuFile — and merges
    /// over the defaults into a valid profile that keeps the defaults' `Ram`
    /// entry, which the host did not measure.
    #[test]
    fn b200_profile_loads_and_merges_over_defaults() {
        let node = Profile::from_path(Path::new(B200_FILE)).unwrap();
        node.validate().unwrap();
        let nfs = &node.storage[&StorageClassKey::Nfs];
        assert_eq!(nfs.plateau().unwrap().picked.files_in_flight, 16);
        assert!(nfs.split_helps);
        assert!(!nfs.gds_registers());
        let local = &node.storage[&StorageClassKey::LocalBlock];
        assert!(local.split_helps);
        assert!(
            !local.gds_registers(),
            "md RAID xfs: unmeasured counts as no"
        );
        assert!(!node.storage.contains_key(&StorageClassKey::Ram));
        assert!(node.device.is_some());

        let merged = Profile::merge(&Profile::default_measured(), &node);
        merged.validate().unwrap();
        assert_eq!(merged.storage[&StorageClassKey::Nfs], *nfs);
        assert_eq!(
            merged.storage[&StorageClassKey::Ram],
            Profile::default_measured().storage[&StorageClassKey::Ram]
        );
        assert_eq!(merged.device, node.device);
    }

    #[test]
    fn default_measured_plateaus_at_sixteen_on_nfs_and_four_locally() {
        let p = Profile::default_measured();
        let nfs = p.storage[&StorageClassKey::Nfs].plateau().unwrap();
        assert_eq!(nfs.picked.files_in_flight, 16);
        assert_eq!(nfs.max.files_in_flight, 32);
        let local = p.storage[&StorageClassKey::LocalBlock].plateau().unwrap();
        assert_eq!(local.picked.files_in_flight, 4);
        assert!(p.storage[&StorageClassKey::LocalBlock].split_helps);
        assert!(!p.storage[&StorageClassKey::Nfs].split_helps);
        assert!(!p.storage[&StorageClassKey::Nfs].gds_registers());
        assert!(p.storage[&StorageClassKey::LocalBlock].gds_registers());
    }

    #[test]
    fn plateau_picks_smallest_within_tolerance() {
        let e = entry(1.0, &[(1, 2.0), (4, 9.0), (8, 9.5), (16, 10.0)], false);
        assert_eq!(e.plateau().unwrap().picked.files_in_flight, 4);
        let single = entry(1.0, &[(2, 5.0)], false);
        assert_eq!(single.plateau().unwrap().picked.files_in_flight, 2);
        // a table that falls off after its peak still picks the first row at the plateau
        let bump = entry(1.0, &[(1, 1.0), (2, 10.0), (4, 3.0)], false);
        assert_eq!(bump.plateau().unwrap().picked.files_in_flight, 2);
    }

    #[test]
    fn unknown_storage_class_key_is_named() {
        let text = r#"{"schema_version":1,"source":"t","storage":{"Floppy":{"single_file_gbps":1,"aggregate_gbps":[{"files_in_flight":1,"gbps":1}],"split_helps":false}}}"#;
        let err = Profile::from_json_str(text).unwrap_err();
        assert!(
            matches!(&err, ProfileError::UnknownStorageClass { key } if key == "Floppy"),
            "{err}"
        );
        assert!(err.to_string().contains("Floppy"), "{err}");
    }

    #[test]
    fn non_positive_rates_are_rejected() {
        let zero = profile_with(&[(StorageClassKey::Nfs, entry(0.0, &[(1, 1.0)], false))]);
        assert!(matches!(
            zero.validate().unwrap_err(),
            ProfileError::NonPositiveRate {
                field: "single_file_gbps",
                ..
            }
        ));
        let negative_row = profile_with(&[(
            StorageClassKey::Nfs,
            entry(1.0, &[(1, 1.0), (2, -3.0)], false),
        )]);
        assert!(matches!(
            negative_row.validate().unwrap_err(),
            ProfileError::NonPositiveRate {
                field: "aggregate_gbps.gbps",
                ..
            }
        ));
        let mut nan = profile_with(&[(StorageClassKey::Nfs, entry(1.0, &[(1, 1.0)], false))]);
        nan.device = Some(DeviceProfile {
            h2d_pinned_gbps: f64::NAN,
            h2d_pageable_gbps: 1.0,
            pinned_alloc_ms_per_mib: 0.0,
        });
        let err = nan.validate().unwrap_err();
        assert!(
            matches!(&err, ProfileError::NonPositiveRate { entry, field: "h2d_pinned_gbps", .. } if entry == "device"),
            "{err}"
        );
        let mut mount = profile_with(&[]);
        mount
            .mounts
            .insert("/mnt/x".into(), entry(1.0, &[(1, 1.0)], false));
        mount
            .mounts
            .get_mut(Path::new("/mnt/x"))
            .unwrap()
            .open_cost_ms = Some(-1.0);
        let err = mount.validate().unwrap_err();
        assert!(
            matches!(&err, ProfileError::NonPositiveRate { entry, field: "open_cost_ms", .. } if entry.contains("/mnt/x")),
            "{err}"
        );
        let mut io = profile_with(&[(StorageClassKey::Nfs, entry(1.0, &[(1, 1.0)], false))]);
        io.storage
            .get_mut(&StorageClassKey::Nfs)
            .unwrap()
            .io_cost_us = Some(-0.5);
        assert!(matches!(
            io.validate().unwrap_err(),
            ProfileError::NonPositiveRate {
                field: "io_cost_us",
                ..
            }
        ));
    }

    #[test]
    fn unsorted_and_empty_tables_are_rejected() {
        let unsorted = profile_with(&[(
            StorageClassKey::Nfs,
            entry(1.0, &[(4, 1.0), (2, 2.0)], false),
        )]);
        assert!(matches!(
            unsorted.validate().unwrap_err(),
            ProfileError::AggregateNotSorted {
                index: 1,
                files_in_flight: 2,
                ..
            }
        ));
        let duplicate = profile_with(&[(
            StorageClassKey::Nfs,
            entry(1.0, &[(4, 1.0), (4, 2.0)], false),
        )]);
        assert!(matches!(
            duplicate.validate().unwrap_err(),
            ProfileError::AggregateNotSorted { index: 1, .. }
        ));
        let zero_files = profile_with(&[(StorageClassKey::Nfs, entry(1.0, &[(0, 1.0)], false))]);
        assert!(matches!(
            zero_files.validate().unwrap_err(),
            ProfileError::AggregateNotSorted { index: 0, .. }
        ));
        let empty = profile_with(&[(StorageClassKey::Nfs, entry(1.0, &[], false))]);
        assert!(matches!(
            empty.validate().unwrap_err(),
            ProfileError::EmptyTable {
                table: "aggregate_gbps",
                ..
            }
        ));
    }

    #[test]
    fn schema_version_and_unknown_fields_are_rejected() {
        let text = r#"{"schema_version":2,"source":"t"}"#;
        assert!(matches!(
            Profile::from_json_str(text).unwrap_err(),
            ProfileError::UnsupportedSchemaVersion {
                found: 2,
                supported: SCHEMA_VERSION
            }
        ));
        let text = r#"{"schema_version":1,"source":"t","extra":1}"#;
        assert!(matches!(
            Profile::from_json_str(text).unwrap_err(),
            ProfileError::Json(_)
        ));
        let text = r#"{"schema_version":1,"source":"t","storage":{"Nfs":{"single_file_gbps":1,"aggregate_gbps":[{"files_in_flight":1,"gbps":1}],"split_helps":false,"bogus":1}}}"#;
        assert!(matches!(
            Profile::from_json_str(text).unwrap_err(),
            ProfileError::Json(_)
        ));
        assert!(matches!(
            Profile::from_path(Path::new("/nonexistent/profile.json")).unwrap_err(),
            ProfileError::Io { .. }
        ));
    }

    #[test]
    fn optional_fields_may_be_omitted() {
        let text = r#"{"schema_version":1,"source":"minimal"}"#;
        let p = Profile::from_json_str(text).unwrap();
        assert!(p.storage.is_empty() && p.mounts.is_empty() && p.device.is_none());
        let text = r#"{"schema_version":1,"source":"t","storage":{"Nfs":{"single_file_gbps":1,"aggregate_gbps":[{"files_in_flight":1,"gbps":1}],"split_helps":false}}}"#;
        let p = Profile::from_json_str(text).unwrap();
        let nfs = &p.storage[&StorageClassKey::Nfs];
        assert!(nfs.page_cache_gbps.is_none() && nfs.open_cost_ms.is_none() && nfs.gds.is_none());
        assert!(nfs.io_cost_us.is_none());
        assert!(!nfs.gds_registers());
    }

    #[test]
    fn merge_replaces_per_section() {
        let mut base = Profile::default_measured();
        base.mounts
            .insert("/base".into(), entry(1.0, &[(1, 1.0)], false));
        let mut over = profile_with(&[(StorageClassKey::Nfs, entry(9.0, &[(1, 9.0)], true))]);
        over.source = "over".into();
        over.mounts
            .insert("/over".into(), entry(2.0, &[(1, 2.0)], true));
        let merged = Profile::merge(&base, &over);
        assert_eq!(merged.storage[&StorageClassKey::Nfs].single_file_gbps, 9.0);
        assert_eq!(
            merged.storage[&StorageClassKey::LocalBlock],
            base.storage[&StorageClassKey::LocalBlock]
        );
        assert_eq!(merged.storage.len(), base.storage.len());
        assert_eq!(merged.mounts.len(), 2);
        assert_eq!(
            merged.device, base.device,
            "device stays when over has none"
        );
        assert!(merged.source.starts_with("over over "), "{}", merged.source);
        merged.validate().unwrap();

        over.device = Some(DeviceProfile {
            h2d_pinned_gbps: 1.0,
            h2d_pageable_gbps: 1.0,
            pinned_alloc_ms_per_mib: 1.0,
        });
        assert_eq!(Profile::merge(&base, &over).device, over.device);
    }

    #[test]
    fn lookup_prefers_longest_mount_then_class_then_fallback() {
        let env = Env::new(
            vec![
                Mount {
                    point: "/".into(),
                    fs_type: "ext4".into(),
                },
                Mount {
                    point: "/mnt/nfs".into(),
                    fs_type: "nfs".into(),
                },
                Mount {
                    point: "/mnt/lustre".into(),
                    fs_type: "lustre".into(),
                },
            ],
            Gds::default(),
            8,
        );
        let mut p = Profile::default_measured();
        p.mounts
            .insert("/mnt/nfs".into(), entry(1.0, &[(1, 1.0)], false));
        p.mounts
            .insert("/mnt/nfs/fast".into(), entry(50.0, &[(1, 50.0)], true));

        let short = p.lookup(&env, Path::new("/mnt/nfs/user/x"));
        assert_eq!(short.label, "mount \"/mnt/nfs\"");
        assert_eq!(short.entry.single_file_gbps, 1.0);
        assert!(short.from_profile);
        let long = p.lookup(&env, Path::new("/mnt/nfs/fast/x"));
        assert_eq!(long.label, "mount \"/mnt/nfs/fast\"");
        assert_eq!(long.entry.single_file_gbps, 50.0);
        let class = p.lookup(&env, Path::new("/etc/x"));
        assert_eq!(class.label, "LocalBlock");
        assert_eq!(class.class, StorageClass::LocalBlock);
        let fallback = p.lookup(&env, Path::new("/mnt/lustre/x"));
        assert!(!fallback.from_profile);
        assert_eq!(fallback.entry, StorageProfile::conservative());
        assert!(
            fallback.label.contains("OtherNetwork(\"lustre\")"),
            "{}",
            fallback.label
        );
        assert!(
            fallback.label.contains("conservative"),
            "{}",
            fallback.label
        );
    }

    #[test]
    fn every_key_parses_and_maps() {
        for key in StorageClassKey::ALL {
            assert_eq!(StorageClassKey::parse(key.name()).unwrap(), key);
            assert_eq!(key.to_string(), key.name());
        }
        assert_eq!(
            StorageClassKey::from(&StorageClass::OtherNetwork("lustre".into())),
            StorageClassKey::OtherNetwork
        );
        assert_eq!(
            StorageClassKey::from(&StorageClass::Other(String::new())),
            StorageClassKey::Other
        );
    }

    /// A valid entry: positive rates, a strictly increasing table.
    pub(crate) fn arb_entry() -> impl Strategy<Value = StorageProfile> {
        (
            0.1f64..100.0,
            proptest::collection::btree_set(1u32..=64, 1..6),
            proptest::collection::vec(0.1f64..100.0, 6),
            any::<bool>(),
            proptest::option::of(0.1f64..100.0),
            proptest::option::of(0.0f64..10.0),
            proptest::option::of(0.0f64..5000.0),
            proptest::option::of((any::<bool>(), proptest::option::of(0.1f64..100.0))),
        )
            .prop_map(
                |(single, counts, rates, split, cache, open, io_cost, gds)| StorageProfile {
                    single_file_gbps: single,
                    aggregate_gbps: counts
                        .into_iter()
                        .zip(rates)
                        .map(|(files_in_flight, gbps)| AggregatePoint {
                            files_in_flight,
                            gbps,
                        })
                        .collect(),
                    split_helps: split,
                    page_cache_gbps: cache,
                    open_cost_ms: open,
                    io_cost_us: io_cost,
                    gds: gds.map(|(registers, read_gbps)| GdsProfile {
                        registers,
                        read_gbps,
                    }),
                },
            )
    }

    /// A valid profile: any subset of classes, up to two mount overrides.
    pub(crate) fn arb_profile() -> impl Strategy<Value = Profile> {
        (
            proptest::collection::btree_map(
                proptest::sample::select(StorageClassKey::ALL.to_vec()),
                arb_entry(),
                0..=6,
            ),
            proptest::collection::btree_map(
                proptest::sample::select(vec![PathBuf::from("/data"), PathBuf::from("/data/fast")]),
                arb_entry(),
                0..=2,
            ),
            proptest::option::of((0.1f64..100.0, 0.1f64..100.0, 0.0f64..10.0)),
            "[a-z0-9-]{1,12}",
        )
            .prop_map(|(storage, mounts, device, source)| Profile {
                schema_version: SCHEMA_VERSION,
                source,
                storage,
                mounts,
                device: device.map(|(pinned, pageable, alloc)| DeviceProfile {
                    h2d_pinned_gbps: pinned,
                    h2d_pageable_gbps: pageable,
                    pinned_alloc_ms_per_mib: alloc,
                }),
            })
    }

    proptest! {
        #[test]
        fn valid_profiles_validate_and_round_trip(p in arb_profile()) {
            p.validate().unwrap_or_else(|e| panic!("{e}"));
            let text = p.to_json_string().unwrap_or_else(|e| panic!("{e}"));
            let back = Profile::from_json_str(&text).unwrap_or_else(|e| panic!("{e}"));
            prop_assert_eq!(back, p);
        }

        #[test]
        fn plateau_is_a_table_row_within_tolerance(e in arb_entry()) {
            let plateau = e.plateau().unwrap_or_else(|| panic!("non-empty table"));
            prop_assert!(e.aggregate_gbps.contains(&plateau.picked));
            prop_assert!(plateau.picked.gbps >= plateau.max.gbps * (1.0 - PLATEAU_TOLERANCE));
            prop_assert!(plateau.picked.files_in_flight <= plateau.max.files_in_flight);
            // nothing smaller qualifies
            for row in &e.aggregate_gbps {
                if row.files_in_flight < plateau.picked.files_in_flight {
                    prop_assert!(row.gbps < plateau.max.gbps * (1.0 - PLATEAU_TOLERANCE));
                }
            }
        }

        #[test]
        fn merge_is_base_where_over_is_silent(base in arb_profile(), over in arb_profile()) {
            let merged = Profile::merge(&base, &over);
            merged.validate().unwrap_or_else(|e| panic!("{e}"));
            for (k, v) in &merged.storage {
                let expected = over.storage.get(k).or_else(|| base.storage.get(k));
                prop_assert_eq!(Some(v), expected);
            }
            for k in base.storage.keys().chain(over.storage.keys()) {
                prop_assert!(merged.storage.contains_key(k));
            }
            prop_assert_eq!(merged.device, over.device.or(base.device));
            prop_assert!(merged.source.contains(&over.source) && merged.source.contains(&base.source));
        }
    }
}
