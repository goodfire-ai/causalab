//! The decision, as a pure function.
//!
//! [`plan_read`] takes an [`Env`], a [`Profile`] and a [`ReadRequest`] and
//! returns a [`ReadPlan`]: how many files to keep in flight, how to split
//! each, which transport moves the bytes, and what staging sits between
//! storage and a device destination. The environment says what the machine
//! *has*; the profile says what it *measured*; the rules here only combine
//! the two, and every reason line names the profile entry it read, so
//! `explain()` is the rules applied to this machine — not a paraphrase.
//!
//! The rules:
//!
//! * **Files in flight** is the smallest count in the entry's aggregate table
//!   within [`PLATEAU_TOLERANCE`] of the table's maximum, capped at the files
//!   requested. The built-in profile's NFS table plateaus at 16.
//! * **Splitting** a file across readers happens only where the entry says
//!   `split_helps`; then `cpus` readers (at most 16) over
//!   [`LOCAL_SPLIT_BYTES`] pieces. Otherwise files are read whole, one reader
//!   each — a file capped by the mount gains nothing from more readers.
//! * **cuFile** needs `nvidia_fs` and `libcufile` in the environment *and*
//!   `gds.registers` in every entry the request touches. Otherwise a device
//!   destination stages through pinned buffers, two per concurrent reader.
//! * A **mixed** request (files on several entries) is planned for its least
//!   scalable entry: one that does not split, then the lowest single-file
//!   rate.
//! * A class the profile does not describe gets
//!   [`StorageProfile::conservative`] — the NFS shape — and the reason says so.
//! * **Coalescing** of a selection's runs ([`coalesce_policy`]): two runs
//!   are read as one when the gap between them is at most what the entry
//!   moves in one read call's fixed cost, `io_cost_us × single_file_gbps`,
//!   clamped to [`MIN_COALESCE_GAP_BYTES`]..=[`MAX_COALESCE_GAP_BYTES`], and
//!   a read never grows past one staging buffer. The request may carry the
//!   [`CoalesceSummary`] of applying that rule, and `explain()` prints it.

use std::path::PathBuf;

use crate::env::{Env, StorageClass};
use crate::error::PlanError;
use crate::profile::{PLATEAU_TOLERANCE, Profile, StorageClassKey, StorageLookup, StorageProfile};
use crate::select::{CoalescePolicy, CoalesceSummary};

/// One file the request reads: where it is and how much of it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileSpec {
    /// The file.
    pub path: PathBuf,
    /// Bytes the request will read from it (not the file's size — only
    /// wanted tensors are read).
    pub wanted_bytes: u64,
}

/// Where the bytes end up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Destination {
    /// Host memory the caller owns.
    Host,
    /// A CUDA device, with available memory reported by the caller.
    Device {
        /// Ordinal.
        device: u32,
        /// Aggregate headroom at planning time; may include reusable allocator cache.
        /// This does not guarantee that a contiguous allocation fits.
        free_bytes: u64,
    },
}

/// What a caller wants read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadRequest {
    /// Files, each with the bytes wanted from it.
    pub files: Vec<FileSpec>,
    /// Peak device allocations: results plus scratch, excluding host staging.
    /// `None` preserves the whole-read byte count for callers without selections.
    pub allocation_bytes: Option<u64>,
    /// Where tensors land.
    pub destination: Destination,
    /// What coalescing the request's selections under [`coalesce_policy`]
    /// did, when the caller built the job from selections; `None` for whole
    /// tensors. Read only for the explanation — the policy itself comes
    /// from the profile.
    pub coalesced: Option<CoalesceSummary>,
}

/// How bytes leave storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transport {
    /// `pread` into the destination (host) or a staging buffer (device).
    Pread,
    /// Map the file and copy out of the mapping.
    Mmap,
    /// GPUDirect Storage: `cuFileRead` straight into device memory.
    CuFile,
}

/// What sits between storage and a device destination.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Staging {
    /// None: host destination, or GDS.
    None,
    /// Pinned host buffers, `count` of `bytes` each, filled by the transport
    /// and drained by asynchronous device copies.
    Pinned {
        /// Buffers in the ring.
        count: usize,
        /// Bytes per buffer.
        bytes: u64,
    },
}

/// The plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadPlan {
    /// Files read concurrently.
    pub files_in_flight: usize,
    /// Bytes per read piece within a file; zero means whole ranges, unsplit.
    pub split_bytes: u64,
    /// Concurrent reads within one file (pieces in flight).
    pub readers_per_file: usize,
    /// The transport.
    pub transport: Transport,
    /// The staging.
    pub staging: Staging,
    /// Why, one line per decision.
    pub reasons: Vec<String>,
    /// The [`Profile::source`] the decisions were read from.
    pub profile_source: String,
}

impl ReadPlan {
    /// The reasons, one per line.
    pub fn explain(&self) -> String {
        self.reasons.join("\n")
    }

    /// Total concurrent reads the plan issues.
    pub fn concurrency(&self) -> usize {
        self.files_in_flight * self.readers_per_file
    }
}

/// Pinned staging: 16 MiB buffers, the size fastsafetensors settled on for
/// its bounce buffers and large enough that PCIe transfers run at rate.
pub const STAGING_BUFFER_BYTES: u64 = 16 * 1024 * 1024;
/// Piece size on entries where one file does scale with readers.
pub const LOCAL_SPLIT_BYTES: u64 = 64 * 1024 * 1024;
/// Most readers one file is split across; beyond the CPU count they only
/// contend.
pub const MAX_READERS_PER_FILE: usize = 16;
/// Smallest gap [`coalesce_policy`] reads through, whatever the profile
/// says: 64 KiB is 21 µs at the measured 3 GB/s, less than any read call
/// costs on any storage, so skipping a smaller gap can never pay. Also the
/// gap for an entry whose `io_cost_us` is unmeasured.
pub const MIN_COALESCE_GAP_BYTES: u64 = 64 * 1024;
/// Largest gap [`coalesce_policy`] reads through, whatever the profile
/// says: half a staging buffer, so a slow mount's large `io_cost_us` cannot
/// turn a sparse selection into reading the whole tensor.
pub const MAX_COALESCE_GAP_BYTES: u64 = 8 * 1024 * 1024;

/// The coalescing rule for one profile entry: runs whose gap is at most
/// `io_cost_us × single_file_gbps` — the bytes the mount moves in the time
/// one read call costs regardless of size, so reading through the gap is
/// never dearer than a second call — are read as one, with the gap clamped
/// to [`MIN_COALESCE_GAP_BYTES`]..=[`MAX_COALESCE_GAP_BYTES`] and the floor
/// used when `io_cost_us` is unmeasured. A read never grows past
/// [`STAGING_BUFFER_BYTES`], the buffer a device piece must fit.
pub fn coalesce_policy_for(entry: &StorageProfile) -> CoalescePolicy {
    // µs × GB/s = 1e-6 s × 1e9 B/s = 1e3 B
    let gap = match entry.io_cost_us {
        Some(us) if us.is_finite() => (us * entry.single_file_gbps * 1000.0).round(),
        _ => 0.0,
    };
    // a NaN or negative product saturates to zero and is then floored
    let gap = if gap.is_finite() && gap > 0.0 {
        gap.min(MAX_COALESCE_GAP_BYTES as f64) as u64
    } else {
        0
    };
    CoalescePolicy {
        max_gap_bytes: gap.clamp(MIN_COALESCE_GAP_BYTES, MAX_COALESCE_GAP_BYTES),
        max_read_bytes: STAGING_BUFFER_BYTES,
    }
}

/// [`coalesce_policy_for`] the profile's entry for a storage class — the
/// conservative shape when it has none. The one rule the Python layer and
/// the read-job builder both apply, so what `explain()` prints is what the
/// engine reads. A caller with a path should prefer
/// [`Profile::lookup`], which also honours mount overrides, and pass its
/// entry to [`coalesce_policy_for`].
pub fn coalesce_policy(profile: &Profile, class: &StorageClass) -> CoalescePolicy {
    match profile.storage.get(&StorageClassKey::from(class)) {
        Some(entry) => coalesce_policy_for(entry),
        None => coalesce_policy_for(&StorageProfile::conservative()),
    }
}

/// Decide how to read.
pub fn plan_read(
    env: &Env,
    profile: &Profile,
    request: &ReadRequest,
) -> Result<ReadPlan, PlanError> {
    if request.files.is_empty() {
        return Err(PlanError::Empty);
    }
    let needed = request
        .allocation_bytes
        .unwrap_or_else(|| request.files.iter().map(|f| f.wanted_bytes).sum());
    let mut reasons = Vec::new();

    if let Destination::Device { free_bytes, .. } = request.destination
        && free_bytes < needed
    {
        return Err(PlanError::DoesNotFit {
            available: free_bytes,
            needed,
        });
    }

    let source = &profile.source;
    let lookups: Vec<StorageLookup> = request
        .files
        .iter()
        .map(|f| profile.lookup(env, &f.path))
        .collect();
    let governing = least_scalable(&lookups);
    let entry = &governing.entry;
    let label = &governing.label;
    let mixed = distinct_labels(&lookups);
    if mixed.len() > 1 {
        reasons.push(format!(
            "files span {}; planned for {label} (least scalable: {})",
            mixed.join(" + "),
            if entry.split_helps {
                "lowest single-file rate"
            } else {
                "one file does not scale with readers"
            }
        ));
    }
    if !governing.from_profile {
        reasons.push(format!(
            "profile {source:?} has no entry for {:?}: using conservative NFS-shaped defaults \
             (one file {} GB/s, plateau at {} files)",
            governing.class,
            entry.single_file_gbps,
            entry
                .plateau()
                .map(|p| p.picked.files_in_flight)
                .unwrap_or(1)
        ));
    }

    // files in flight: the plateau of the aggregate table, capped at the files given
    let plateau = entry.plateau();
    let plateau_files = plateau
        .map(|p| usize::try_from(p.picked.files_in_flight).unwrap_or(usize::MAX))
        .unwrap_or(1)
        .max(1);
    let files_in_flight = request.files.len().min(plateau_files);
    match plateau {
        Some(p) if p.picked.files_in_flight == p.max.files_in_flight => reasons.push(format!(
            "profile {source:?} {label}.aggregate_gbps: {} files → {} GB/s (table maximum); \
             {files_in_flight} of {} files in flight",
            p.picked.files_in_flight,
            p.picked.gbps,
            request.files.len()
        )),
        Some(p) => reasons.push(format!(
            "profile {source:?} {label}.aggregate_gbps: {} files → {} GB/s (plateau {} at {}; within {}%); \
             {files_in_flight} of {} files in flight",
            p.picked.files_in_flight,
            p.picked.gbps,
            p.max.gbps,
            p.max.files_in_flight,
            (PLATEAU_TOLERANCE * 100.0).round(),
            request.files.len()
        )),
        None => reasons.push(format!(
            "profile {source:?} {label}.aggregate_gbps is empty; 1 file in flight"
        )),
    }

    // split: only where the entry says one file scales with readers
    let (split_bytes, readers_per_file) = if entry.split_helps {
        let readers = env.cpus.clamp(1, MAX_READERS_PER_FILE);
        reasons.push(format!(
            "profile {source:?} {label}.split_helps: one file scales with readers; \
             {readers} readers per file over {} MiB pieces (cpus={})",
            LOCAL_SPLIT_BYTES >> 20,
            env.cpus
        ));
        (LOCAL_SPLIT_BYTES, readers)
    } else {
        reasons.push(format!(
            "profile {source:?} {label}.single_file_gbps: one file caps near {} GB/s however split; \
             each file unsplit, one reader",
            entry.single_file_gbps
        ));
        (0, 1)
    };

    let all_register = lookups.iter().all(|l| l.entry.gds_registers());
    let (transport, staging) = match request.destination {
        Destination::Host => {
            reasons.push("host destination: pread straight into the caller's buffers".into());
            (Transport::Pread, Staging::None)
        }
        Destination::Device { .. } if env.gds.available() && all_register => {
            let rate = match entry.gds.and_then(|g| g.read_gbps) {
                Some(gbps) => format!("{gbps} GB/s"),
                None => "unmeasured".to_owned(),
            };
            reasons.push(format!(
                "device destination, nvidia_fs loaded, libcufile at {}, profile {source:?} \
                 {label}.gds.registers: cuFileRead, no staging (gds.read_gbps {rate})",
                env.gds
                    .cufile_library
                    .as_deref()
                    .map(|p| p.display().to_string())
                    .unwrap_or_default()
            ));
            (Transport::CuFile, Staging::None)
        }
        Destination::Device { .. } => {
            let why = if !env.gds.module_loaded {
                "nvidia_fs module not loaded".to_owned()
            } else if env.gds.cufile_library.is_none() {
                "libcufile not found".to_owned()
            } else {
                let blocker = lookups
                    .iter()
                    .find(|l| !l.entry.gds_registers())
                    .map_or(label.as_str(), |l| l.label.as_str());
                format!(
                    "profile {source:?} {blocker}.gds.registers is not true: storage is not a GDS-capable mount"
                )
            };
            let count = (files_in_flight * readers_per_file * 2).clamp(2, 64);
            let h2d = match profile.device {
                Some(d) => format!(
                    " (profile device.h2d_pinned_gbps {} GB/s, pinned_alloc {} ms/MiB)",
                    d.h2d_pinned_gbps, d.pinned_alloc_ms_per_mib
                ),
                None => String::new(),
            };
            reasons.push(format!(
                "device destination, {why}: pread into {count} pinned {} MiB staging buffers, async copies onward{h2d}",
                STAGING_BUFFER_BYTES >> 20
            ));
            (
                Transport::Pread,
                Staging::Pinned {
                    count,
                    bytes: STAGING_BUFFER_BYTES,
                },
            )
        }
    };

    if let Some(summary) = &request.coalesced
        && summary.coalesced_anything()
    {
        let policy = coalesce_policy_for(entry);
        let provenance = match entry.io_cost_us {
            Some(us) => format!(
                "profile {source:?} {label}.io_cost_us {us} µs × single_file_gbps {} GB/s",
                entry.single_file_gbps
            ),
            None => format!("profile {source:?} {label}.io_cost_us unmeasured, floor"),
        };
        reasons.push(format!(
            "selections: {} runs coalesced into {} reads, amplification x{:.2} \
             (gaps up to {} KiB from {provenance}; reads up to {} MiB)",
            summary.runs,
            summary.reads,
            summary.amplification(),
            policy.max_gap_bytes >> 10,
            policy.max_read_bytes >> 20
        ));
    }

    Ok(ReadPlan {
        files_in_flight,
        split_bytes,
        readers_per_file,
        transport,
        staging,
        reasons,
        profile_source: profile.source.clone(),
    })
}

/// The entry a mixed request is planned for: one that does not split before
/// any that does, then the lowest single-file rate, then the first given.
fn least_scalable(lookups: &[StorageLookup]) -> &StorageLookup {
    let mut best = &lookups[0];
    for candidate in &lookups[1..] {
        let (b, c): (&StorageProfile, &StorageProfile) = (&best.entry, &candidate.entry);
        let worse = (b.split_helps && !c.split_helps)
            || (b.split_helps == c.split_helps && c.single_file_gbps < b.single_file_gbps);
        if worse {
            best = candidate;
        }
    }
    best
}

fn distinct_labels(lookups: &[StorageLookup]) -> Vec<String> {
    let mut names: Vec<String> = lookups.iter().map(|l| l.label.clone()).collect();
    names.sort();
    names.dedup();
    names
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::{Gds, Mount};
    use crate::profile::{AggregatePoint, GdsProfile, StorageClassKey};
    use proptest::prelude::*;

    fn env_with(fs_type: &str, gds: Gds, cpus: usize) -> Env {
        Env::new(
            vec![Mount {
                point: "/".into(),
                fs_type: fs_type.into(),
            }],
            gds,
            cpus,
        )
    }

    fn files(n: usize, bytes: u64) -> Vec<FileSpec> {
        (0..n)
            .map(|i| FileSpec {
                path: format!("/data/shard-{i}").into(),
                wanted_bytes: bytes,
            })
            .collect()
    }

    fn defaults() -> Profile {
        Profile::default_measured()
    }

    fn entry(
        single: f64,
        table: &[(u32, f64)],
        split_helps: bool,
        registers: Option<bool>,
    ) -> StorageProfile {
        StorageProfile {
            single_file_gbps: single,
            aggregate_gbps: table
                .iter()
                .map(|&(files_in_flight, gbps)| AggregatePoint {
                    files_in_flight,
                    gbps,
                })
                .collect(),
            split_helps,
            page_cache_gbps: None,
            open_cost_ms: None,
            io_cost_us: None,
            gds: registers.map(|registers| GdsProfile {
                registers,
                read_gbps: None,
            }),
        }
    }

    const GDS: Gds = Gds {
        module_loaded: true,
        cufile_library: None,
    };

    fn gds_present() -> Gds {
        Gds {
            cufile_library: Some("/usr/local/cuda/lib64/libcufile.so".into()),
            ..GDS
        }
    }

    const DEVICE: Destination = Destination::Device {
        device: 0,
        free_bytes: 1 << 40,
    };

    #[test]
    fn nfs_keeps_many_files_in_flight_unsplit() {
        let env = env_with("nfs", Gds::default(), 16);
        let plan = plan_read(
            &env,
            &defaults(),
            &ReadRequest {
                files: files(26, 1 << 30),
                destination: Destination::Host,
                allocation_bytes: None,
                coalesced: None,
            },
        )
        .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(plan.files_in_flight, 16);
        assert_eq!(plan.split_bytes, 0);
        assert_eq!(plan.readers_per_file, 1);
        assert_eq!(plan.transport, Transport::Pread);
        assert_eq!(plan.staging, Staging::None);
        let text = plan.explain();
        assert!(text.contains("caps near 3 GB/s"), "{text}");
        assert!(
            text.contains(
                "Nfs.aggregate_gbps: 16 files → 8.3 GB/s (plateau 9.7 at 32; within 15%)"
            ),
            "{text}"
        );
        assert!(
            text.contains(&format!("profile {:?}", defaults().source)),
            "{text}"
        );
        assert_eq!(plan.profile_source, defaults().source);
    }

    #[test]
    fn local_splits_files_across_readers() {
        let env = env_with("xfs", Gds::default(), 8);
        let plan = plan_read(
            &env,
            &defaults(),
            &ReadRequest {
                files: files(2, 1 << 30),
                destination: Destination::Host,
                allocation_bytes: None,
                coalesced: None,
            },
        )
        .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(plan.split_bytes, LOCAL_SPLIT_BYTES);
        assert_eq!(plan.readers_per_file, 8);
        assert_eq!(plan.files_in_flight, 2);
        assert!(
            plan.explain().contains("LocalBlock.split_helps"),
            "{}",
            plan.explain()
        );
        let many = plan_read(
            &env,
            &defaults(),
            &ReadRequest {
                files: files(26, 1 << 30),
                destination: Destination::Host,
                allocation_bytes: None,
                coalesced: None,
            },
        )
        .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            many.files_in_flight, 4,
            "the local-file rule: four local files in flight"
        );
    }

    #[test]
    fn gds_only_with_module_library_and_local_storage() {
        let local = plan_read(
            &env_with("ext4", gds_present(), 8),
            &defaults(),
            &ReadRequest {
                files: files(1, 1 << 20),
                destination: DEVICE,
                allocation_bytes: None,
                coalesced: None,
            },
        )
        .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(local.transport, Transport::CuFile);
        assert_eq!(local.staging, Staging::None);
        assert!(
            local.explain().contains("LocalBlock.gds.registers"),
            "{}",
            local.explain()
        );
        let nfs = plan_read(
            &env_with("nfs", gds_present(), 8),
            &defaults(),
            &ReadRequest {
                files: files(1, 1 << 20),
                destination: DEVICE,
                allocation_bytes: None,
                coalesced: None,
            },
        )
        .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(nfs.transport, Transport::Pread);
        assert!(matches!(nfs.staging, Staging::Pinned { .. }));
        assert!(nfs.explain().contains("not a GDS-capable"));
        assert!(
            nfs.explain().contains("Nfs.gds.registers"),
            "{}",
            nfs.explain()
        );
        assert!(
            nfs.explain().contains("device.h2d_pinned_gbps 55 GB/s"),
            "{}",
            nfs.explain()
        );
        let no_module = plan_read(
            &env_with(
                "ext4",
                Gds {
                    module_loaded: false,
                    cufile_library: Some("/x".into()),
                },
                8,
            ),
            &defaults(),
            &ReadRequest {
                files: files(1, 1 << 20),
                destination: DEVICE,
                allocation_bytes: None,
                coalesced: None,
            },
        )
        .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(no_module.transport, Transport::Pread);
        assert!(no_module.explain().contains("nvidia_fs module not loaded"));
        let no_library = plan_read(
            &env_with("ext4", GDS, 8),
            &defaults(),
            &ReadRequest {
                files: files(1, 1 << 20),
                destination: DEVICE,
                allocation_bytes: None,
                coalesced: None,
            },
        )
        .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(no_library.transport, Transport::Pread);
        assert!(no_library.explain().contains("libcufile not found"));
    }

    #[test]
    fn profile_plateau_at_four_yields_four_in_flight() {
        let mut profile = defaults();
        profile.source = "four".into();
        profile.storage.insert(
            StorageClassKey::Nfs,
            entry(
                3.0,
                &[(1, 2.6), (4, 7.4), (16, 7.6), (32, 7.7)],
                false,
                Some(false),
            ),
        );
        let plan = plan_read(
            &env_with("nfs", Gds::default(), 16),
            &profile,
            &ReadRequest {
                files: files(26, 1 << 30),
                destination: Destination::Host,
                allocation_bytes: None,
                coalesced: None,
            },
        )
        .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(plan.files_in_flight, 4);
        assert_eq!(plan.readers_per_file, 1);
        assert!(
            plan.explain().contains(
                "profile \"four\" Nfs.aggregate_gbps: 4 files → 7.4 GB/s (plateau 7.7 at 32"
            ),
            "{}",
            plan.explain()
        );
        // fewer files than the plateau: all of them
        let two = plan_read(
            &env_with("nfs", Gds::default(), 16),
            &profile,
            &ReadRequest {
                files: files(2, 1 << 30),
                destination: Destination::Host,
                allocation_bytes: None,
                coalesced: None,
            },
        )
        .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(two.files_in_flight, 2);
        assert!(
            two.explain().contains("2 of 2 files in flight"),
            "{}",
            two.explain()
        );
    }

    #[test]
    fn mount_override_beats_its_class() {
        let mut profile = defaults();
        profile.mounts.insert(
            "/data".into(),
            entry(12.0, &[(1, 12.0), (2, 12.5)], true, Some(false)),
        );
        let env = env_with("nfs", Gds::default(), 8);
        let request = ReadRequest {
            files: files(26, 1 << 30),
            destination: Destination::Host,
            allocation_bytes: None,
            coalesced: None,
        };
        let plan = plan_read(&env, &profile, &request).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            plan.files_in_flight, 1,
            "the mount's table plateaus at one file"
        );
        assert_eq!(plan.split_bytes, LOCAL_SPLIT_BYTES);
        assert_eq!(plan.readers_per_file, 8);
        assert!(
            plan.explain().contains("mount \"/data\".aggregate_gbps"),
            "{}",
            plan.explain()
        );
        // the same class without the override plans as NFS
        let class_plan = plan_read(&env, &defaults(), &request).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(class_plan.files_in_flight, 16);
        assert_eq!(class_plan.split_bytes, 0);
        // a file outside the mount is unaffected
        let outside = plan_read(
            &env,
            &profile,
            &ReadRequest {
                files: vec![FileSpec {
                    path: "/elsewhere/shard".into(),
                    wanted_bytes: 1,
                }],
                destination: Destination::Host,
                allocation_bytes: None,
                coalesced: None,
            },
        )
        .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(outside.split_bytes, 0);
    }

    #[test]
    fn registers_false_blocks_cufile_despite_module_and_library() {
        let mut profile = defaults();
        profile.storage.insert(
            StorageClassKey::LocalBlock,
            entry(3.0, &[(1, 2.6), (4, 7.4)], true, Some(false)),
        );
        let request = ReadRequest {
            files: files(1, 1 << 20),
            destination: DEVICE,
            allocation_bytes: None,
            coalesced: None,
        };
        let env = env_with("ext4", gds_present(), 8);
        let plan = plan_read(&env, &profile, &request).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(plan.transport, Transport::Pread);
        assert!(matches!(plan.staging, Staging::Pinned { .. }));
        assert!(
            plan.explain()
                .contains("LocalBlock.gds.registers is not true"),
            "{}",
            plan.explain()
        );
        // no gds entry at all is also "does not register"
        profile.storage.insert(
            StorageClassKey::LocalBlock,
            entry(3.0, &[(1, 2.6), (4, 7.4)], true, None),
        );
        let plan = plan_read(&env, &profile, &request).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(plan.transport, Transport::Pread);
        // and a mount override with registers: false blocks a registering class
        let mut over = defaults();
        over.mounts.insert(
            "/data".into(),
            entry(3.0, &[(1, 2.6), (4, 7.4)], true, Some(false)),
        );
        let plan = plan_read(&env, &over, &request).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(plan.transport, Transport::Pread);
        assert!(
            plan.explain().contains("mount \"/data\".gds.registers"),
            "{}",
            plan.explain()
        );
        // the profile's read_gbps rides along when cuFile is chosen
        let mut fast = defaults();
        fast.storage.insert(
            StorageClassKey::LocalBlock,
            StorageProfile {
                gds: Some(GdsProfile {
                    registers: true,
                    read_gbps: Some(12.0),
                }),
                ..entry(3.0, &[(1, 2.6), (4, 7.4)], true, None)
            },
        );
        let plan = plan_read(&env, &fast, &request).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(plan.transport, Transport::CuFile);
        assert!(
            plan.explain().contains("gds.read_gbps 12 GB/s"),
            "{}",
            plan.explain()
        );
    }

    #[test]
    fn unprofiled_class_falls_back_to_conservative_defaults() {
        let env = env_with("lustre", gds_present(), 16);
        let plan = plan_read(
            &env,
            &defaults(),
            &ReadRequest {
                files: files(26, 1 << 30),
                destination: DEVICE,
                allocation_bytes: None,
                coalesced: None,
            },
        )
        .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(plan.files_in_flight, 16);
        assert_eq!(plan.split_bytes, 0);
        assert_eq!(
            plan.transport,
            Transport::Pread,
            "the fallback never registers"
        );
        let text = plan.explain();
        assert!(
            text.contains("no entry for OtherNetwork(\"lustre\")"),
            "{text}"
        );
        assert!(text.contains("conservative NFS-shaped defaults"), "{text}");
        assert!(text.contains("OtherNetwork(\"lustre\") (no profile entry; conservative NFS-shaped defaults).aggregate_gbps"), "{text}");
    }

    #[test]
    fn mixed_request_is_planned_for_the_least_scalable_entry() {
        let env = Env::new(
            vec![
                Mount {
                    point: "/".into(),
                    fs_type: "ext4".into(),
                },
                Mount {
                    point: "/nfs".into(),
                    fs_type: "nfs".into(),
                },
            ],
            gds_present(),
            8,
        );
        let plan = plan_read(
            &env,
            &defaults(),
            &ReadRequest {
                files: vec![
                    FileSpec {
                        path: "/local/a".into(),
                        wanted_bytes: 1,
                    },
                    FileSpec {
                        path: "/nfs/b".into(),
                        wanted_bytes: 1,
                    },
                ],
                destination: DEVICE,
                allocation_bytes: None,
                coalesced: None,
            },
        )
        .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(plan.split_bytes, 0, "NFS governs");
        assert_eq!(
            plan.transport,
            Transport::Pread,
            "one non-registering entry blocks cuFile"
        );
        assert!(
            plan.explain()
                .contains("files span LocalBlock + Nfs; planned for Nfs"),
            "{}",
            plan.explain()
        );
    }

    #[test]
    fn coalesce_policy_is_the_profile_gap_clamped() {
        // the built-in NFS placeholder: 200 µs × 3 GB/s = 600 KB
        let nfs = coalesce_policy(&defaults(), &crate::env::StorageClass::Nfs);
        assert_eq!(nfs.max_gap_bytes, 600_000);
        assert_eq!(nfs.max_read_bytes, STAGING_BUFFER_BYTES);
        // unmeasured: the floor
        let local = coalesce_policy(&defaults(), &crate::env::StorageClass::LocalBlock);
        assert_eq!(local.max_gap_bytes, MIN_COALESCE_GAP_BYTES);
        // no entry: the conservative NFS shape
        let lustre = coalesce_policy(
            &defaults(),
            &crate::env::StorageClass::OtherNetwork("lustre".into()),
        );
        assert_eq!(lustre, nfs);
        // clamped at both ends
        let mut e = entry(3.0, &[(1, 1.0)], false, None);
        e.io_cost_us = Some(1.0);
        assert_eq!(
            coalesce_policy_for(&e).max_gap_bytes,
            MIN_COALESCE_GAP_BYTES
        );
        e.io_cost_us = Some(1e9);
        assert_eq!(
            coalesce_policy_for(&e).max_gap_bytes,
            MAX_COALESCE_GAP_BYTES
        );
        e.io_cost_us = Some(f64::NAN);
        assert_eq!(
            coalesce_policy_for(&e).max_gap_bytes,
            MIN_COALESCE_GAP_BYTES
        );
        e.io_cost_us = Some(100.0);
        assert_eq!(coalesce_policy_for(&e).max_gap_bytes, 300_000);
    }

    #[test]
    fn explain_names_coalescing_when_the_request_did_any() {
        let env = env_with("nfs", Gds::default(), 16);
        let summary = crate::select::CoalesceSummary {
            runs: 7168,
            reads: 128,
            wanted_bytes: 33_030_144,
            read_bytes: 264_241_152,
        };
        let plan = plan_read(
            &env,
            &defaults(),
            &ReadRequest {
                files: files(1, 33_030_144),
                destination: Destination::Host,
                allocation_bytes: None,
                coalesced: Some(summary),
            },
        )
        .unwrap_or_else(|e| panic!("{e}"));
        let text = plan.explain();
        assert!(
            text.contains("selections: 7168 runs coalesced into 128 reads, amplification x8.00"),
            "{text}"
        );
        assert!(text.contains("gaps up to 585 KiB from profile"), "{text}");
        assert!(
            text.contains("Nfs.io_cost_us 200 µs × single_file_gbps 3 GB/s"),
            "{text}"
        );
        assert!(text.contains("reads up to 16 MiB"), "{text}");
        // nothing coalesced: no line
        let plain = plan_read(
            &env,
            &defaults(),
            &ReadRequest {
                files: files(1, 8),
                destination: Destination::Host,
                allocation_bytes: None,
                coalesced: Some(crate::select::CoalesceSummary {
                    runs: 3,
                    reads: 3,
                    wanted_bytes: 8,
                    read_bytes: 8,
                }),
            },
        )
        .unwrap_or_else(|e| panic!("{e}"));
        assert!(
            !plain.explain().contains("coalesced"),
            "{}",
            plain.explain()
        );
        // an unmeasured entry says so
        let local = plan_read(
            &env_with("ext4", Gds::default(), 8),
            &defaults(),
            &ReadRequest {
                files: files(1, 8),
                destination: Destination::Host,
                allocation_bytes: None,
                coalesced: Some(summary),
            },
        )
        .unwrap_or_else(|e| panic!("{e}"));
        assert!(
            local.explain().contains("gaps up to 64 KiB from profile"),
            "{}",
            local.explain()
        );
        assert!(
            local
                .explain()
                .contains("LocalBlock.io_cost_us unmeasured, floor"),
            "{}",
            local.explain()
        );
    }

    #[test]
    fn refusals() {
        let env = env_with("ext4", Gds::default(), 8);
        assert!(matches!(
            plan_read(
                &env,
                &defaults(),
                &ReadRequest {
                    files: vec![],
                    destination: Destination::Host,
                    allocation_bytes: None,
                    coalesced: None,
                }
            ),
            Err(PlanError::Empty)
        ));
        assert!(matches!(
            plan_read(
                &env,
                &defaults(),
                &ReadRequest {
                    files: files(2, 10),
                    destination: Destination::Device {
                        device: 0,
                        free_bytes: 19
                    },
                    allocation_bytes: None,
                    coalesced: None,
                }
            ),
            Err(PlanError::DoesNotFit {
                available: 19,
                needed: 20
            })
        ));
    }

    proptest! {
        #[test]
        fn plans_are_bounded_and_explained(
            profile in crate::profile::tests::arb_profile(),
            fs in proptest::sample::select(vec!["nfs", "ext4", "xfs", "tmpfs", "fuse.s3fs", "lustre", "weird"]),
            n_files in 1usize..64,
            bytes in 1u64..(1 << 34),
            cpus in 1usize..256,
            module in any::<bool>(),
            lib in any::<bool>(),
            device in any::<bool>(),
        ) {
            let gds = Gds { module_loaded: module, cufile_library: lib.then(|| PathBuf::from("/lib/libcufile.so")) };
            let env = env_with(fs, gds, cpus);
            let destination = if device { Destination::Device { device: 0, free_bytes: u64::MAX } } else { Destination::Host };
            let request = ReadRequest { files: files(n_files, bytes), destination, allocation_bytes: None, coalesced: None };
            let plan = plan_read(&env, &profile, &request).unwrap_or_else(|e| panic!("{e}"));
            let lookup = profile.lookup(&env, &request.files[0].path);
            let plateau = lookup.entry.plateau().map(|p| p.picked.files_in_flight as usize).unwrap_or(1);
            prop_assert!(plan.files_in_flight >= 1 && plan.files_in_flight <= n_files);
            prop_assert!(plan.files_in_flight <= plateau, "{} > plateau {plateau}", plan.files_in_flight);
            prop_assert!(plan.readers_per_file >= 1 && plan.readers_per_file <= MAX_READERS_PER_FILE);
            prop_assert_eq!(plan.split_bytes == 0, !lookup.entry.split_helps);
            prop_assert!(!plan.reasons.is_empty());
            prop_assert_eq!(&plan.profile_source, &profile.source);
            let text = plan.explain();
            prop_assert!(text.contains(&format!("profile {:?}", profile.source)), "{text}");
            prop_assert!(text.contains(&format!("{}.aggregate_gbps", lookup.label)), "{text}");
            if !lookup.from_profile {
                prop_assert!(text.contains("conservative NFS-shaped defaults"), "{text}");
            }
            // GDS is never chosen without both halves in the environment, the
            // profile saying it registers, and a device destination
            if plan.transport == Transport::CuFile {
                prop_assert!(module && lib && device);
                prop_assert!(lookup.entry.gds_registers());
                prop_assert_eq!(plan.staging, Staging::None);
            }
            if device && module && lib && lookup.entry.gds_registers() {
                prop_assert_eq!(plan.transport, Transport::CuFile);
            }
            // a device destination without GDS always stages through pinned memory
            if device && plan.transport != Transport::CuFile {
                let staged = matches!(plan.staging, Staging::Pinned { count, .. } if (2..=64).contains(&count));
                prop_assert!(staged, "device destination without GDS must stage: {:?}", plan.staging);
            }
            if !device {
                prop_assert_eq!(plan.staging, Staging::None);
                prop_assert_eq!(plan.transport, Transport::Pread);
            }
        }
    }
}
