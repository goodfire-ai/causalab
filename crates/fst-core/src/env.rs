//! What the machine is. Probed once, carried as a value, handed to the planner.
//!
//! Nothing here decides anything; it only observes. The planner reads an
//! [`Env`] and never the machine, so a test can hand it any machine it likes.

use std::path::{Path, PathBuf};

/// The kind of file system a path lives on, as far as the transport choice
/// cares. Derived from the mount table's type field.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StorageClass {
    /// A local block device (ext4, xfs, btrfs, apfs, ...). Parallel reads of
    /// one file scale; GPUDirect Storage is possible.
    LocalBlock,
    /// NFS. Measured on an NFSv3 mount (2026-09-08): one file caps near
    /// 3 GB/s however it is split, several files in flight reach the mount's
    /// aggregate.
    Nfs,
    /// A FUSE mount (blob stores, sshfs, ...). No page cache to count on.
    Fuse,
    /// RAM-backed (tmpfs, ramfs, devtmpfs). Reads are memcpy.
    Ram,
    /// A network or cluster file system not listed above (lustre, gpfs,
    /// weka, beegfs, cifs). Treated like NFS until measured.
    OtherNetwork(String),
    /// Anything else, by its mount-table name.
    Other(String),
}

impl StorageClass {
    /// Classify a mount-table file-system type string.
    pub fn from_fs_type(fs_type: &str) -> StorageClass {
        match fs_type {
            "ext4" | "ext3" | "ext2" | "xfs" | "btrfs" | "f2fs" | "zfs" | "apfs" | "hfs" => {
                StorageClass::LocalBlock
            }
            "nfs" | "nfs4" => StorageClass::Nfs,
            "fuse" | "fuseblk" | "fuse.sshfs" | "fuse.s3fs" | "fuse.gcsfuse" | "fuse.blobfuse2" => {
                StorageClass::Fuse
            }
            other if other.starts_with("fuse.") => StorageClass::Fuse,
            "tmpfs" | "ramfs" | "devtmpfs" => StorageClass::Ram,
            "lustre" | "gpfs" | "wekafs" | "beegfs" | "cifs" | "smb3" | "ceph" | "glusterfs" => {
                StorageClass::OtherNetwork(other_owned(fs_type))
            }
            other => StorageClass::Other(other_owned(other)),
        }
    }

    /// Whether a single file's read throughput is expected to scale with the
    /// number of concurrent readers on it.
    pub fn single_file_scales(&self) -> bool {
        matches!(self, StorageClass::LocalBlock | StorageClass::Ram)
    }
}

fn other_owned(s: &str) -> String {
    s.to_owned()
}

/// One mount-table line the classifier needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mount {
    /// Where it is mounted.
    pub point: PathBuf,
    /// Its file-system type string.
    pub fs_type: String,
}

/// Whether GPUDirect Storage could be used, and why not if not.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Gds {
    /// The `nvidia_fs` kernel module is loaded.
    pub module_loaded: bool,
    /// `libcufile` was found here.
    pub cufile_library: Option<PathBuf>,
}

impl Gds {
    /// GDS needs both halves.
    pub fn available(&self) -> bool {
        self.module_loaded && self.cufile_library.is_some()
    }
}

/// The observed machine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Env {
    /// Mount table, longest mount points first so the first prefix match wins.
    pub mounts: Vec<Mount>,
    /// GPUDirect Storage support.
    pub gds: Gds,
    /// Logical CPUs available to this process.
    pub cpus: usize,
}

impl Env {
    /// The class of the file system holding `path` — the longest mount point
    /// that is a prefix of it. A path no mount claims is [`StorageClass::Other`]
    /// with an empty name.
    pub fn storage_class(&self, path: &Path) -> StorageClass {
        self.mounts
            .iter()
            .find(|m| path.starts_with(&m.point))
            .map(|m| StorageClass::from_fs_type(&m.fs_type))
            .unwrap_or_else(|| StorageClass::Other(String::new()))
    }

    /// Build an [`Env`] from parts; sorts the mounts so lookup is by longest
    /// prefix. What tests use instead of [`Env::probe`].
    pub fn new(mut mounts: Vec<Mount>, gds: Gds, cpus: usize) -> Env {
        mounts.sort_by_key(|m| std::cmp::Reverse(m.point.as_os_str().len()));
        Env { mounts, gds, cpus }
    }

    /// Observe this machine: `/proc/self/mounts` and `/proc/modules` on
    /// Linux, the well-known `libcufile` locations, the CPU count. On other
    /// platforms the mount table is empty and GDS is absent.
    pub fn probe() -> Env {
        let mounts = read_proc_mounts().unwrap_or_default();
        let module_loaded = std::fs::read_to_string("/proc/modules")
            .map(|text| text.lines().any(|line| line.starts_with("nvidia_fs ")))
            .unwrap_or(false);
        let cufile_library = CUFILE_CANDIDATES
            .iter()
            .map(PathBuf::from)
            .find(|p| p.is_file());
        let cpus = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1);
        Env::new(
            mounts,
            Gds {
                module_loaded,
                cufile_library,
            },
            cpus,
        )
    }
}

/// Where `libcufile` is usually installed.
pub const CUFILE_CANDIDATES: &[&str] = &[
    "/usr/local/cuda/lib64/libcufile.so",
    "/usr/local/cuda/lib64/libcufile.so.0",
    "/usr/lib/x86_64-linux-gnu/libcufile.so.0",
    "/usr/lib/x86_64-linux-gnu/libcufile.so",
];

fn read_proc_mounts() -> Option<Vec<Mount>> {
    let text = std::fs::read_to_string("/proc/self/mounts").ok()?;
    Some(parse_mounts(&text))
}

/// Parse `/proc/self/mounts` text: `device point type options dump pass`.
/// Octal escapes in mount points (`\040` for a space) are decoded.
pub fn parse_mounts(text: &str) -> Vec<Mount> {
    text.lines()
        .filter_map(|line| {
            let mut fields = line.split_whitespace();
            let _device = fields.next()?;
            let point = unescape_mount(fields.next()?);
            let fs_type = fields.next()?.to_owned();
            Some(Mount {
                point: PathBuf::from(point),
                fs_type,
            })
        })
        .collect()
}

fn unescape_mount(raw: &str) -> String {
    let bytes = raw.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let escaped = (bytes[i] == b'\\' && i + 4 <= bytes.len())
            .then(|| std::str::from_utf8(&bytes[i + 1..i + 4]).ok())
            .flatten()
            .and_then(|oct| u8::from_str_radix(oct, 8).ok());
        match escaped {
            Some(code) => {
                out.push(code);
                i += 4;
            }
            None => {
                out.push(bytes[i]);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    const MOUNTS: &str = "\
rootfs / ext4 rw 0 0
server:/export/a /mnt/nfs nfs rw,vers=3 0 0
tmpfs /dev/shm tmpfs rw 0 0
/dev/nvme0n1 /tmp xfs rw 0 0
blobfuse2 /mnt/nfs/user/blob fuse.blobfuse2 rw 0 0
s3fs /mnt/with\\040space fuse.s3fs rw 0 0
";

    fn env() -> Env {
        Env::new(parse_mounts(MOUNTS), Gds::default(), 16)
    }

    #[test]
    fn longest_prefix_wins() {
        let env = env();
        assert_eq!(
            env.storage_class(Path::new("/mnt/nfs/user/x")),
            StorageClass::Nfs
        );
        assert_eq!(
            env.storage_class(Path::new("/mnt/nfs/user/blob/x")),
            StorageClass::Fuse
        );
        assert_eq!(
            env.storage_class(Path::new("/tmp/x")),
            StorageClass::LocalBlock
        );
        assert_eq!(
            env.storage_class(Path::new("/dev/shm/x")),
            StorageClass::Ram
        );
        assert_eq!(
            env.storage_class(Path::new("/etc/x")),
            StorageClass::LocalBlock
        );
        assert_eq!(
            env.storage_class(Path::new("/mnt/with space/x")),
            StorageClass::Fuse
        );
    }

    #[test]
    fn classes_by_fs_type() {
        assert_eq!(StorageClass::from_fs_type("nfs4"), StorageClass::Nfs);
        assert_eq!(
            StorageClass::from_fs_type("lustre"),
            StorageClass::OtherNetwork("lustre".into())
        );
        assert_eq!(
            StorageClass::from_fs_type("fuse.rclone"),
            StorageClass::Fuse
        );
        assert_eq!(
            StorageClass::from_fs_type("squashfs"),
            StorageClass::Other("squashfs".into())
        );
        assert!(StorageClass::LocalBlock.single_file_scales());
        assert!(!StorageClass::Nfs.single_file_scales());
    }

    #[test]
    fn gds_needs_both_halves() {
        assert!(
            !Gds {
                module_loaded: true,
                cufile_library: None
            }
            .available()
        );
        assert!(
            !Gds {
                module_loaded: false,
                cufile_library: Some("/x".into())
            }
            .available()
        );
        assert!(
            Gds {
                module_loaded: true,
                cufile_library: Some("/x".into())
            }
            .available()
        );
    }

    #[test]
    fn probe_does_not_panic() {
        let env = Env::probe();
        assert!(env.cpus >= 1);
    }
}
