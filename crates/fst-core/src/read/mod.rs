//! The read engine: a [`ReadPlan`] executed over a [`Storage`].
//!
//! [`plan::plan_read`](crate::plan::plan_read) decides *how* bytes should
//! move; this module moves them. The caller describes *what* as a
//! [`ReadJob`]: per file, the byte ranges wanted and where each lands — a
//! host slice the caller owns, or an offset into a device buffer the caller
//! owns. [`execute`] then keeps at most `files_in_flight` files open, splits
//! each file's ranges into pieces of `split_bytes`, runs at most
//! `readers_per_file` pieces per file concurrently, opens one
//! [`RangeReader`](crate::storage::RangeReader) per file, and writes every
//! destination byte exactly once.
//!
//! Transports: [`Transport::Pread`] and [`Transport::Mmap`] both read through
//! the `Storage` the caller passes — the caller picks the backend that
//! matches the plan (`PosixStorage` for `Pread`, `MmapStorage` for `Mmap`,
//! `SimStorage` for tests). [`Transport::CuFile`] reads device pieces
//! through the [`DirectStorage`] the caller passes, straight into device
//! memory with no staging; host pieces still go through the `Storage`, since
//! GDS only ever targets device memory.
//!
//! cuFile fallback. GDS can fail per file at run time — on an H100 host
//! whose `/tmp` is xfs over md RAID, `cuFileHandleRegister` refuses every
//! file there — and a wrong planning decision must cost a slow path, never a
//! failed load. So under a `CuFile` plan: a piece whose cuFile read fails
//! with [`CudaError::Register`] is re-read through pread and staging, and
//! the file stays on pread for the rest of the job (one [`Fallback`] naming
//! the file); [`CudaError::Unavailable`] — `libcufile` itself is gone —
//! switches the whole job to pread (one [`Fallback`] with no path); a plan
//! that says `CuFile` when the caller passed no `DirectStorage` runs
//! entirely on pread (one [`Fallback`], `"no DirectStorage provided"`). Any
//! other cuFile error is [`ReadError::DirectRead`] and fails the job. The
//! staging ring for the fallback is the plan's if it has one, else the
//! planner's own pread geometry, and is allocated only when a piece first
//! stages — a job that reads everything through cuFile pins no memory.
//!
//! Device destinations off the cuFile path go through a ring of pinned
//! staging buffers ([`Staging::Pinned`]) and a [`DeviceCopier`]: a piece is
//! read into a free slot, an asynchronous copy to the device is enqueued,
//! and the slot is marked dirty. A dirty slot becomes free again only after
//! [`DeviceCopier::synchronize`] — see [`staging`] for the exact rule.
//!
//! Placements. A [`Transfer`] may carry [`Placement`]s: the read is a
//! coalesced range (see [`crate::select::coalesce`]) and only the bytes the
//! placements name land, back to back, the gaps between them read and
//! dropped. A host piece with placements is read into a per-worker scratch
//! buffer and copied out placement by placement. A device piece with
//! placements always stages (cuFile lands a range whole, so a placed piece
//! under a `CuFile` plan takes the staging route by design, not as a
//! fallback); when its placements are a regular pattern — equal lengths,
//! constant source and destination strides, the strided-slice case — one
//! [`DeviceCopier::copy_to_device_2d`] lands the whole piece, otherwise one
//! copy per placement does, and the report counts both
//! ([`ReadReport::placed_pieces`], [`ReadReport::scatter_copies`]). A
//! placed transfer split into pieces is cut so that every piece starts at a
//! placement and ends at one: no piece begins or ends inside a gap.
//!
//! Failure: the first error cancels the job. No new piece starts after the
//! failure is recorded; pieces already started run to their own end (their
//! destinations may or may not be filled); every destination whose piece
//! never started is untouched. Enqueued device copies are synchronized before
//! returning so no staging buffer is freed under an in-flight DMA.

use std::path::PathBuf;
use std::time::Duration;

use fst_cuda::{CudaError, DeviceCopier, DevicePtr, DirectStorage};

use crate::error::StorageError;
use crate::plan::{ReadPlan, STAGING_BUFFER_BYTES, Staging, Transport};
use crate::storage::{ReadRange, Storage};

pub use crate::select::Placement;

mod pool;
mod staging;
#[cfg(test)]
mod tests;

/// Where one transfer's bytes land.
#[derive(Debug)]
pub enum Dest<'a> {
    /// A caller-owned host buffer, exactly as long as the transfer's range.
    Host(&'a mut [u8]),
    /// A caller-owned device buffer, filled from `offset`: through cuFile
    /// under a [`Transport::CuFile`] plan, else through pinned staging and
    /// the job's [`DeviceCopier`].
    Device {
        /// The buffer.
        ptr: DevicePtr,
        /// Byte offset into the buffer where the range lands.
        offset: u64,
    },
}

impl Dest<'_> {
    /// Bytes available at the destination for this transfer: a host slice's
    /// length, or a device buffer's bytes from the offset on.
    fn available(&self) -> u64 {
        match self {
            Dest::Host(slice) => slice.len() as u64,
            Dest::Device { ptr, offset } => ptr.nbytes.saturating_sub(*offset),
        }
    }

    /// Whether `len` landed bytes fit here without overrun or slack. A host
    /// slice must be exactly the landing (it is the destination); a device
    /// buffer may be larger (several transfers share one buffer).
    fn holds(&self, len: u64) -> bool {
        match self {
            Dest::Host(_) => self.available() == len,
            Dest::Device { .. } => self.available() >= len,
        }
    }
}

/// One contiguous byte range of one file and where it lands.
#[derive(Debug)]
pub struct Transfer<'a> {
    /// The range within the file.
    pub range: ReadRange,
    /// Its destination; must hold exactly `range.len` bytes, or exactly the
    /// placements' total when there are placements.
    pub dest: Dest<'a>,
    /// How the range's bytes are placed. Empty: the whole range lands
    /// contiguously at the destination. Otherwise the range is a coalesced
    /// read and `placements[i]` copies `len` bytes from `src` within the
    /// range to `dst` within the destination; placements are sorted by
    /// `src`, do not overlap, and land back to back from `dst == 0`, so the
    /// destination is exactly their total (host) or holds it from its offset
    /// (device). Bytes no placement names are read and dropped. What
    /// [`crate::select::coalesce`] produces, with a [`crate::select::Read`]'s
    /// placements taken as they are.
    pub placements: Vec<Placement>,
}

impl Transfer<'_> {
    /// Bytes this transfer lands: the range, or the placements' total.
    pub fn landed_bytes(&self) -> u64 {
        if self.placements.is_empty() {
            self.range.len
        } else {
            self.placements.iter().map(|p| p.len).sum()
        }
    }
}

/// Why a transfer's placements do not describe a gather of its range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlacementFault {
    /// The placement's source bytes run past the transfer's range.
    OutOfRange {
        /// The placement's `src`.
        src: u64,
        /// Its `len`.
        len: u64,
        /// The range's length.
        range_len: u64,
    },
    /// The placement starts before the previous one ended, or is empty.
    Unordered {
        /// The placement's `src`.
        src: u64,
        /// Where the previous placement's source ended.
        previous_end: u64,
    },
    /// The placement does not land where the previous one ended.
    NotPacked {
        /// The placement's `dst`.
        dst: u64,
        /// Where it had to land.
        expected: u64,
    },
}

impl std::fmt::Display for PlacementFault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlacementFault::OutOfRange {
                src,
                len,
                range_len,
            } => write!(
                f,
                "source {src}..{} runs past the {range_len}-byte range",
                src + len
            ),
            PlacementFault::Unordered { src, previous_end } => write!(
                f,
                "source {src} is not after the previous placement's end {previous_end}, or the placement is empty"
            ),
            PlacementFault::NotPacked { dst, expected } => {
                write!(
                    f,
                    "lands at {dst} but the previous placement ended at {expected}"
                )
            }
        }
    }
}

/// One file and its transfers. Transfers may be issued in any order; each
/// lands at its own destination.
#[derive(Debug)]
pub struct FileJob<'a> {
    /// The file.
    pub path: PathBuf,
    /// The ranges to read from it.
    pub transfers: Vec<Transfer<'a>>,
}

/// Everything one [`execute`] call reads.
#[derive(Debug, Default)]
pub struct ReadJob<'a> {
    /// Files, in the order they are started (subject to `files_in_flight`).
    pub files: Vec<FileJob<'a>>,
}

impl ReadJob<'_> {
    /// Whether any transfer lands on a device.
    fn has_device_dest(&self) -> bool {
        self.files
            .iter()
            .flat_map(|f| f.transfers.iter())
            .any(|t| matches!(t.dest, Dest::Device { .. }))
    }
}

/// A run-time departure from the plan's cuFile transport, towards pread.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fallback {
    /// The file that fell back, or `None` when the whole job did: no
    /// `DirectStorage` was given, or `libcufile` answered `Unavailable`.
    pub path: Option<PathBuf>,
    /// What cuFile said, or why it was never asked.
    pub reason: String,
}

/// What a successful [`execute`] did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadReport {
    /// Bytes read from storage (equals bytes landed): the sum of the three
    /// routes below.
    pub bytes_read: u64,
    /// Bytes landed in device memory straight from storage through cuFile.
    pub cufile_bytes: u64,
    /// Bytes landed in device memory through pread into pinned staging and
    /// an asynchronous copy.
    pub staged_bytes: u64,
    /// Bytes landed in host buffers through the `Storage` reader.
    pub host_bytes: u64,
    /// Pieces issued, over all routes.
    pub pieces: usize,
    /// Bytes read and dropped: the gaps inside coalesced pieces. Bytes
    /// landed are `bytes_read - gap_bytes`.
    pub gap_bytes: u64,
    /// Pieces that landed through placements, host memcpy or device copies.
    pub placed_pieces: usize,
    /// Device copies issued one per placement, for placed pieces whose
    /// pattern was not regular enough for a single 2-D copy. A regular piece
    /// issues one copy and adds nothing here.
    pub scatter_copies: usize,
    /// Files opened through the `Storage` (one reader each). A file read
    /// entirely through cuFile opens no reader and is not counted.
    pub files_opened: usize,
    /// Where the plan said cuFile and the engine read through pread, and why.
    /// Empty for pread and mmap plans and for a cuFile plan that held.
    pub fallbacks: Vec<Fallback>,
    /// Wall time from validation to the final synchronize.
    pub elapsed: Duration,
}

/// Why a read job did not complete.
#[derive(Debug, thiserror::Error)]
pub enum ReadError {
    /// The job has a device destination but no [`DeviceCopier`] was given.
    /// Required under every transport: a `CuFile` plan needs it for the
    /// fallback, which must never itself be what fails.
    #[error("job has device destinations but no device copier was supplied")]
    CopierRequired,
    /// The job has a device destination but the plan stages nothing and is
    /// not a `CuFile` plan (which gets a fallback ring of its own).
    #[error("job has device destinations but the plan has no pinned staging")]
    StagingRequired,
    /// The plan's staging ring has no capacity.
    #[error("staging ring of {count} x {bytes} bytes cannot hold a piece")]
    EmptyStaging {
        /// Buffers in the ring.
        count: usize,
        /// Bytes per buffer.
        bytes: u64,
    },
    /// A destination does not hold exactly the transfer's range.
    #[error("{path}: range {}..{} needs {} bytes but the destination has {available}",
            range.offset, range.offset + range.len, range.len)]
    DestinationMismatch {
        /// The file.
        path: PathBuf,
        /// The transfer's range.
        range: ReadRange,
        /// Bytes the destination has from its offset.
        available: u64,
    },
    /// A destination does not hold exactly what a placed transfer lands.
    #[error("{path}: range {}..{} places {wanted} bytes but the destination has {available}",
            range.offset, range.offset + range.len)]
    PlacedDestinationMismatch {
        /// The file.
        path: PathBuf,
        /// The transfer's range.
        range: ReadRange,
        /// Bytes the placements land.
        wanted: u64,
        /// Bytes the destination has from its offset.
        available: u64,
    },
    /// A transfer's placements are not a gather of its range: one runs past
    /// the range, overlaps its predecessor, or does not land where the
    /// previous one ended.
    #[error("{path}: range {}..{}, placement {index}: {fault}",
            range.offset, range.offset + range.len)]
    InvalidPlacement {
        /// The file.
        path: PathBuf,
        /// The transfer's range.
        range: ReadRange,
        /// Which placement, in the transfer's order.
        index: usize,
        /// What is wrong with it.
        fault: PlacementFault,
    },
    /// A file could not be opened.
    #[error("{path}: open failed: {source}")]
    Open {
        /// The file.
        path: PathBuf,
        /// The backend's error.
        #[source]
        source: StorageError,
    },
    /// A piece could not be read.
    #[error("{path}: read of {}..{} failed: {source}", range.offset, range.offset + range.len)]
    Read {
        /// The file.
        path: PathBuf,
        /// The piece that failed.
        range: ReadRange,
        /// The backend's error.
        #[source]
        source: StorageError,
    },
    /// A cuFile read failed with something other than a registration or
    /// availability problem (those fall back to pread; see the module docs).
    #[error("{path}: cuFile read of {}..{} failed: {source}", range.offset, range.offset + range.len)]
    DirectRead {
        /// The file.
        path: PathBuf,
        /// The piece that failed.
        range: ReadRange,
        /// What cuFile said.
        #[source]
        source: CudaError,
    },
    /// A piece was larger than a staging buffer. `execute` sizes pieces to
    /// the ring, so this names a bug in the engine, not in the caller.
    #[error("{path}: piece {}..{} exceeds the {capacity}-byte staging buffer",
            range.offset, range.offset + range.len)]
    PieceExceedsStaging {
        /// The file.
        path: PathBuf,
        /// The piece.
        range: ReadRange,
        /// Bytes one staging buffer holds.
        capacity: u64,
    },
    /// Pinned staging could not be allocated or synchronized.
    #[error("staging: {source}")]
    Staging {
        /// The runtime's error.
        #[source]
        source: CudaError,
    },
    /// A piece could not be copied to its device destination.
    #[error("{path}: device copy of {}..{} failed: {source}", range.offset, range.offset + range.len)]
    Copy {
        /// The file.
        path: PathBuf,
        /// The piece that failed.
        range: ReadRange,
        /// The runtime's error.
        #[source]
        source: CudaError,
    },
}

/// Execute `job` as `plan` says, reading through `storage`.
///
/// `copier` is required when any transfer lands on a device — under every
/// transport, since a `CuFile` job may fall back to staging — and ignored
/// otherwise. `direct` is used only under a [`Transport::CuFile`] plan; a
/// `CuFile` plan without it runs on pread and says so in the report. The
/// job is consumed: its host slices are split into pieces and handed to
/// worker threads for the duration of the call. Returns after every
/// destination byte has landed (device copies synchronized), or with the
/// first error.
pub fn execute(
    plan: &ReadPlan,
    storage: &dyn Storage,
    copier: Option<&dyn DeviceCopier>,
    direct: Option<&dyn DirectStorage>,
    job: ReadJob<'_>,
) -> Result<ReadReport, ReadError> {
    let started = std::time::Instant::now();
    let setup = validate(plan, copier, &job)?;
    let mut fallbacks = Vec::new();
    let direct = match (plan.transport, direct) {
        (Transport::CuFile, Some(direct)) => Some(direct),
        (Transport::CuFile, None) if job.has_device_dest() => {
            fallbacks.push(Fallback {
                path: None,
                reason: "no DirectStorage provided".into(),
            });
            None
        }
        _ => None,
    };
    let ring = staging::LazyRing::new(copier, setup.staging);
    // without cuFile every device piece stages: allocate before any I/O so
    // a pinned-memory failure is reported before a byte moves. With cuFile
    // the ring waits for the first piece that actually falls back.
    if direct.is_none() && setup.staging.is_some() {
        ring.get()?;
    }
    let device_block = setup.staging.map(|(_, bytes)| bytes);
    let pieces_per_file: Vec<(PathBuf, Vec<pool::Piece<'_>>)> = job
        .files
        .into_iter()
        .map(|file| {
            let pieces = pool::split_file(file.transfers, plan.split_bytes, device_block);
            (file.path, pieces)
        })
        .collect();
    let outcome = pool::run(plan, storage, copier, direct, &ring, pieces_per_file);
    // every enqueued copy must land before the staging buffers are dropped,
    // whether or not the job succeeded
    let synced = match (ring.allocated(), copier) {
        (Some(_), Some(copier)) => copier
            .synchronize()
            .map_err(|source| ReadError::Staging { source }),
        _ => Ok(()),
    };
    let counts = outcome?;
    synced?;
    fallbacks.extend(counts.fallbacks);
    Ok(ReadReport {
        bytes_read: counts.cufile_bytes + counts.staged_bytes + counts.host_bytes,
        cufile_bytes: counts.cufile_bytes,
        staged_bytes: counts.staged_bytes,
        host_bytes: counts.host_bytes,
        pieces: counts.pieces,
        gap_bytes: counts.gap_bytes,
        placed_pieces: counts.placed_pieces,
        scatter_copies: counts.scatter_copies,
        files_opened: counts.files_opened,
        fallbacks,
        elapsed: started.elapsed(),
    })
}

/// The first placement that is not part of a gather of `range`: sources in
/// order and disjoint within the range, destinations back to back from zero.
fn placement_fault(range: &ReadRange, placements: &[Placement]) -> Option<(usize, PlacementFault)> {
    let mut src_end = 0u64;
    let mut dst_end = 0u64;
    for (index, p) in placements.iter().enumerate() {
        let fault = if p.len == 0 || p.src < src_end {
            Some(PlacementFault::Unordered {
                src: p.src,
                previous_end: src_end,
            })
        } else if p.src.checked_add(p.len).is_none_or(|end| end > range.len) {
            Some(PlacementFault::OutOfRange {
                src: p.src,
                len: p.len,
                range_len: range.len,
            })
        } else if p.dst != dst_end {
            Some(PlacementFault::NotPacked {
                dst: p.dst,
                expected: dst_end,
            })
        } else {
            None
        };
        if let Some(fault) = fault {
            return Some((index, fault));
        }
        src_end = p.src + p.len;
        dst_end += p.len;
    }
    None
}

/// What validation settles before any I/O starts.
struct Setup {
    /// `(count, bytes)` of the staging ring when the job may need one.
    staging: Option<(usize, u64)>,
}

/// The ring a `CuFile` plan — which stages nothing — falls back to: the
/// planner's own pread rule, two buffers per concurrent reader within
/// `2..=64`, of [`STAGING_BUFFER_BYTES`] each. Mirrors `plan_read`; it lives
/// here because the planner is pure and does not know cuFile can fail per
/// file.
fn fallback_staging(plan: &ReadPlan) -> (usize, u64) {
    ((plan.concurrency() * 2).clamp(2, 64), STAGING_BUFFER_BYTES)
}

fn validate(
    plan: &ReadPlan,
    copier: Option<&dyn DeviceCopier>,
    job: &ReadJob<'_>,
) -> Result<Setup, ReadError> {
    for file in &job.files {
        for transfer in &file.transfers {
            if transfer.placements.is_empty() {
                if !transfer.dest.holds(transfer.range.len) {
                    return Err(ReadError::DestinationMismatch {
                        path: file.path.clone(),
                        range: transfer.range.clone(),
                        available: transfer.dest.available(),
                    });
                }
                continue;
            }
            if let Some((index, fault)) = placement_fault(&transfer.range, &transfer.placements) {
                return Err(ReadError::InvalidPlacement {
                    path: file.path.clone(),
                    range: transfer.range.clone(),
                    index,
                    fault,
                });
            }
            let wanted = transfer.landed_bytes();
            if !transfer.dest.holds(wanted) {
                return Err(ReadError::PlacedDestinationMismatch {
                    path: file.path.clone(),
                    range: transfer.range.clone(),
                    wanted,
                    available: transfer.dest.available(),
                });
            }
        }
    }
    let staging = if job.has_device_dest() {
        if copier.is_none() {
            return Err(ReadError::CopierRequired);
        }
        match (plan.staging, plan.transport) {
            (Staging::Pinned { count, bytes }, _) if count == 0 || bytes == 0 => {
                return Err(ReadError::EmptyStaging { count, bytes });
            }
            (Staging::Pinned { count, bytes }, _) => Some((count, bytes)),
            (Staging::None, Transport::CuFile) => Some(fallback_staging(plan)),
            (Staging::None, Transport::Pread | Transport::Mmap) => {
                return Err(ReadError::StagingRequired);
            }
        }
    } else {
        None
    };
    Ok(Setup { staging })
}
