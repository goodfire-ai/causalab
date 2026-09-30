//! Writing a safetensors object from buffers the caller owns.
//!
//! The reference `save_file` joins header and tensors into one buffer and
//! writes that: every byte is copied once more, and in Python the copy holds
//! the GIL. The alternative is to build the header, then hand the header and
//! each tensor's memory to the file in turn; this module does that against
//! [`crate::storage::PartWriter`], with two additions:
//!
//! * a part may live on a device. It is drained through a ring of pinned
//!   host buffers ([`StagingRing`]) with the copy of the next chunk running
//!   while the current one is written; see [`staging`].
//! * the write can be [`WriteOptions::durable`] (`fsync` before returning)
//!   and [`WriteOptions::atomic`] (written under a sibling temp name and
//!   renamed into place, so the final name never holds a half-written
//!   object); see [`atomic`].
//!
//! [`serialize_to_vec`] is the same header and ordering code for callers who
//! want the object as one buffer (the `save() -> bytes` path).
//!
//! Every failure is a [`WriteError`] naming the path and, where one is
//! involved, the caller's part. When `atomic`, a failure removes the temp
//! object (best effort; a cleanup failure is logged, never masks the cause).

use std::fmt;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use fst_cuda::{CudaError, DeviceCopier, DevicePtr};

use crate::error::{FormatError, StorageError};
use crate::format::{self, BuiltHeader, TensorSpec};
use crate::storage::{PartWriter, Storage};

mod atomic;
mod staging;
#[cfg(test)]
mod tests;

pub use staging::StagingRing;

/// One tensor's bytes and where they live. Borrowed: the caller keeps the
/// memory alive and unmodified until the write returns.
#[derive(Debug, Clone, Copy)]
pub enum Part<'a> {
    /// Host memory.
    Host(&'a [u8]),
    /// The first `nbytes` of a device buffer, staged through pinned memory.
    Device {
        /// The buffer, owned by the caller (torch).
        ptr: DevicePtr,
        /// Bytes of it that are the tensor.
        nbytes: u64,
    },
}

impl Part<'_> {
    /// Bytes this part contributes to the data section.
    pub fn nbytes(&self) -> u64 {
        match self {
            Part::Host(bytes) => bytes.len() as u64,
            Part::Device { nbytes, .. } => *nbytes,
        }
    }

    /// Whether this part lives on a device.
    pub fn is_device(&self) -> bool {
        matches!(self, Part::Device { .. })
    }
}

/// A safetensors object ready to write: the built header and the caller's
/// parts in data-section order.
#[derive(Debug, Clone)]
pub struct Payload<'a> {
    header: BuiltHeader,
    /// Parts in data-section order; `header.order[k]` is the caller's index
    /// of `parts[k]` and `names[k]` its tensor name.
    parts: Vec<Part<'a>>,
    names: Vec<String>,
}

impl<'a> Payload<'a> {
    /// Build the header for `specs` and `metadata` and pair each spec with
    /// its part (`parts[i]` belongs to `specs[i]`), reordering into the
    /// layout the header describes. Refuses a part whose length disagrees
    /// with its spec, and a device part longer than its buffer.
    pub fn new(
        specs: &[TensorSpec],
        parts: &[Part<'a>],
        metadata: Option<&[(String, String)]>,
    ) -> Result<Self, WriteError> {
        if parts.len() != specs.len() {
            return Err(WriteError::PartCount {
                expected: specs.len(),
                count: parts.len(),
            });
        }
        let header = format::build_header(specs, metadata)?;
        for (index, (spec, part)) in specs.iter().zip(parts).enumerate() {
            if part.nbytes() != spec.nbytes {
                return Err(WriteError::PartLength {
                    index,
                    name: spec.name.clone(),
                    expected: spec.nbytes,
                    actual: part.nbytes(),
                });
            }
            if let Part::Device { ptr, nbytes } = part
                && *nbytes > ptr.nbytes
            {
                return Err(WriteError::DeviceOverrun {
                    index,
                    name: spec.name.clone(),
                    nbytes: *nbytes,
                    capacity: ptr.nbytes,
                });
            }
        }
        let ordered = header.order.iter().map(|&i| parts[i]).collect();
        let names = header
            .order
            .iter()
            .map(|&i| specs[i].name.clone())
            .collect();
        Ok(Payload {
            header,
            parts: ordered,
            names,
        })
    }

    /// The header: bytes, the caller-index order, the layout.
    pub fn header(&self) -> &BuiltHeader {
        &self.header
    }

    /// The parts in data-section order.
    pub fn parts(&self) -> &[Part<'a>] {
        &self.parts
    }

    /// Tensor names in data-section order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.names.iter().map(String::as_str)
    }

    /// The whole object's length: header plus every part.
    pub fn nbytes(&self) -> u64 {
        self.header.bytes.len() as u64 + self.parts.iter().map(Part::nbytes).sum::<u64>()
    }

    /// Whether any part lives on a device (and so a copier is needed).
    pub fn has_device_parts(&self) -> bool {
        self.parts.iter().any(Part::is_device)
    }

    fn largest_device_part(&self) -> u64 {
        self.parts
            .iter()
            .filter(|p| p.is_device())
            .map(Part::nbytes)
            .max()
            .unwrap_or(0)
    }
}

/// How the object reaches storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WriteOptions {
    /// Force the bytes and the name to stable storage before returning.
    pub durable: bool,
    /// Write under a sibling temp name and rename into place when complete.
    pub atomic: bool,
    /// The pinned ring for device parts; unused when every part is on the
    /// host.
    pub staging: StagingRing,
}

/// What a write did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WriteReport {
    /// The object's final path.
    pub path: PathBuf,
    /// Bytes written, header included.
    pub bytes: u64,
    /// Tensor parts in the payload.
    pub parts: usize,
    /// Chunks that went through the staging ring.
    pub staging_chunks: usize,
    /// Wall time from entry to return.
    pub elapsed: Duration,
}

/// Where in a write a storage operation failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stage {
    /// Creating the object (or its temp sibling).
    Create,
    /// Writing the header.
    Header,
    /// Writing one part, by the caller's index and tensor name.
    Part {
        /// Index into the caller's specs.
        index: usize,
        /// The tensor's name.
        name: String,
    },
    /// Finishing the writer (`fsync` when durable, then close).
    Finish,
    /// Renaming the temp object into place.
    Rename,
}

impl fmt::Display for Stage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Stage::Create => f.write_str("creating"),
            Stage::Header => f.write_str("writing the header"),
            Stage::Part { index, name } => write!(f, "writing part {index} ({name:?})"),
            Stage::Finish => f.write_str("finishing"),
            Stage::Rename => f.write_str("renaming into place"),
        }
    }
}

/// What can go wrong writing an object.
#[derive(Debug, thiserror::Error)]
pub enum WriteError {
    /// The specs do not make a valid header.
    #[error(transparent)]
    Format(#[from] FormatError),
    /// Not one part per spec.
    #[error("{count} parts given for {expected} tensors")]
    PartCount {
        /// Specs given.
        expected: usize,
        /// Parts given.
        count: usize,
    },
    /// A part's length disagrees with its spec.
    #[error("part {index} ({name:?}) holds {actual} bytes but its spec declares {expected}")]
    PartLength {
        /// Index into the caller's specs.
        index: usize,
        /// The tensor's name.
        name: String,
        /// Bytes the spec declares.
        expected: u64,
        /// Bytes the part holds.
        actual: u64,
    },
    /// A device part is longer than the buffer it points into.
    #[error("part {index} ({name:?}) asks {nbytes} bytes of a {capacity}-byte device buffer")]
    DeviceOverrun {
        /// Index into the caller's specs.
        index: usize,
        /// The tensor's name.
        name: String,
        /// Bytes the part claims.
        nbytes: u64,
        /// Bytes the buffer holds.
        capacity: u64,
    },
    /// Device parts were given without a copier to stage them.
    #[error("{path}: device-resident parts need a device copier")]
    NoCopier {
        /// The object.
        path: PathBuf,
    },
    /// The staging ring has no buffers or zero-byte buffers.
    #[error("{path}: a staging ring of {count} buffers of {bytes} bytes cannot stage anything")]
    EmptyStaging {
        /// The object.
        path: PathBuf,
        /// Buffers asked for.
        count: usize,
        /// Bytes per buffer asked for.
        bytes: u64,
    },
    /// An atomic write needs a file name to derive its temp name from.
    #[error("{path}: no file name to derive a temporary name from")]
    NoFileName {
        /// The path given.
        path: PathBuf,
    },
    /// Storage refused. `path` is the object the caller asked for; the
    /// source names the file actually touched (the temp sibling when
    /// atomic).
    #[error("{path}: {stage}: {source}")]
    Storage {
        /// The object the caller asked for.
        path: PathBuf,
        /// Which operation.
        stage: Stage,
        /// Storage's answer.
        #[source]
        source: StorageError,
    },
    /// Staging a device part failed.
    #[error("{path}: staging part {index} ({name:?}) from device: {source}")]
    Cuda {
        /// The object the caller asked for.
        path: PathBuf,
        /// Index into the caller's specs.
        index: usize,
        /// The tensor's name.
        name: String,
        /// The copier's answer.
        #[source]
        source: CudaError,
    },
}

/// Write `payload` to `path` on `storage`: the header, then every part in
/// data-section order, each handed to the writer as-is. Device parts need
/// `copier`; host-only payloads ignore it.
pub fn write_object(
    storage: &dyn Storage,
    path: &Path,
    payload: &Payload<'_>,
    options: &WriteOptions,
    copier: Option<&dyn DeviceCopier>,
) -> Result<WriteReport, WriteError> {
    let started = Instant::now();
    let copier = match (payload.has_device_parts(), copier) {
        (false, _) => None,
        (true, None) => {
            return Err(WriteError::NoCopier {
                path: path.to_owned(),
            });
        }
        (true, Some(copier)) => {
            options.staging.validate(path)?;
            Some(copier)
        }
    };
    let temp = options
        .atomic
        .then(|| atomic::temp_name(path))
        .transpose()?;
    let target = temp.as_deref().unwrap_or(path);

    let writer = storage
        .create_writer(target)
        .map_err(|source| WriteError::Storage {
            path: path.to_owned(),
            stage: Stage::Create,
            source,
        })?;
    let outcome = stream(writer, path, payload, options, copier).and_then(|chunks| {
        if let Some(temp) = &temp {
            atomic::commit(storage, temp, path, options.durable)?;
        }
        Ok(chunks)
    });
    let staging_chunks = match outcome {
        Ok(chunks) => chunks,
        Err(err) => {
            if let Some(temp) = &temp {
                atomic::discard(storage, temp);
            }
            return Err(err);
        }
    };
    let report = WriteReport {
        path: path.to_owned(),
        bytes: payload.nbytes(),
        parts: payload.parts.len(),
        staging_chunks,
        elapsed: started.elapsed(),
    };
    tracing::debug!(
        path = %path.display(),
        bytes = report.bytes,
        parts = report.parts,
        staging_chunks,
        elapsed_ms = report.elapsed.as_millis() as u64,
        backend = storage.name(),
        "wrote safetensors object"
    );
    Ok(report)
}

/// Header, then each non-empty part, then `finish`. Returns the number of
/// staging chunks.
fn stream(
    mut writer: Box<dyn PartWriter>,
    path: &Path,
    payload: &Payload<'_>,
    options: &WriteOptions,
    copier: Option<&dyn DeviceCopier>,
) -> Result<usize, WriteError> {
    let storage_error = |stage: Stage, source| WriteError::Storage {
        path: path.to_owned(),
        stage,
        source,
    };
    writer
        .write_parts(&[&payload.header.bytes])
        .map_err(|source| storage_error(Stage::Header, source))?;

    let mut ring =
        copier.map(|c| staging::Ring::new(c, options.staging, payload.largest_device_part()));
    let mut chunks = 0;
    for (k, part) in payload.parts.iter().enumerate() {
        let index = payload.header.order[k];
        let name = payload.names[k].as_str();
        match part {
            Part::Host([]) => {}
            Part::Host(bytes) => writer.write_parts(&[bytes]).map_err(|source| {
                storage_error(
                    Stage::Part {
                        index,
                        name: name.to_owned(),
                    },
                    source,
                )
            })?,
            Part::Device { nbytes: 0, .. } => {}
            Part::Device { ptr, nbytes } => {
                // `write_object` refused device parts without a copier
                let Some(ring) = ring.as_mut() else {
                    return Err(WriteError::NoCopier {
                        path: path.to_owned(),
                    });
                };
                let part = staging::PartRef { path, index, name };
                chunks += ring.drain(*ptr, *nbytes, writer.as_mut(), part)?;
            }
        }
    }
    writer
        .finish(options.durable)
        .map_err(|source| storage_error(Stage::Finish, source))?;
    Ok(chunks)
}

/// The whole object as one buffer: the header and every part in data-section
/// order. One copy of the data; for callers who want bytes, not a file.
pub fn serialize_to_vec(
    specs: &[TensorSpec],
    parts: &[&[u8]],
    metadata: Option<&[(String, String)]>,
) -> Result<Vec<u8>, WriteError> {
    let host: Vec<Part<'_>> = parts.iter().map(|p| Part::Host(p)).collect();
    let payload = Payload::new(specs, &host, metadata)?;
    let mut out = Vec::with_capacity(payload.nbytes() as usize);
    out.extend_from_slice(&payload.header.bytes);
    for part in &payload.parts {
        if let Part::Host(bytes) = part {
            out.extend_from_slice(bytes);
        }
    }
    Ok(out)
}
