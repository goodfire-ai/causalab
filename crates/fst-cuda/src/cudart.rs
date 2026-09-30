//! [`CudaRuntime`]: [`DeviceCopier`] over `libcudart`, loaded at run time.
//!
//! One non-blocking stream per runtime; every copy is `cudaMemcpyAsync`
//! (or `cudaMemcpy2DAsync` for a strided one) on it and
//! [`DeviceCopier::synchronize`] waits for the stream. Pinned buffers
//! come from `cudaHostAlloc`; a dropped buffer goes back to the runtime's
//! pool (up to [`PINNED_POOL_CAP`] bytes) and the next request for the same
//! size takes it without another `cudaHostAlloc`. Pinning is a heavy driver
//! call that serializes across the processes of a node: eight ranks each
//! staging 128-384 MiB per read job spent 0.1-0.2 s of every job in
//! `cudaHostAlloc`/`cudaFreeHost` (measured on one eight-GPU node). Every
//! range is checked against both buffers before any call crosses into the
//! library, so an overrun is a [`CudaError::Overrun`] and never a CUDA fault.
//!
//! `cudaSetDevice` is per host thread, so it is re-applied before each call:
//! the copier is shared across a thread pool whose threads never chose a
//! device. The call is a per-thread table lookup once the context exists.

use std::sync::{Arc, Mutex, PoisonError};

use crate::ffi;
use crate::{Copy2D, CudaError, DeviceCopier, DevicePtr, Fence, PinnedBuffer};

/// Refuse a copy of `nbytes` at `offset` into a `capacity`-byte buffer that
/// would run past its end. Pure; runs before any FFI call and never
/// overflows.
pub fn check_range(nbytes: u64, offset: u64, capacity: u64) -> Result<(), CudaError> {
    match offset.checked_add(nbytes) {
        Some(end) if end <= capacity => Ok(()),
        _ => Err(CudaError::Overrun {
            nbytes,
            offset,
            capacity,
        }),
    }
}

/// A `u64` byte count as the `usize` the runtime takes.
fn byte_count(nbytes: u64, call: &'static str) -> Result<usize, CudaError> {
    usize::try_from(nbytes).map_err(|_| CudaError::Runtime {
        call,
        code: ffi::CUDA_ERROR_MEMORY_ALLOCATION,
    })
}

/// `address + offset` as a device address; `offset` has already been range
/// checked against the buffer, so a failure here is a corrupt pointer.
fn device_address(ptr: DevicePtr, offset: u64) -> Result<usize, CudaError> {
    usize::try_from(offset)
        .ok()
        .and_then(|offset| ptr.address.checked_add(offset))
        .ok_or(CudaError::Overrun {
            nbytes: 0,
            offset,
            capacity: ptr.nbytes,
        })
}

/// Pinned host bytes a runtime keeps for reuse across jobs. Two jobs' worth
/// of the largest staging ring the planner makes (128 buffers of 16 MiB).
pub const PINNED_POOL_CAP: u64 = 4 << 30;

/// Pinned buffers returned by dropped [`PinnedBuffer`]s, reused by size.
#[derive(Default)]
struct PinnedPool {
    /// `(buffers, bytes held)`; the lock is uncontended outside job start/end.
    state: Mutex<(Vec<ffi::HostAllocation>, u64)>,
}

impl PinnedPool {
    fn lock(&self) -> std::sync::MutexGuard<'_, (Vec<ffi::HostAllocation>, u64)> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// A retained buffer of exactly `nbytes`, if one is waiting.
    fn take(&self, nbytes: usize) -> Option<ffi::HostAllocation> {
        let mut state = self.lock();
        let at = state.0.iter().position(|a| a.nbytes() == nbytes)?;
        let alloc = state.0.swap_remove(at);
        state.1 -= alloc.nbytes() as u64;
        Some(alloc)
    }

    /// Keep `alloc` for a later `take`, or let it free itself when the pool
    /// is at its cap.
    fn give(&self, alloc: ffi::HostAllocation) {
        let nbytes = alloc.nbytes() as u64;
        let mut state = self.lock();
        if nbytes > 0 && state.1 + nbytes <= PINNED_POOL_CAP {
            state.1 += nbytes;
            state.0.push(alloc);
        }
    }

    /// `(buffers, bytes)` held.
    fn held(&self) -> (usize, u64) {
        let state = self.lock();
        (state.0.len(), state.1)
    }
}

impl std::fmt::Debug for PinnedPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (buffers, bytes) = self.held();
        write!(f, "PinnedPool({buffers} buffers, {bytes} bytes)")
    }
}

/// The CUDA runtime for one device.
#[derive(Debug)]
pub struct CudaRuntime {
    lib: Arc<ffi::Cudart>,
    stream: ffi::Stream,
    device: u32,
    pool: Arc<PinnedPool>,
}

impl std::fmt::Debug for ffi::Stream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Stream")
    }
}

impl CudaRuntime {
    /// Load `libcudart` from the usual places, select `device` and create the
    /// copy stream. Without the library: [`CudaError::Unavailable`] naming
    /// it; without a usable device: the runtime's error from `cudaSetDevice`.
    pub fn load(device: u32) -> Result<CudaRuntime, CudaError> {
        CudaRuntime::load_from(ffi::CUDART_CANDIDATES, device)
    }

    /// [`CudaRuntime::load`] with an explicit list of library names or paths
    /// to try in order.
    pub fn load_from(candidates: &[&str], device: u32) -> Result<CudaRuntime, CudaError> {
        let lib = Arc::new(ffi::Cudart::load(candidates)?);
        lib.set_device(device)?;
        let stream = ffi::Stream::create(&lib)?;
        Ok(CudaRuntime {
            lib,
            stream,
            device,
            pool: Arc::new(PinnedPool::default()),
        })
    }

    /// Pinned buffers and bytes the runtime is holding for reuse.
    pub fn pooled_pinned(&self) -> (usize, u64) {
        self.pool.held()
    }

    /// The device ordinal this runtime was opened for.
    pub fn device(&self) -> u32 {
        self.device
    }

    /// Which library name or path loaded.
    pub fn library(&self) -> &str {
        &self.lib.loaded_from
    }

    /// `cudaRuntimeGetVersion` as `(major, minor)`.
    pub fn runtime_version(&self) -> Result<(u32, u32), CudaError> {
        let v = self.lib.runtime_version()?;
        let v = u32::try_from(v).unwrap_or(0);
        Ok((v / 1000, (v % 1000) / 10))
    }
}

/// A `cudaHostAlloc` buffer behind the [`PinnedBuffer`] trait; drop hands
/// the allocation to the pool and leaves the empty one behind.
struct CudaPinned {
    alloc: ffi::HostAllocation,
    pool: Arc<PinnedPool>,
}

impl PinnedBuffer for CudaPinned {
    fn as_mut_slice(&mut self) -> &mut [u8] {
        self.alloc.as_mut_slice()
    }
    fn as_slice(&self) -> &[u8] {
        self.alloc.as_slice()
    }
    fn capacity(&self) -> u64 {
        self.alloc.nbytes() as u64
    }
}

impl Drop for CudaPinned {
    fn drop(&mut self) {
        let alloc = std::mem::replace(&mut self.alloc, ffi::HostAllocation::empty());
        self.pool.give(alloc);
    }
}

/// A recorded CUDA event behind the [`Fence`] trait.
struct EventFence(ffi::Event);

impl Fence for EventFence {
    fn wait(&self) -> Result<(), CudaError> {
        self.0.wait()
    }
}

impl DeviceCopier for CudaRuntime {
    fn alloc_pinned(&self, nbytes: u64) -> Result<Box<dyn PinnedBuffer>, CudaError> {
        let nbytes = byte_count(nbytes, "cudaHostAlloc")?;
        let alloc = match self.pool.take(nbytes) {
            Some(alloc) => alloc,
            None => {
                self.lib.set_device(self.device)?;
                ffi::Cudart::host_alloc(&self.lib, nbytes)?
            }
        };
        Ok(Box::new(CudaPinned {
            alloc,
            pool: Arc::clone(&self.pool),
        }))
    }

    fn copy_to_device(
        &self,
        src: &dyn PinnedBuffer,
        nbytes: u64,
        dst: DevicePtr,
        offset: u64,
    ) -> Result<(), CudaError> {
        check_range(nbytes, 0, src.capacity())?;
        check_range(nbytes, offset, dst.nbytes)?;
        let n = byte_count(nbytes, "cudaMemcpyAsync")?;
        let address = device_address(dst, offset)?;
        self.lib.set_device(self.device)?;
        self.lib
            .memcpy_h2d(address, &src.as_slice()[..n], &self.stream)
    }

    fn copy_to_device_2d(
        &self,
        src: &dyn PinnedBuffer,
        dst: DevicePtr,
        shape: Copy2D,
    ) -> Result<(), CudaError> {
        shape.check(src.capacity(), dst.nbytes)?;
        if shape.width == 0 || shape.height == 0 {
            return Ok(());
        }
        const CALL: &str = "cudaMemcpy2DAsync";
        let src_offset = byte_count(shape.src_offset, CALL)?;
        let address = device_address(dst, shape.dst_offset)?;
        self.lib.set_device(self.device)?;
        self.lib.memcpy_2d_h2d(
            address,
            byte_count(shape.dst_pitch, CALL)?,
            &src.as_slice()[src_offset..],
            byte_count(shape.src_pitch, CALL)?,
            byte_count(shape.width, CALL)?,
            byte_count(shape.height, CALL)?,
            &self.stream,
        )
    }

    fn copy_to_host(
        &self,
        src: DevicePtr,
        offset: u64,
        nbytes: u64,
        dst: &mut dyn PinnedBuffer,
    ) -> Result<(), CudaError> {
        check_range(nbytes, offset, src.nbytes)?;
        check_range(nbytes, 0, dst.capacity())?;
        let n = byte_count(nbytes, "cudaMemcpyAsync")?;
        let address = device_address(src, offset)?;
        self.lib.set_device(self.device)?;
        self.lib
            .memcpy_d2h(&mut dst.as_mut_slice()[..n], address, &self.stream)
    }

    fn synchronize(&self) -> Result<(), CudaError> {
        self.stream.synchronize()
    }

    fn record(&self) -> Result<Box<dyn Fence>, CudaError> {
        self.lib.set_device(self.device)?;
        Ok(Box::new(EventFence(ffi::Event::record(
            &self.lib,
            &self.stream,
        )?)))
    }

    fn free_bytes(&self, device: u32) -> Result<u64, CudaError> {
        self.lib.set_device(device)?;
        let info = self.lib.mem_get_info();
        if device != self.device {
            // Restore this thread's device even when the query failed.
            self.lib.set_device(self.device)?;
        }
        info.map(|(free, _total)| free)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_inside_the_buffer_pass() {
        assert!(check_range(0, 0, 0).is_ok());
        assert!(check_range(8, 0, 8).is_ok());
        assert!(check_range(4, 4, 8).is_ok());
        assert!(check_range(0, 8, 8).is_ok());
    }

    #[test]
    fn overruns_are_typed_and_never_overflow() {
        assert!(matches!(
            check_range(4, 6, 8),
            Err(CudaError::Overrun {
                nbytes: 4,
                offset: 6,
                capacity: 8
            })
        ));
        assert!(matches!(
            check_range(1, 8, 8),
            Err(CudaError::Overrun {
                nbytes: 1,
                offset: 8,
                capacity: 8
            })
        ));
        assert!(matches!(
            check_range(2, u64::MAX, 8),
            Err(CudaError::Overrun {
                nbytes: 2,
                offset: u64::MAX,
                capacity: 8
            })
        ));
    }

    #[test]
    fn device_address_adds_the_checked_offset() {
        let ptr = DevicePtr {
            address: 0x7000_0000,
            nbytes: 64,
            device: 0,
        };
        assert_eq!(device_address(ptr, 16).ok(), Some(0x7000_0010));
        assert!(matches!(
            device_address(
                DevicePtr {
                    address: usize::MAX,
                    ..ptr
                },
                1
            ),
            Err(CudaError::Overrun { .. })
        ));
    }

    #[test]
    fn missing_library_names_libcudart() {
        match CudaRuntime::load_from(&["/nonexistent/libcudart.so.99"], 0) {
            Err(CudaError::Unavailable { library, reason }) => {
                assert_eq!(library, "libcudart");
                assert!(reason.contains("/nonexistent/libcudart.so.99"), "{reason}");
            }
            other => panic!("expected Unavailable, got {other:?}"),
        }
    }
}
