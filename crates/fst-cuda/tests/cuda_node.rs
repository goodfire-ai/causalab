//! Integration tests against a real GPU. All `#[ignore]`d and additionally
//! gated on `FST_CUDA_TESTS=1`, so `cargo test` stays green on a laptop:
//!
//! ```text
//! FST_CUDA_TESTS=1 cargo test -p fst-cuda --test cuda_node -- --ignored --nocapture
//! ```
//!
//! Device memory is allocated here with `cudaMalloc` — production code never
//! does; the Python layer owns device buffers. The runtime is dlopen'd a
//! second time for that, independently of the crate under test.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::ffi::c_void;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use fst_cuda::cudart::CudaRuntime;
use fst_cuda::cufile::CuFile;
use fst_cuda::{CudaError, DeviceCopier, DevicePtr, DirectStorage, probe};
use libloading::Library;

const MIB: u64 = 1 << 20;
const GIB: u64 = 1 << 30;

fn gated() -> bool {
    if std::env::var_os("FST_CUDA_TESTS").is_none() {
        eprintln!("FST_CUDA_TESTS not set; skipping");
        return false;
    }
    true
}

/// A `cudaMalloc` allocation for the tests, freed on drop.
struct DeviceMem {
    ptr: DevicePtr,
    free: unsafe extern "C" fn(*mut c_void) -> i32,
    _lib: Library,
}

impl DeviceMem {
    fn alloc(nbytes: u64) -> DeviceMem {
        type Malloc = unsafe extern "C" fn(*mut *mut c_void, usize) -> i32;
        type Free = unsafe extern "C" fn(*mut c_void) -> i32;
        type Memset = unsafe extern "C" fn(*mut c_void, i32, usize) -> i32;
        type DeviceSync = unsafe extern "C" fn() -> i32;
        let lib = [
            "libcudart.so.13",
            "libcudart.so.12",
            "/usr/local/cuda/lib64/libcudart.so",
        ]
        .iter()
        .find_map(|name| unsafe { Library::new(*name) }.ok())
        .expect("libcudart for the test's own cudaMalloc");
        let malloc: Malloc = *unsafe { lib.get::<Malloc>("cudaMalloc") }.unwrap();
        let free: Free = *unsafe { lib.get::<Free>("cudaFree") }.unwrap();
        let memset: Memset = *unsafe { lib.get::<Memset>("cudaMemset") }.unwrap();
        let sync: DeviceSync = *unsafe { lib.get::<DeviceSync>("cudaDeviceSynchronize") }.unwrap();
        let mut raw: *mut c_void = std::ptr::null_mut();
        let code = unsafe { malloc(&mut raw, nbytes as usize) };
        assert_eq!(code, 0, "cudaMalloc({nbytes})");
        assert_eq!(unsafe { memset(raw, 0xEE, nbytes as usize) }, 0);
        // cudaMemset runs on the legacy default stream, which the copier's
        // non-blocking stream does not order against: wait for it here.
        assert_eq!(unsafe { sync() }, 0);
        DeviceMem {
            ptr: DevicePtr {
                address: raw as usize,
                nbytes,
                device: 0,
            },
            free,
            _lib: lib,
        }
    }
}

impl Drop for DeviceMem {
    fn drop(&mut self) {
        unsafe { (self.free)(self.ptr.address as *mut c_void) };
    }
}

/// Deterministic pseudo-random bytes.
fn pattern(seed: u64, len: usize) -> Vec<u8> {
    let mut x = seed | 1;
    let mut out = vec![0u8; len];
    for chunk in out.chunks_mut(8) {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        let bytes = x.to_le_bytes();
        chunk.copy_from_slice(&bytes[..chunk.len()]);
    }
    out
}

fn gbps(nbytes: u64, elapsed: Duration) -> f64 {
    nbytes as f64 / elapsed.as_secs_f64() / 1e9
}

fn runtime() -> CudaRuntime {
    CudaRuntime::load(0).unwrap_or_else(|e| panic!("CudaRuntime::load: {e}"))
}

/// Copy a device range back to the host through the copier under test.
fn readback(rt: &CudaRuntime, src: DevicePtr, offset: u64, nbytes: u64) -> Vec<u8> {
    let mut back = rt.alloc_pinned(nbytes).unwrap();
    rt.copy_to_host(src, offset, nbytes, back.as_mut()).unwrap();
    rt.synchronize().unwrap();
    back.as_slice().to_vec()
}

#[test]
#[ignore = "needs a CUDA device; FST_CUDA_TESTS=1 cargo test -p fst-cuda -- --ignored"]
fn probe_sees_the_node() {
    if !gated() {
        return;
    }
    let a = probe();
    eprintln!("{a}");
    assert!(a.cudart.is_ok(), "{a:?}");
    assert!(a.cufile.is_ok(), "{a:?}");
    assert!(a.devices >= 1, "{a:?}");
}

#[test]
#[ignore = "needs a CUDA device; FST_CUDA_TESTS=1 cargo test -p fst-cuda -- --ignored"]
fn pinned_round_trip_through_device() {
    if !gated() {
        return;
    }
    let rt = runtime();
    eprintln!("cudart {} version {:?}", rt.library(), rt.runtime_version());

    let t = Instant::now();
    let mut pinned = rt.alloc_pinned(16 * MIB).unwrap();
    eprintln!("alloc_pinned(16 MiB): {:?}", t.elapsed());
    assert_eq!(pinned.capacity(), 16 * MIB);
    assert!(
        pinned.as_slice().iter().all(|&b| b == 0),
        "pinned memory is zeroed"
    );

    let data = pattern(7, (16 * MIB) as usize);
    pinned.as_mut_slice().copy_from_slice(&data);

    let dev = DeviceMem::alloc(32 * MIB);
    rt.copy_to_device(pinned.as_ref(), 16 * MIB, dev.ptr, 8 * MIB)
        .unwrap();
    let fence = rt.record().unwrap();
    fence.wait().unwrap();
    fence.wait().unwrap(); // waiting twice is fine

    let back = readback(&rt, dev.ptr, 8 * MIB, 16 * MIB);
    assert!(back == data, "device bytes differ from the pinned source");
    let untouched = readback(&rt, dev.ptr, 0, 8 * MIB);
    assert!(
        untouched.iter().all(|&b| b == 0xEE),
        "bytes before the copy were touched"
    );

    // A partial copy: the first 4 KiB only, to offset 0.
    rt.copy_to_device(pinned.as_ref(), 4096, dev.ptr, 0)
        .unwrap();
    rt.synchronize().unwrap();
    assert_eq!(readback(&rt, dev.ptr, 0, 4096), data[..4096]);

    // Overruns are refused before any call.
    assert!(matches!(
        rt.copy_to_device(pinned.as_ref(), 16 * MIB, dev.ptr, 17 * MIB),
        Err(CudaError::Overrun { .. })
    ));
    assert!(matches!(
        rt.copy_to_device(pinned.as_ref(), 16 * MIB + 1, dev.ptr, 0),
        Err(CudaError::Overrun { .. })
    ));

    let free = rt.free_bytes(0).unwrap();
    eprintln!("free_bytes(0): {:.1} GiB", free as f64 / GIB as f64);
    assert!(
        free > GIB && free < 1024 * GIB,
        "free bytes {free} implausible"
    );
}

#[test]
#[ignore = "needs a CUDA device; FST_CUDA_TESTS=1 cargo test -p fst-cuda -- --ignored"]
fn h2d_bandwidth_one_gib() {
    if !gated() {
        return;
    }
    let rt = runtime();
    let t = Instant::now();
    let mut pinned = rt.alloc_pinned(GIB).unwrap();
    eprintln!("alloc_pinned(1 GiB): {:?}", t.elapsed());
    let data = pattern(11, GIB as usize);
    pinned.as_mut_slice().copy_from_slice(&data);
    let dev = DeviceMem::alloc(GIB);

    let mut best = 0f64;
    for i in 0..4 {
        let t = Instant::now();
        rt.copy_to_device(pinned.as_ref(), GIB, dev.ptr, 0).unwrap();
        rt.synchronize().unwrap();
        let gb = gbps(GIB, t.elapsed());
        eprintln!("H2D 1 GiB run {i}: {gb:.1} GB/s ({:?})", t.elapsed());
        if i > 0 {
            best = best.max(gb);
        }
    }
    eprintln!("H2D best of warm runs: {best:.1} GB/s");
    assert!(best > 5.0, "H2D bandwidth {best:.1} GB/s implausibly low");

    let mut back = rt.alloc_pinned(GIB).unwrap();
    let t = Instant::now();
    rt.copy_to_host(dev.ptr, 0, GIB, back.as_mut()).unwrap();
    rt.synchronize().unwrap();
    eprintln!(
        "D2H 1 GiB: {:.1} GB/s ({:?})",
        gbps(GIB, t.elapsed()),
        t.elapsed()
    );
    assert!(back.as_slice() == data.as_slice());
}

fn write_file(path: &Path, data: &[u8]) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let mut f = File::create(path).unwrap();
    f.write_all(data).unwrap();
    f.sync_all().unwrap();
}

/// Read `path` fully and in an unaligned sub-range through cuFile, compare.
fn cufile_round_trip(
    cf: &CuFile,
    rt: &CudaRuntime,
    path: &Path,
    data: &[u8],
) -> Result<f64, CudaError> {
    let len = data.len() as u64;
    let dev = DeviceMem::alloc(len + 8192);

    let t = Instant::now();
    cf.read_into_device(path, 0, len, dev.ptr, 4096)?;
    let gb = gbps(len, t.elapsed());
    eprintln!(
        "cuFileRead {} MiB from {}: {gb:.2} GB/s ({:?})",
        len / MIB,
        path.display(),
        t.elapsed()
    );
    assert!(
        readback(rt, dev.ptr, 4096, len) == data,
        "full read differs"
    );
    assert!(readback(rt, dev.ptr, 0, 4096).iter().all(|&b| b == 0xEE));

    // Unaligned everything: odd file offset, odd length, odd buffer offset.
    let (foff, n, boff) = (4097u64, 1_000_003u64, 13u64);
    let dev2 = DeviceMem::alloc(2 * MIB);
    cf.read_into_device(path, foff, n, dev2.ptr, boff)?;
    let got = readback(rt, dev2.ptr, boff, n);
    assert!(
        got == data[foff as usize..(foff + n) as usize],
        "unaligned read differs"
    );

    // Small read: below the registration threshold.
    let dev3 = DeviceMem::alloc(64 * 1024);
    cf.read_into_device(path, 12345, 40_000, dev3.ptr, 0)?;
    assert!(readback(rt, dev3.ptr, 0, 40_000) == data[12345..12345 + 40_000]);
    Ok(gb)
}

fn local_dir() -> PathBuf {
    PathBuf::from(format!("/tmp/fst-cuda-tests-{}", std::process::id()))
}

#[test]
#[ignore = "needs a CUDA device; FST_CUDA_TESTS=1 cargo test -p fst-cuda -- --ignored"]
fn cufile_reads_from_local_xfs_or_fails_typed() {
    if !gated() {
        return;
    }
    let rt = runtime();
    let cf = CuFile::load().unwrap_or_else(|e| panic!("CuFile::load: {e}"));
    eprintln!("cufile {} version {:?}", cf.library(), cf.version());

    let dir = local_dir();
    let path = dir.join("weights.bin");
    let data = pattern(3, (256 * MIB) as usize);
    write_file(&path, &data);
    let dev = DeviceMem::alloc(GIB);

    // Checks that run before any cuFile call, whatever the mount does.
    // Past the end: typed.
    assert!(matches!(
        cf.read_into_device(&path, 256 * MIB - 10, 20, dev.ptr, 0),
        Err(CudaError::ShortRead {
            wanted: 20,
            got: 10,
            ..
        })
    ));
    // Missing file: typed.
    assert!(matches!(
        cf.read_into_device(&dir.join("missing.bin"), 0, 20, dev.ptr, 0),
        Err(CudaError::Register { .. })
    ));
    // Overrun of the destination: typed, before opening anything.
    assert!(matches!(
        cf.read_into_device(&path, 0, GIB + 1, dev.ptr, 0),
        Err(CudaError::Overrun { .. })
    ));
    // One open: the short read opened before fstat, the missing file never did.
    assert_eq!(cf.stats().direct_opens + cf.stats().plain_opens, 1);

    let result = cufile_round_trip(&cf, &rt, &path, &data);
    let stats = cf.stats();
    eprintln!("stats after /tmp reads: {stats:?}");
    match &result {
        Ok(_) => {
            assert_eq!(
                stats.buffers_registered + stats.buffer_registrations_skipped,
                3
            );
            // Bandwidth on a 1 GiB file, from a fresh open each time.
            let big = dir.join("big.bin");
            let big_data = pattern(5, GIB as usize);
            write_file(&big, &big_data);
            for i in 0..3 {
                let t = Instant::now();
                cf.read_into_device(&big, 0, GIB, dev.ptr, 0).unwrap();
                eprintln!(
                    "cuFileRead 1 GiB /tmp run {i}: {:.2} GB/s ({:?})",
                    gbps(GIB, t.elapsed()),
                    t.elapsed()
                );
            }
            assert!(readback(&rt, dev.ptr, 0, GIB) == big_data);
        }
        // an H100 host whose /tmp is xfs on md RAID10 (over dm), where
        // libcufile 1.15 fails "error getting volume attributes" for that
        // block device inside cuFileHandleRegister, so the handle is refused
        // with CU_FILE_HANDLE_NOT_REGISTERED (5027) even in compat mode.
        Err(e @ CudaError::Register { reason, .. }) => {
            eprintln!("/tmp: cuFile refuses the file, typed: {e}");
            assert!(reason.contains("cuFileHandleRegister"), "{reason}");
        }
        Err(e) => panic!("/tmp: unexpected error kind: {e}"),
    }

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
#[ignore = "needs a CUDA device; FST_CUDA_TESTS=1 cargo test -p fst-cuda -- --ignored"]
fn cufile_reads_from_nfs_home_or_fails_typed() {
    if !gated() {
        return;
    }
    let Some(home) = std::env::var_os("HOME") else {
        eprintln!("no HOME; skipping");
        return;
    };
    let rt = runtime();
    let cf = CuFile::load().unwrap_or_else(|e| panic!("CuFile::load: {e}"));
    let dir = PathBuf::from(home)
        .join(".cache")
        .join(format!("fst-cuda-tests-{}", std::process::id()));
    let path = dir.join("weights.bin");
    let data = pattern(9, (256 * MIB) as usize);
    write_file(&path, &data);

    let result = cufile_round_trip(&cf, &rt, &path, &data);
    eprintln!("stats after NFS reads: {:?}", cf.stats());
    match &result {
        Ok(gb) => eprintln!("NFS: cuFile read works (compat mode), {gb:.2} GB/s"),
        Err(e @ (CudaError::CuFile { .. } | CudaError::Register { .. })) => {
            eprintln!("NFS: cuFile read fails typed: {e}")
        }
        Err(e) => panic!("NFS: unexpected error kind: {e}"),
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

/// A dropped pinned buffer is retained and the next request of the same size
/// gets its memory back without another `cudaHostAlloc`.
#[test]
#[ignore]
fn pinned_buffers_are_pooled_by_size() {
    if !gated() {
        return;
    }
    let rt = CudaRuntime::load(0).expect("runtime");
    assert_eq!(rt.pooled_pinned(), (0, 0));
    let first = rt.alloc_pinned(16 * MIB).expect("alloc");
    let address = first.as_slice().as_ptr() as usize;
    drop(first);
    assert_eq!(rt.pooled_pinned(), (1, 16 * MIB));
    let other = rt.alloc_pinned(8 * MIB).expect("alloc");
    assert_eq!(
        rt.pooled_pinned(),
        (1, 16 * MIB),
        "a different size does not take it"
    );
    let again = rt.alloc_pinned(16 * MIB).expect("alloc");
    assert_eq!(again.as_slice().as_ptr() as usize, address);
    assert_eq!(rt.pooled_pinned(), (0, 0));
    drop(other);
    drop(again);
    assert_eq!(rt.pooled_pinned(), (2, 24 * MIB));
}
