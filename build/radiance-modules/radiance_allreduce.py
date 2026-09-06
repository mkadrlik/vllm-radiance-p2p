#!/usr/bin/env python3
"""radiance custom all-reduce: one-shot P2P-BAR AR for dual R9700 (gfx1201, TP=2, PCIe).

Mirrors vLLM's CustomAllreduce (ca_comm) interface, so it slots into CudaCommunicator.all_reduce's
existing size-gated dispatch. should_custom_ar returns True only for small messages in the win band;
larger or other messages fall through to RCCL.

Kernel: radiance_ar_ext (pybind11 + HIP .so), PUSH model. Each rank writes its input into the peer's
IPC scratch, then reduces (local input + peer data now in my scratch) into out. Only scratch and flags
are IPC-shared; input/output stay local, so no cudagraph buffer registration is needed. cudagraph-safe:
the sequence number is a device-resident counter the kernel increments (a host-passed seq would freeze
at capture), and scratch is double-buffered by seq parity. Multi-block: each block owns a contiguous
chunk and drives its own flag handshake, spreading the push across CUs (PCIe) and the reduce across CUs.

Env:
  RADIANCE_FAST_REDUCE (1)     1 = install the kernel on the TP group, 0 = RCCL
  RADIANCE_AR_MAX_KB (32768)   messages up to this size use the kernel; larger fall back to RCCL
  RADIANCE_AR_QUANT (1)        1 = quantize the payload to block-scaled fp8 (e4m3) for large messages
  RADIANCE_AR_QUANT_MIN_KB (128) fp8 only for messages >= this size (smaller stay on the bf16 kernel)
"""
import os
import sys
from contextlib import contextmanager

import torch
import torch.distributed as dist

_DTYPE_CODE = {torch.bfloat16: 0, torch.float16: 1, torch.float32: 2}

# Diagnostics (RADIANCE_AR_DIAG=1): bake-count + seq/flag trajectory logging to
# root-cause replay-time wedges. Default OFF; zero hot-path cost when off.
_DIAG = os.environ.get("RADIANCE_AR_DIAG", "0") == "1"
import threading


def _log(msg):
    sys.stderr.write(f"[radiance] {msg}\n")
    sys.stderr.flush()


def _diag(msg):
    if _DIAG:
        sys.stderr.write(f"[ar-diag] {msg}\n")
        sys.stderr.flush()


class RadianceAllreduce:
    """Drop-in for vLLM's ca_comm (CustomAllreduce). Same public surface:
    `disabled`, `should_custom_ar(inp)`, `custom_all_reduce(inp)`, `capture()`."""

    def __init__(self, group, device):
        self.disabled = True
        self.group = group
        try:
            self.world_size = dist.get_world_size(group)
            self.rank = dist.get_rank(group)
        except Exception as e:
            _log(f"custom AR: no process group ({e!r})")
            return
        if self.world_size != 2:
            _log(f"custom AR disabled: world_size={self.world_size} (only 2 supported)")
            return

        try:
            import radiance_ar_ext as ext
        except Exception as e:
            _log(f"custom AR disabled: radiance_ar_ext import failed ({e!r})")
            return
        self._ext = ext

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        self.device = device
        # 32MB covers decode and prefill messages; larger falls back to RCCL. Buffer cost is
        # 2*max_bytes/rank, trivial on 32GB. The kernel is bit-identical to RCCL at every size.
        self.max_bytes = int(os.environ.get("RADIANCE_AR_MAX_KB", "32768")) * 1024
        self.max_bytes = (self.max_bytes // 16) * 16     # 16B (uint4) alignment
        self.slot16 = self.max_bytes // 16               # per-slot capacity in 16B words
        self.drain = 3        # s_wait_storecnt drain (correct for fine-grained/uncached scratch)
        self.acq = 0          # no explicit acquire fence (uncached reads are already fresh)
        self.nt = 1024        # threads/block (tuned)
        # Runtime spin cap (0 = kernel default ~4000s). Short values turn a diverged
        # handshake into a fast, logged abort instead of an invisible multi-hour wedge.
        self.spin_max = int(os.environ.get("RADIANCE_AR_SPIN_MAX", "0"))
        self.min_nb = 4       # block count scales with message size, clamped to [min_nb, max_nb]
        self.max_nb = min(24, int(ext.MAX_BLOCKS))
        self.words_per_block = 1400   # 16B words/block target for the block-count heuristic
        fine = True           # fine-grained (uncached) IPC scratch: posts straight to the fabric
        maxb = int(ext.MAX_BLOCKS)

        try:
            torch.cuda.set_device(self.device)
            # double-buffered scratch (2 slots) + per-block data flags + per-block
            # done flags (slot-reuse guard), all IPC-shared
            self._scratch, sc_h, fine_used = self._alloc(2 * self.max_bytes, fine)
            self._flags, fl_h, _ = self._alloc(maxb * 4, fine)
            self._done, dn_h, _ = self._alloc(maxb * 4, fine)
            sc_handles = [None] * self.world_size
            fl_handles = [None] * self.world_size
            dn_handles = [None] * self.world_size
            dist.all_gather_object(sc_handles, sc_h, group=group)
            dist.all_gather_object(fl_handles, fl_h, group=group)
            dist.all_gather_object(dn_handles, dn_h, group=group)
            peer = 1 - self.rank
            self._peer_scratch = ext.open_shared(sc_handles[peer])
            self._peer_flags = ext.open_shared(fl_handles[peer])
            self._peer_done = ext.open_shared(dn_handles[peer])
            # per-BLOCK device-resident seq counters (kernel increments -> replay-safe).
            # INVARIANT (enforced by the capture gate in should_custom_ar): the kernel
            # ONLY ever executes from graph REPLAY of vLLM-captured CUDA graphs. Eager
            # launches (which are where rank-divergent calls happen: dynamo/inductor
            # rank-local state, warmups, profiling) are gated to RCCL, and captured
            # kernels do not execute at capture time. vLLM's graph dispatch is driven
            # by the broadcast scheduler output, so both ranks replay the identical
            # graph sequence and seq_ctr[b] stays in lockstep.
            # Upper half (offset MAX_BLOCKS) is the spin-abort log: a kernel whose
            # peer never posted its seq records that seq there (radiance_ar_ext.hip).
            self._seq = torch.zeros(2 * maxb, dtype=torch.int32, device=self.device)
        except Exception as e:
            _log(f"custom AR disabled: IPC setup failed ({e!r})")
            return

        self.disabled = False
        _log(f"custom all-reduce INSTALLED (rank={self.rank} max={self.max_bytes // 1024}KB "
             f"finegrained={fine_used} drain={self.drain} acq={self.acq} nt={self.nt} "
             f"nb={self.min_nb}..{self.max_nb})")

        # ---- diagnostics (RADIANCE_AR_DIAG=1) --------------------------------
        # A wedged spin kernel is invisible on the host: the rank that wedges is
        # stuck inside a collective with no logs. The device-resident seq/flag
        # trajectory IS the ground truth: seq_ctrs[b] advances on every launch
        # (replays included), my_flags[b] advances only when the PEER posts.
        # seq>flag locally == this rank is (or was) spinning on block b. A
        # seq_ctr delta between ranks == the SPMD lockstep assumption broke
        # (rank-divergent AR call sequence). A background sampler dumps both
        # every few seconds so the wedge state is captured after the hang.
        self._calls = 0          # eager host-side invocations (bakes do NOT tick this)
        self._bakes = 0          # kernels baked into graphs during capture

        # Optional fp8 payload path (RADIANCE_AR_QUANT). Additive; shares this class's scratch,
        # flags and seq counters. Only large (bandwidth-bound) messages take it; smaller ones keep
        # the exact bf16 kernel above. On by default; set 0 for the exact (RCCL-identical) bf16 path.
        self.ar_quant = os.environ.get("RADIANCE_AR_QUANT", "1") == "1"
        self.quant_min_bytes = int(os.environ.get("RADIANCE_AR_QUANT_MIN_KB", "128")) * 1024
        self.qnt = 1024        # threads/block for the fp8 push (tuned; saturates PCIe)
        self.qmax_nb = 48      # block cap for the fp8 path (tuned)
        self._qext = None
        self._qgroup = 0
        if self.ar_quant:
            try:
                import radiance_ar_quant_ext as qext
                self._qext = qext
                self._qgroup = int(qext.GROUP)
                _log(f"AR_QUANT ON (fp8 payload; min={self.quant_min_bytes // 1024}KB "
                     f"group={self._qgroup} nt={self.qnt} max_nb={self.qmax_nb})")
            except Exception as e:
                _log(f"AR_QUANT disabled: radiance_ar_quant_ext import failed ({e!r})")
                self.ar_quant = False

        if _DIAG:
            self._maxb = maxb
            self._stop = threading.Event()
            self._sampler = threading.Thread(target=self._diag_loop, daemon=True)
            self._sampler.start()

    def _diag_loop(self):
        # Async side-stream D2H into pinned RAM. A plain synchronous hipMemcpy from
        # this thread blocks holding the GIL until the compute queue drains — i.e.
        # until a spin kernel times out — which starves the main thread's next
        # replay and manufactures the very wedge being observed. The async copy
        # rides the DMA engine; ev.synchronize() releases the GIL.
        # Short RADIANCE_AR_SPIN_MAX is still required in diagnostic windows: with
        # the default 4000s spin a wedged handshake never yields inside the window.
        prev = None
        stalls = 0
        read_errs = 0
        # HIP device context is per-thread: without this, rank 1's sampler reads
        # pointers belonging to cuda:1 from a thread defaulted to cuda:0.
        torch.cuda.set_device(self.device)
        st = torch.cuda.Stream()
        seq_h = torch.zeros(2 * self._maxb, dtype=torch.int32, pin_memory=True)
        flg_h = torch.zeros(self._maxb, dtype=torch.int32, pin_memory=True)
        dn_h = torch.zeros(self._maxb, dtype=torch.int32, pin_memory=True)
        seq_p = self._seq.data_ptr()
        flg_p = self._flags
        dn_p = self._done
        while not self._stop.wait(3.0):
            try:
                sh = st.cuda_stream
                self._ext.copy_u32_async(seq_h.data_ptr(), seq_p, 2 * self._maxb, sh)
                self._ext.copy_u32_async(flg_h.data_ptr(), flg_p, self._maxb, sh)
                self._ext.copy_u32_async(dn_h.data_ptr(), dn_p, self._maxb, sh)
                ev = torch.cuda.Event()
                ev.record(st)
                ev.synchronize()
                seq = seq_h.tolist()
                flg = [x & 0xFFFFFFFF for x in flg_h.tolist()]
                dn = [x & 0xFFFFFFFF for x in dn_h.tolist()]
            except Exception as e:
                # capture on the device makes foreign-stream copies illegal
                # (startup graph capture races this thread). Survive, keep sampling.
                read_errs += 1
                if read_errs <= 3 or read_errs % 100 == 0:
                    _diag(f"sampler read failed ({read_errs}x): {e!r}")
                continue
            abort = [(b, seq[self._maxb + b]) for b in range(self._maxb) if seq[self._maxb + b]]
            seq = seq[:self._maxb]
            if abort:
                # raw values: high bit set = done-wait (slot reuse) timeout,
                # clear = data-wait timeout; low 31 bits = the seq that gave up.
                raw = " ".join(f"b{b}:0x{v:x}" for b, v in abort)
                _diag(f"SPIN-ABORT {raw} — peer AR sequence diverged; "
                      f"seq[0:8]={seq[:8]} my_flags[0:8]={flg[:8]} my_done[0:8]={dn[:8]}")
                return
            sig = (tuple(seq), tuple(flg))
            if sig == prev:
                stalls += 1
                if stalls == 2:   # one quiet sample set -> log trajectory once
                    spin = [b for b in range(self._maxb) if seq[b] > flg[b]]
                    _diag(f"IDLE seq[0:8]={seq[:8]} my_flags[0:8]={flg[:8]} "
                          f"my_done[0:8]={dn[:8]} "
                          f"spinning_blocks={spin} calls={self._calls} bakes={self._bakes}")
            else:
                stalls = 0
                moved = [b for b in range(self._maxb)
                         if prev is None or seq[b] != prev[0][b] or flg[b] != prev[1][b]]
                _diag(f"TRAJ seq[0:8]={seq[:8]} my_flags[0:8]={flg[:8]} my_done[0:8]={dn[:8]} "
                      f"moved={moved[:12]} calls={self._calls} bakes={self._bakes}")
                prev = sig

    def _alloc(self, nbytes, fine):
        """Allocate a shared buffer, falling back fine->coarse if fine-grained IPC fails.
        Deterministic across ranks (same env/HW), so the fallback stays symmetric."""
        try:
            ptr, h = self._ext.alloc_shared(nbytes, fine)
            return ptr, h, fine
        except Exception as e:
            if fine:
                _log(f"fine-grained alloc failed ({e!r}); falling back to coarse")
                ptr, h = self._ext.alloc_shared(nbytes, False)
                return ptr, h, False
            raise

    def should_custom_ar(self, inp: torch.Tensor) -> bool:
        if self.disabled:
            return False
        # FAST-REDUCE ONLY INSIDE CUDA-GRAPH CAPTURE. An eager launch here is a
        # fire-and-forget spin kernel: if the ranks ever disagree on the eager
        # all_reduce sequence (dynamo recompile asymmetry, rank-0-only work,
        # profile runs), the peer's kernel spins on my_flags[b] < seq for
        # ~4000s (RADIANCE_SPIN_MAX uncached PCIe reads) and the next device-wide
        # sync — which torch.cuda.graph() capture_begin performs — wedges that
        # rank forever. That is the MTP drafter capture deadlock: hang at
        # "Capturing CUDA graphs 0/5", both ranks watchdog on the next NCCL
        # collective. Captured kernels do not execute at capture time, so they
        # cannot spin during capture, and at replay both ranks execute the
        # identical scheduler-driven graph sequence, which is exactly the SPMD
        # lockstep assumption the per-block device-resident seq counters need.
        # Eager all_reduces (warmup, profiling, >max-capture prefill batches)
        # fall through to RCCL: symmetric and deadlock-free.
        if not torch.cuda.is_current_stream_capturing():
            return False
        if inp.dtype not in _DTYPE_CODE:
            return False
        nbytes = inp.numel() * inp.element_size()
        if nbytes == 0 or nbytes % 16 != 0:  # uint4 push alignment
            return False
        if nbytes > self.max_bytes:
            return False
        return inp.is_contiguous()

    def _nblocks(self, n16: int) -> int:
        # FIXED grid: every launch uses max_nb blocks regardless of message size.
        # The kernel's fixed word partition (block b owns [b*chunk, (b+1)*chunk),
        # chunk=slot16/nb) is only sound when nb is constant: a size-proportional
        # grid makes block ranges shift between graphs, so one rank's write at
        # launch s overlaps the peer's different-index read at s-2 and the
        # per-block done guard misses it. Empty-range blocks just handshake.
        return self.max_nb

    def _quant_ok(self, inp: torch.Tensor) -> bool:
        # fp8 path only for large (bandwidth-bound) bf16/fp16 messages that tile the scale group.
        if not self.ar_quant or self._qext is None:
            return False
        if inp.dtype not in (torch.bfloat16, torch.float16):
            return False
        nbytes = inp.numel() * inp.element_size()
        if nbytes < self.quant_min_bytes:
            return False
        return inp.numel() % self._qgroup == 0

    def _nblocks_q(self, nbytes: int) -> int:
        nb = nbytes // (self.words_per_block * 16)
        if nb < self.min_nb:
            nb = self.min_nb
        if nb > self.qmax_nb:
            nb = self.qmax_nb
        return nb

    def custom_all_reduce(self, inp: torch.Tensor):
        if not self.should_custom_ar(inp):
            return None
        out = torch.empty_like(inp)
        nbytes = inp.numel() * inp.element_size()
        stream = torch.cuda.current_stream().cuda_stream
        if _DIAG:
            if torch.cuda.is_current_stream_capturing():
                self._bakes += 1
                if self._bakes % 50 == 1:
                    _diag(f"BAKE #{self._bakes} nbytes={nbytes} nb={self._nblocks(nbytes // 16)}")
            else:
                self._calls += 1
        if self._quant_ok(inp):
            self._qext.all_reduce_fp8(
                self._peer_scratch, self._scratch, self._peer_flags, self._flags,
                self._seq.data_ptr(), self.max_bytes, self.max_bytes // 2,
                inp.data_ptr(), out.data_ptr(), inp.numel(),
                _DTYPE_CODE[inp.dtype], stream, self._nblocks_q(nbytes), self.qnt,
                self.drain, self.acq,
            )
        else:
            self._ext.all_reduce_mb(
                self._peer_scratch, self._scratch, self._peer_flags, self._flags,
                self._peer_done, self._done,
                self._seq.data_ptr(), self.slot16,
                inp.data_ptr(), out.data_ptr(), inp.numel(),
                _DTYPE_CODE[inp.dtype], stream, self._nblocks(nbytes // 16), self.nt,
                self.drain, self.acq, self.spin_max,
            )
        return out

    @contextmanager
    def capture(self):
        # No graph-buffer registration needed (push model: only fixed IPC scratch/flags
        # are peer-shared; input/output stay local & are captured with stable addresses).
        yield


def install_custom_ar():
    """Wrap CudaCommunicator.all_reduce so small TP messages take the one-shot kernel and everything
    else falls through to RCCL. We wrap rather than replace ca_comm because vLLM's RocmAiter fusion
    pass asserts isinstance(ca_comm, CustomAllreduce), and ca_comm is already inert on ROCm. Env-gated
    by RADIANCE_FAST_REDUCE, idempotent."""
    if os.environ.get("RADIANCE_FAST_REDUCE", "1") != "1":
        return
    from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator

    if getattr(CudaCommunicator, "_radiance_ar_patched", False):
        return
    _orig_init = CudaCommunicator.__init__
    _orig_all_reduce = CudaCommunicator.all_reduce

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        self.radiance_comm = None
        try:
            # only the TP group does custom AR (matches vLLM's own gating)
            if "tp" in getattr(self, "unique_name", "") and getattr(self, "world_size", 1) == 2:
                comm = RadianceAllreduce(self.cpu_group, self.device)
                if not comm.disabled:
                    self.radiance_comm = comm
        except Exception as e:
            _log(f"custom AR attach failed: {e!r}")

    def _patched_all_reduce(self, input_):
        rc = getattr(self, "radiance_comm", None)
        if rc is not None and not rc.disabled and rc.should_custom_ar(input_):
            out = rc.custom_all_reduce(input_)
            if out is not None:
                return out
        return _orig_all_reduce(self, input_)

    CudaCommunicator.__init__ = _patched_init
    CudaCommunicator.all_reduce = _patched_all_reduce
    CudaCommunicator._radiance_ar_patched = True
    _log("fast-reduce hook armed (RADIANCE_FAST_REDUCE=1, all_reduce wrap)")
