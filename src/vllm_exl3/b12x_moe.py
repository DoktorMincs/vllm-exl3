"""Optional b12x W4A16 trellis MoE backend for `mul1` EXL3 packs.

Selected only by an explicit ``VLLM_EXL3_MOE_KERNEL=b12x``; every other value
(including ``auto``) leaves the existing native/ExLlamaV3/python-loop dispatch
untouched.  The backend runs NVIDIA's ``b12x`` fused MoE (the CuTe-DSL W4A16
trellis kernel from the hyperflare/flashinfer experiment) against the *live*
EXL3 arenas, zero-copy, with the ``mul1`` codebook decoded by the fp16 arm
added in ``b12x._lib.intrinsics.trellis.packed_dequant_trellis_mul1_to_half2x4``
(bit-exact to ``codebook.cuh`` cb==2; the pre-existing mul1 intrinsic rounds to
E4M3 and is 2.6% rel-L2 on the weights, so it is not used here).

Scope of this backend, by construction:

* routed experts only, uniform K in {2, 3, 4}, one codebook per layer
  (``(False, True)`` = mul1 for every projection);
* gated silu, with or without the DS4.1 clamp (``swiglu_limit = 10.0``); the
  clamp law is b12x's ``swiglu_clamp_after_silu=False`` path, which matches the
  plugin's reference loop exactly;
* no ``expert_map`` (TP-only, not EP) and no mixed-K store;
* above ``VLLM_EXL3_B12X_MIN_M`` rows (default 64): at 8 rows/expert the b12x
  MoE is ~15% faster than the fused ``exl3_moe``, but at decode sizes it is
  5-9x slower than the gemv path, so narrow batches keep the established
  dispatch.

Every entry point degrades instead of raising: a missing import, an
unsupported geometry or a launch failure clears ``layer._exl3_b12x_state`` and
leaves ``layer._exl3_b12x_error`` for the diagnostics line, and the caller
falls back to the paths it already had.
"""

from __future__ import annotations

import os
import sys
from collections import OrderedDict
from functools import lru_cache
from typing import Any

import torch

BACKEND = "b12x"
_MIN_ROWS_DEFAULT = 64
# How many distinct (rows, topk) workspaces to keep, process-wide.
_MAX_CACHED_SHAPES = 4
# Scratch is geometry, not layer: every layer of a pipeline stage has the same
# E/H/I/K, and these buffers are pure workspace (weights are read-only views in
# `prepared`). Sharing them across layers turns 8 copies per rank into one.
# Measured 18/11: per-layer workspaces (8 layers x ~112 MB at M=512) ate the
# whole KV budget and aborted the boot ("16384 needs 0.05 GiB, 0.04 available").
_SHARED: "OrderedDict[tuple, tuple]" = OrderedDict()
# Block sizes whose kernel specialization has been compiled in this process.
# The CuTe cache key has no M in it, but it DOES carry `moe_block_size` (via
# the fc1/fc2 GEMM keys), and `select_route_block_size_m` picks 8 for M<=384
# and 16 for M=512+. A prefill chunk that lands on the other block size would
# then JIT-compile inside a serving step: measured 18/11, a 1744-token prompt
# split into 512/512/512/208 hung the pp_broadcast collective until the NCCL
# watchdog killed the engine. Warm both at load, and let an operator pin one.
_WARMED_BLOCKS: set[tuple] = set()
_WARMUP_ROWS = {8: 256, 16: 512, 32: 1024}



def _logger():
    try:
        from vllm.logger import init_logger

        # The vLLM "vllm." prefix is what carries the configured level and the
        # handler; a bare "vllm_exl3.b12x_moe" logger drops every INFO line
        # (measured 18/11: the whole b12x diagnosis was invisible at boot).
        return init_logger("vllm." + __name__)
    except Exception:  # pragma: no cover - import-time safety
        import logging

        return logging.getLogger(__name__)


def _flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def requested() -> bool:
    """True only when the operator asked for this backend by name."""
    return os.environ.get("VLLM_EXL3_MOE_KERNEL", "auto").strip().lower() == BACKEND


def min_rows() -> int:
    raw = os.environ.get("VLLM_EXL3_B12X_MIN_M", "").strip()
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            pass
    return _MIN_ROWS_DEFAULT


def max_rows() -> int:
    """Upper bound for this backend (0 = no bound)."""
    raw = os.environ.get("VLLM_EXL3_B12X_MAX_M", "").strip()
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            pass
    return 0


def _vendor_dir() -> str | None:
    """Directory that contains the ``b12x`` package (self-contained tree)."""
    for name in ("VLLM_EXL3_B12X_PATH", "VLLM_EXL3_B12X_SRC"):
        raw = os.environ.get(name, "").strip()
        if raw:
            # Accept either the package's parent or a flashinfer/experimental root.
            for cand in (raw, os.path.join(raw, "flashinfer", "experimental")):
                if os.path.isdir(os.path.join(cand, "b12x")):
                    return cand
            return raw
    for cand in (
        os.path.expanduser("~/b12x-vendor"),
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "third_party",
            "b12x",
        ),
    ):
        if os.path.isdir(os.path.join(cand, "b12x")):
            return cand
    return None


@lru_cache(maxsize=1)
def _libs():
    """Import the b12x entry points once; returns None when unavailable."""
    vendor = _vendor_dir()
    if vendor is None:
        return None
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    try:
        from b12x.moe._shared.kernels.w4a16.host import (
            make_w4a16_packed_buffers,
            select_route_block_size_m,
        )
        from b12x.moe._shared.kernels.w4a16.kernel.execution import run_w4a16_moe
        from b12x.moe._shared.kernels.w4a16.prepare import (
            prepare_trellis256_moe_weights,
        )
    except Exception as exc:  # pragma: no cover - depends on the install
        _logger().warning("b12x MoE backend unavailable: %r", exc)
        return None
    return {
        "make_buffers": make_w4a16_packed_buffers,
        "select_block": select_route_block_size_m,
        "run": run_w4a16_moe,
        "prepare": prepare_trellis256_moe_weights,
    }


def available() -> bool:
    return _libs() is not None


def joint_w13_wanted() -> bool:
    """Whether gate/up should share one projection-major arena at load time.

    ``b12x``'s full-rotation path requires ``[2,E,H/16,I/16,16*bits]i16``
    contiguous; the plugin's per-projection arenas cannot be viewed that way,
    so the planner allocates the pair up front.  The per-expert Parameters stay
    views of that storage, which the existing EXL3/exllamav3 paths dereference
    by pointer, so nothing else changes.
    """
    return requested() and _flag("VLLM_EXL3_B12X_JOINT_W13", True) and available()


def plan_joint_w13(
    layer: Any,
    shapes_by_proj: dict[str, dict[int, tuple[int, ...]]],
    ats_eids: dict[str, set[int]] | None = None,
) -> dict[tuple[int, ...], list[int]]:
    """``shape -> [expert_id, ...]`` groups that may share a gate/up arena."""
    if not joint_w13_wanted():
        return {}
    if getattr(layer, "_exl3_mixed_k", False) or getattr(
        layer, "_exl3_has_frac_k", False
    ):
        return {}
    gate = shapes_by_proj.get("gate") or {}
    up = shapes_by_proj.get("up") or {}
    if not gate or set(gate) != set(up):
        return {}
    if any(tuple(up[e]) != tuple(shapes_by_proj["gate"][e]) for e in up):
        return {}
    if (ats_eids or {}).get("gate") or (ats_eids or {}).get("up"):
        return {}
    groups: dict[tuple[int, ...], list[int]] = {}
    for eid in sorted(shapes_by_proj["gate"]):
        shape = tuple(int(x) for x in shapes_by_proj["gate"][eid])
        if len(shape) != 3 or int(shape[-1]) % 16 != 0:
            return {}
        groups.setdefault(shape, []).append(int(eid))
    return groups


def _layer_act_is_plain_silu(layer: Any) -> bool:
    try:
        from .exl3 import MOE_ACT_SILU, resolve_moe_act

        act = getattr(layer, "_exl3_moe_act", None)
        if act is None:
            act = resolve_moe_act(layer)
        return int(act) == int(MOE_ACT_SILU)
    except Exception:
        return False


def prepare_layer(layer: Any) -> dict[str, Any] | None:
    """Build this layer's ``PreparedW4A16MoeWeights`` once, after weight load.

    Returns the state dict (also stored on ``layer._exl3_b12x_state``) or None
    when the layer is out of scope.  Never raises: failures land in
    ``layer._exl3_b12x_error`` and the layer keeps its established dispatch.
    """
    if not requested():
        return None
    if getattr(layer, "_exl3_b12x_tried", False):
        return getattr(layer, "_exl3_b12x_state", None)
    layer._exl3_b12x_tried = True

    def _skip(reason: str) -> None:
        layer._exl3_b12x_state = None
        layer._exl3_b12x_error = reason
        _logger().info("b12x MoE disabled for this layer: %s", reason)

    libs = _libs()
    if libs is None:
        _skip("b12x import failed")
        return None
    try:
        from .exl3 import _uva_trellis_placement_requested

        if _uva_trellis_placement_requested(layer):
            _skip("UVA trellis placement is not qualified for b12x")
            return None
    except Exception:
        pass

    if getattr(layer, "_exl3_gateless", False):
        _skip("gateless layer")
        return None
    # `_exl3_mixed_k` is not the right predicate: EXL3_FORCE_MIXEDK=1 sets it on
    # every layer to route uniform-K layers through exl3_moe_mixedk. The real
    # question is whether this layer's experts share one (Kg,Ku,Kd), which the
    # loader stores in _exl3_k_tuple (None = heterogeneous or fractional).
    if getattr(layer, "_exl3_has_frac_k", False):
        _skip("fractional K")
        return None
    k_tuple = getattr(layer, "_exl3_k_tuple", None)
    if k_tuple is None or int(k_tuple[0]) != int(k_tuple[1]) or int(k_tuple[0]) != int(
        k_tuple[2]
    ):
        _skip(f"K is not uniform across the layer ({k_tuple!r})")
        return None
    flags = tuple(getattr(layer, "_exl3_codebook_flags", (True, False) * 3))
    if flags != (False, True) * 3:
        _skip(f"codebook flags {flags} are not uniformly mul1")
        return None
    if not _layer_act_is_plain_silu(layer):
        _skip("activation is not plain silu")
        return None
    # DS4.1 runs a clamped SwiGLU (``swiglu_limit = 10.0``): gate clamped from
    # above, up clamped symmetrically, silu applied to the clamped gate. b12x
    # implements exactly that law (activation.py::_clamp_swiglu_inputs with
    # swiglu_clamp_after_silu=False), and the plugin's own reference loop does
    # the same, so the clamp travels to the kernel instead of disqualifying the
    # layer. It is constant per layer and the activation is compiled into the
    # launch, so apply() refuses a per-call value that disagrees.
    _limit = getattr(layer, "_exl3_swiglu_limit", None)
    try:
        swiglu_limit = float(_limit) if _limit is not None else None
    except (TypeError, ValueError):
        _skip(f"swiglu clamp limit is not a number ({_limit!r})")
        return None
    if swiglu_limit is not None and swiglu_limit <= 0.0:
        swiglu_limit = None

    joint = getattr(layer, "_exl3_b12x_joint_w13", None) or {}
    if len(joint) != 1:
        _skip(f"needs exactly one joint gate/up arena, got {len(joint)}")
        return None
    shape, base = next(iter(joint.items()))
    down_arenas = list(getattr(layer, "_exl3_down_trellis_arenas", None) or [])
    if len(down_arenas) != 1:
        _skip(f"needs exactly one down arena, got {len(down_arenas)}")
        return None
    w2 = down_arenas[0]

    if base.dim() != 5 or int(base.shape[0]) != 2:
        _skip(f"joint arena has shape {tuple(base.shape)}")
        return None
    num_experts = int(base.shape[1])
    hidden = int(layer._exl3_hidden_size)
    intermediate = int(layer._exl3_intermediate_local)
    bits = int(shape[-1]) // 16
    if tuple(int(x) for x in base.shape[2:]) != (
        int(layer._exl3_hidden_size) // 16,
        int(layer._exl3_intermediate_local) // 16,
        16 * bits,
    ):
        _skip(f"joint arena does not match H/I geometry: {tuple(base.shape)}")
        return None
    if bits not in (2, 3, 4):
        _skip(f"mul1 is defined for K2/K3/K4, got K{bits}")
        return None
    if hidden % 128 or intermediate % 128:
        _skip(f"tile geometry needs multiples of 128 (H={hidden} I={intermediate})")
        return None

    try:
        gate_suh = layer.w13_suh[:, 0].contiguous()
        up_suh = layer.w13_suh[:, 1].contiguous()
        down_svh = layer.w2_svh.contiguous()
        # [gate_svh | up_svh | down_suh] per expert, the order the full-rotation
        # kernel indexes (see b12x prepare: intermediate_rotations is [E, 3I]).
        intermediate_rotations = torch.cat(
            [layer.w13_svh[:, 0], layer.w13_svh[:, 1], layer.w2_suh], dim=1
        ).contiguous()
        prepared = libs["prepare"](
            w13=base,
            w2=w2,
            hidden_size=hidden,
            intermediate_size=intermediate,
            num_experts=num_experts,
            activation="silu",
            fc1_tile_n=128,
            fc2_tile_n=128,
            w13_layout="trellis_t256_proj",
            trellis_bits=bits,
            codebook="mul1",
            params_dtype=torch.float16,
            gate_suh=gate_suh,
            up_suh=up_suh,
            intermediate_rotations=intermediate_rotations,
            down_svh=down_svh,
        )
    except Exception as exc:
        _skip(f"prepare failed: {exc!r}")
        return None

    state = {
        "prepared": prepared,
        "num_experts": num_experts,
        "hidden": hidden,
        "intermediate": intermediate,
        "bits": bits,
        "swiglu_limit": swiglu_limit,
        "device": base.device,
        "buffers": {},
        "blocks": {},
        "scratch": {},
        # Keep every table alive: prepared holds only views/pointers into them.
        "keepalive": (
            base,
            w2,
            gate_suh,
            up_suh,
            down_svh,
            intermediate_rotations,
        ),
        "error": None,
    }
    layer._exl3_b12x_state = state
    layer._exl3_b12x_error = None
    _logger().info(
        "b12x MoE prepared: K=%d E=%d H=%d I=%d", bits, num_experts, hidden, intermediate
    )
    # Compile every route-block specialization now, while no request is in
    # flight: a first-use compile inside a serving step stalls the pipeline
    # long enough to trip the NCCL watchdog (measured 18/11).
    try:
        _warm_blocks(state, libs)
    except Exception as exc:  # a failed warmup must not sink the load
        _logger().warning("b12x MoE warmup failed: %r", exc)
    return state


def _buffers(state: dict[str, Any], libs, m: int, topk: int):
    key = (int(m), int(topk))
    cached = state["buffers"].get(key)
    if cached is not None:
        return cached, state["blocks"][key], state["scratch"][key]
    shared_key = (
        state["device"],
        int(state["hidden"]),
        int(state["intermediate"]),
        int(state["num_experts"]),
        int(state["bits"]),
        int(m),
        int(topk),
    )
    entry = _SHARED.get(shared_key)
    if entry is None:
        block = _block_size(libs, m, topk, state["num_experts"])
        bufs = libs["make_buffers"](
            state["prepared"],
            m=m,
            topk=topk,
            dtype=torch.float16,
            device=state["device"],
            route_num_experts=state["num_experts"],
            full_rotation=True,
            block_size_m=block,
        )
        scratch = {
            "a": torch.empty(
                (m, state["hidden"]), dtype=torch.float16, device=state["device"]
            ),
            "tw": torch.empty((m, topk), dtype=torch.float32, device=state["device"]),
        }
        entry = (bufs, block, scratch)
        _SHARED[shared_key] = entry
        # Never evict during CUDA graph capture: a buffer recorded into a graph
        # must stay alive for every replay, so a capture grows the cache instead.
        if not torch.cuda.is_current_stream_capturing():
            while len(_SHARED) > _MAX_CACHED_SHAPES:
                _SHARED.popitem(last=False)
    else:
        _SHARED.move_to_end(shared_key)
    bufs, block, scratch = entry
    state["buffers"][key] = bufs
    state["blocks"][key] = block
    state["scratch"][key] = scratch
    return bufs, block, scratch


def _block_size(libs, m: int, topk: int, num_experts: int) -> int:
    """Route block for this batch shape (env-pinnable).

    ``VLLM_EXL3_B12X_BLOCK_M`` forces one specialization for every M, which is
    what a serving boot wants: the alternative is a JIT compile the first time
    a prefill chunk lands in the other block-size bucket.
    """
    raw = os.environ.get("VLLM_EXL3_B12X_BLOCK_M", "").strip()
    if raw:
        try:
            pinned = int(raw)
        except ValueError:
            pinned = 0
        if pinned > 0:
            return pinned
    return int(libs["select_block"](m, topk, num_experts))


def _run_dispatch(libs, state, a, tw, ti, bufs, block: int) -> None:
    """One ``run_w4a16_moe`` launch -- the single call site for apply/warmup."""
    libs["run"](
        a,
        state["prepared"],
        tw,
        ti,
        activation="silu",
        swiglu_limit=state.get("swiglu_limit"),
        swiglu_clamp_after_silu=False,
        intermediate_cache13=bufs.intermediate_cache13,
        intermediate_cache2=bufs.intermediate_cache2,
        output=bufs.output,
        fc1_c_tmp=bufs.fc1_c_tmp,
        fc2_c_tmp=bufs.fc2_c_tmp,
        packed_route_indices=bufs.packed_route_indices,
        block_expert_ids=bufs.block_expert_ids,
        packed_route_count=bufs.packed_route_count,
        expert_offsets=bufs.expert_offsets,
        expert_counts=bufs.expert_counts,
        route_block_size_m=block,
        full_rotation=True,
        suh_gate_table=state["prepared"].gate_suh,
        suh_up_table=state["prepared"].up_suh,
        svh_table=state["prepared"].down_svh,
        intermediate_rotation_scales=state["prepared"].intermediate_rotations,
        rotation_a_gate=bufs.rotation_a_gate,
        rotation_a_up=bufs.rotation_a_up,
    )


def _warm_blocks(state, libs, topk: int = 6) -> None:
    """Compile every route-block specialization this boot can reach.

    Runs one real dispatch per block size on throwaway data. A CUDA graph has
    not been captured yet at this point, so the temporaries are plain
    allocations; the workspaces themselves stay in ``_SHARED`` (they are the
    geometry, and losing them would just move the compile back into serving).
    Errors are swallowed: a failed warmup must not sink the boot.
    """
    if state is None:
        return
    key_base = (
        state["device"],
        int(state["hidden"]),
        int(state["intermediate"]),
        int(state["num_experts"]),
        int(state["bits"]),
    )
    if key_base in _WARMED_BLOCKS:
        return
    pinned = os.environ.get("VLLM_EXL3_B12X_BLOCK_M", "").strip()
    try:
        blocks = [int(pinned)] if pinned and int(pinned) > 0 else [8, 16]
    except ValueError:
        blocks = [8, 16]
    for block in blocks:
        m = _WARMUP_ROWS.get(block, max(block * 8, 64))
        try:
            bufs, block_used, scratch = _buffers(state, libs, m, topk)
            a = scratch["a"].zero_()
            tw = scratch["tw"].fill_(1.0 / topk)
            ti = torch.zeros(
                (m, topk), dtype=torch.int32, device=state["device"]
            )
            _run_dispatch(libs, state, a, tw, ti, bufs, block_used)
            torch.cuda.synchronize()
            _logger().info(
                "b12x MoE warmup compiled: block=%d M=%d topk=%d",
                block_used,
                m,
                topk,
            )
        except Exception as exc:  # never sink the boot over a warmup
            _logger().warning("b12x MoE warmup (block=%s) failed: %r", block, exc)
    _WARMED_BLOCKS.add(key_base)


def apply(
    layer: Any,
    x2d: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    expert_map: torch.Tensor | None,
    limit: float | None = None,
) -> torch.Tensor | None:
    """Run one layer through b12x; None means "use the established path"."""
    state = getattr(layer, "_exl3_b12x_state", None)
    if state is None:
        return None
    if expert_map is not None:
        layer._exl3_b12x_error = "expert_map (EP) is not qualified"
        return None
    if limit is not None and float(limit) != (state.get("swiglu_limit") or 0.0):
        # The clamp is baked into the launched activation; a per-call value that
        # disagrees with the one the layer was prepared with would silently
        # change the math, so fall back rather than serve a different function.
        layer._exl3_b12x_error = (
            f"per-call swiglu limit {limit!r} differs from prepared "
            f"{state.get('swiglu_limit')!r}"
        )
        return None
    tokens = int(x2d.shape[0])
    if tokens < min_rows():
        return None
    cap = max_rows()
    if cap and tokens > cap:
        return None
    topk = int(ids.shape[-1])
    if topk != 6:
        # DS4.1 routes 6 of 384; other arities are untested here.
        layer._exl3_b12x_error = f"topk={topk} is not qualified"
        return None
    try:
        libs = _libs()
        if libs is None:
            return None
        device = x2d.device
        bufs, block, scratch = _buffers(state, libs, tokens, topk)
        # Fixed scratch, never a per-call allocation: this runs inside CUDA
        # graph capture, where fresh allocations belong to the graph pool.
        a = scratch["a"]
        a.copy_(x2d)
        tw = scratch["tw"]
        tw.copy_(weights.reshape(tokens, topk))
        ti = ids.contiguous()
        if ti.device != device or tuple(ti.shape) != (tokens, topk):
            return None
        if not state.get("logged_active"):
            state["logged_active"] = True
            _logger().info(
                "b12x MoE active: first dispatch M=%d topk=%d block=%d",
                tokens,
                topk,
                block,
            )
        _run_dispatch(libs, state, a, tw, ti, bufs, block)
        # The kernel writes every output row (verified: the buffer starts as
        # torch.empty and still matches), so no zeroing is needed. Only the
        # fp32-activation case would hand back this reusable buffer itself.
        return bufs.output if x2d.dtype != torch.float32 else bufs.output.clone()
    except Exception as exc:  # never let the optional backend break a step
        layer._exl3_b12x_error = repr(exc)
        _warn = getattr(_logger(), "warning_once", _logger().warning)
        _warn(
            "b12x EXL3 MoE dispatch failed; falling back to the established "
            "path: %s",
            exc,
        )
        return None
