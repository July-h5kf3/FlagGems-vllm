# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import logging

import flag_gems
import torch

from flaggems_vllm.runtime import torch_device_fn

logger = logging.getLogger(__name__)


def bmm_w8a8_int8(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Channel-scaled INT8 BMM with INT32 accumulation on MTT S5000.

    A[B,M,K] is contiguous; W[B,K,N] is a transpose view of [B,N,K].
    FP32 scales have shapes [B,M] / [B,N], optionally [B,M,1] / [B,1,N].
    Inputs are prequantized; batch broadcasting and autograd are unsupported.

    Compute reuses the FlagGems MThreads implementation. This entry enforces
    the vLLM layout and scale contract.
    """
    logger.debug("GEMS_MTHREADS BMM_W8A8_INT8")
    if activation.ndim != 3 or weight.ndim != 3:
        raise ValueError("expected activation[B,M,K] and weight[B,K,N]")
    batch, rows, reduction = activation.shape
    columns = weight.shape[2]
    if weight.shape[:2] != (batch, reduction):
        raise ValueError("batch and reduction dimensions must match")
    if activation.dtype != torch.int8 or weight.dtype != torch.int8:
        raise TypeError("activation and weight must be signed INT8 tensors")
    if any(
        tensor.device != activation.device
        for tensor in (weight, activation_scale, weight_scale)
    ):
        raise ValueError("inputs and scales must be on the same MUSA device")
    if not activation.is_contiguous() or not weight.transpose(1, 2).is_contiguous():
        raise ValueError("expected contiguous activation and column-major weight")
    if activation_scale.shape not in ((batch, rows), (batch, rows, 1)):
        raise ValueError("activation_scale must have shape [B,M] or [B,M,1]")
    if weight_scale.shape not in ((batch, columns), (batch, 1, columns)):
        raise ValueError("weight_scale must have shape [B,N] or [B,1,N]")
    if any(
        scale.dtype != torch.float32 or not scale.is_contiguous()
        for scale in (activation_scale, weight_scale)
    ):
        raise TypeError("scales must be contiguous FP32 tensors")
    if any(scale.requires_grad for scale in (activation_scale, weight_scale)):
        raise NotImplementedError("bmm_w8a8_int8 is forward-only")
    if out_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError("output dtype must be bfloat16, float16 or float32")
    if out is not None and (not out.is_contiguous() or out.requires_grad):
        raise ValueError("out must be contiguous and must not require gradients")
    if out is not None and out.numel() == 0:
        if (
            out.shape != (batch, rows, columns)
            or out.dtype != out_dtype
            or out.device != activation.device
        ):
            raise ValueError("out must have matching shape, dtype and device")
        return out
    if out is None:
        out = torch.empty(
            (batch, rows, columns), dtype=out_dtype, device=activation.device
        )
    elif (
        out.shape != (batch, rows, columns)
        or out.dtype != out_dtype
        or out.device != activation.device
    ):
        raise ValueError("out must have matching shape, dtype and device")
    elif out.numel() and any(
        out.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()
        for tensor in (activation, weight, activation_scale, weight_scale)
    ):
        raise ValueError("out must not share storage with an input")
    if out.numel() == 0:
        return out
    with torch_device_fn.device(activation.device):
        return flag_gems.bmm_w8a8(
            activation,
            weight,
            activation_scale,
            weight_scale,
            out_dtype=out_dtype,
            out=out,
        )
