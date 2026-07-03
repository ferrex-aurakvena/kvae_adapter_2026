from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F


def nonlinearity(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)


class SafeConv3d(nn.Conv3d):
    def forward(self, x: torch.Tensor, write_to: torch.Tensor | None = None) -> torch.Tensor:
        memory_gb = x.numel() * x.element_size() / 1e9
        if memory_gb <= 3:
            out = super().forward(x)
            if write_to is not None:
                write_to[...] = out
                return write_to
            return out

        kernel_t = self.kernel_size[0]
        chunks = torch.chunk(x, math.ceil(memory_gb / 2), dim=2)
        outputs: list[torch.Tensor] = []
        time_offset = 0
        for i, chunk in enumerate(chunks):
            z = chunk if i == 0 or kernel_t == 1 else torch.cat([z[:, :, -kernel_t + 1 :], chunk], dim=2)
            out = super().forward(z)
            if i != 0 and kernel_t != 1:
                out = out[:, :, -chunk.shape[2] :]
            if write_to is not None:
                write_to[:, :, time_offset : time_offset + out.shape[2]] = out
                time_offset += out.shape[2]
            else:
                outputs.append(out)
        return write_to if write_to is not None else torch.cat(outputs, dim=2)


class CachedCausalConv3d(nn.Module):
    def __init__(
        self,
        chan_in: int,
        chan_out: int,
        kernel_size: int | tuple[int, int, int],
        stride: tuple[int, int, int] = (1, 1, 1),
        dilation: tuple[int, int, int] = (1, 1, 1),
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        kt, kh, kw = kernel_size
        self.height_pad = kh // 2
        self.width_pad = kw // 2
        self.time_pad = kt - 1
        self.time_kernel_size = kt
        self.stride = stride
        self.conv = SafeConv3d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)

    def forward(self, x: torch.Tensor, cache: dict[str, torch.Tensor | None]) -> torch.Tensor:
        t_stride = self.stride[0]
        x_parallel = F.pad(x, (self.width_pad, self.width_pad, self.height_pad, self.height_pad, 0, 0), mode="replicate")
        if cache["padding"] is None:
            first = x_parallel[:, :, :1]
            padding = first.expand(first.shape[0], first.shape[1], self.time_pad, first.shape[3], first.shape[4])
        else:
            padding = cache["padding"]

        out_size = list(x.shape)
        out_size[1] = self.conv.out_channels
        if t_stride == 2:
            out_size[2] = (x.shape[2] + 1) // 2
        out = torch.empty(tuple(out_size), dtype=x.dtype, device=x.device)

        offset_out = math.ceil(padding.shape[2] / t_stride)
        offset_in = offset_out * t_stride - padding.shape[2]
        if offset_out > 0:
            poisoned = torch.cat([padding, x_parallel[:, :, : offset_in + self.time_kernel_size - t_stride]], dim=2)
            out[:, :, :offset_out] = self.conv(poisoned)
        if offset_out < out.shape[2]:
            out[:, :, offset_out:] = self.conv(x_parallel[:, :, offset_in:])

        pad_offset = (
            offset_in
            + t_stride * math.trunc((x_parallel.shape[2] - offset_in - self.time_kernel_size) / t_stride)
            + t_stride
        )
        cache["padding"] = torch.clone(x_parallel[:, :, pad_offset:])
        return out


class RMSNorm(nn.Module):
    def __init__(self, in_channels: int, **_: Any) -> None:
        super().__init__()
        self.scale = float(in_channels) ** 0.5
        self.gamma = nn.Parameter(torch.ones(in_channels, 1, 1, 1))

    def forward(self, x: torch.Tensor, cache: dict[str, Any] | None = None) -> torch.Tensor:
        dtype = x.dtype
        y = F.normalize(x.float(), dim=1).to(dtype=dtype)
        y = y * (self.gamma.to(dtype=dtype) * self.scale)
        if cache is not None and cache.get("mean") is None and cache.get("var") is None:
            cache["mean"] = 1
            cache["var"] = 1
        return y


class CachedSpatialNorm3D(nn.Module):
    def __init__(self, f_channels: int, zq_channels: int, add_conv: bool = False, **_: Any) -> None:
        super().__init__()
        self.norm_layer = RMSNorm(f_channels)
        self.add_conv = add_conv
        if add_conv:
            self.conv = CachedCausalConv3d(zq_channels, zq_channels, kernel_size=3)
        self.conv_y = SafeConv3d(zq_channels, f_channels, kernel_size=1)
        self.conv_b = SafeConv3d(zq_channels, f_channels, kernel_size=1)

    def forward(self, f: torch.Tensor, zq: torch.Tensor, cache: dict[str, Any]) -> torch.Tensor:
        if cache["norm"].get("mean") is None and cache["norm"].get("var") is None:
            f_first, f_rest = f[:, :, :1], f[:, :, 1:]
            zq_first, zq_rest = zq[:, :, :1], zq[:, :, 1:]
            zq_first = F.interpolate(zq_first, size=f_first.shape[-3:], mode="nearest")
            if zq.shape[2] > 1 and f_rest.shape[2] > 0:
                zq_rest = F.interpolate(zq_rest, size=f_rest.shape[-3:], mode="nearest")
                zq = torch.cat([zq_first, zq_rest], dim=2)
            else:
                zq = zq_first
        else:
            zq = F.interpolate(zq, size=f.shape[-3:], mode="nearest")
        if self.add_conv:
            zq = self.conv(zq, cache["add_conv"])
        norm_f = self.norm_layer(f, cache["norm"])
        return norm_f * self.conv_y(zq) + self.conv_b(zq)


class CachedResnetBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        dropout: float = 0.0,
        temb_channels: int = 0,
        zq_ch: int | None = None,
        add_conv: bool = False,
        conv_shortcut: bool = False,
    ) -> None:
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        norm = RMSNorm if zq_ch is None else lambda c, **kw: CachedSpatialNorm3D(c, zq_ch, add_conv=add_conv)
        self.norm1 = norm(in_channels)
        self.conv1 = CachedCausalConv3d(in_channels, out_channels, kernel_size=3)
        self.norm2 = norm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = CachedCausalConv3d(out_channels, out_channels, kernel_size=3)
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)
        if in_channels != out_channels:
            if conv_shortcut:
                self.conv_shortcut = CachedCausalConv3d(in_channels, out_channels, kernel_size=3)
            else:
                self.nin_shortcut = SafeConv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(
        self,
        x: torch.Tensor,
        temb: torch.Tensor | None,
        layer_cache: dict[str, Any],
        zq: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = x
        h = self.norm1(h, cache=layer_cache["norm1"]) if zq is None else self.norm1(h, zq, layer_cache["norm1"])
        h = F.silu(h, inplace=False)
        h = self.conv1(h, cache=layer_cache["conv1"])
        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None, None]
        h = self.norm2(h, cache=layer_cache["norm2"]) if zq is None else self.norm2(h, zq, layer_cache["norm2"])
        h = F.silu(h, inplace=False)
        h = self.conv2(self.dropout(h), cache=layer_cache["conv2"])
        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x, cache=layer_cache["conv_shortcut"]) if self.use_conv_shortcut else self.nin_shortcut(x)
        return x + h


class CachedPXSDownsampleV2(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, compress_time: bool, factor: int = 2) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.temporal_compress = compress_time
        self.factor = factor
        self.unshuffle = nn.PixelUnshuffle(factor)
        self.spatial_conv = SafeConv3d(
            in_channels,
            out_channels,
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
            padding_mode="reflect",
        )
        if compress_time:
            self.temporal_conv = nn.ModuleList(
                [
                    CachedCausalConv3d(out_channels, out_channels, kernel_size=(2, 1, 1), stride=(2, 1, 1)),
                    CachedCausalConv3d(out_channels, out_channels, kernel_size=(2, 1, 1), stride=(2, 1, 1)),
                ]
            )
        self.linear = nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1)

    def spatial_downsample(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        pxs = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        pxs = self.unshuffle(pxs)
        _, c4, hh, ww = pxs.shape
        if c4 % self.out_channels != 0:
            raise RuntimeError(f"Cannot reduce pixel-unshuffle channels {c4} to {self.out_channels}")
        pxs = pxs.view(b * t, self.out_channels, c4 // self.out_channels, hh, ww).mean(dim=2)
        pxs = pxs.view(b, t, self.out_channels, hh, ww).permute(0, 2, 1, 3, 4)
        return self.spatial_conv(x) + pxs

    def temporal_downsample(self, x: torch.Tensor, cache: Sequence[dict[str, Any]]) -> torch.Tensor:
        b, c, t, h, w = x.shape
        permuted = x.permute(0, 3, 4, 1, 2).reshape(b * h * w, c, t)
        if cache[0]["padding"] is None:
            first, rest = permuted[..., :1], permuted[..., 1:]
            interp = torch.cat([first, F.avg_pool1d(rest, kernel_size=2, stride=2)], dim=-1) if rest.shape[-1] > 0 else first
        else:
            interp = F.avg_pool1d(permuted, kernel_size=2, stride=2) if permuted.shape[-1] > 0 else permuted
        interp = interp.view(b, h, w, c, interp.shape[-1]).permute(0, 3, 4, 1, 2)
        conv = self.temporal_conv[0](x, cache[0]) + self.temporal_conv[1](x, cache[1])
        return conv + interp

    def forward(self, x: torch.Tensor, cache: Sequence[dict[str, Any]]) -> torch.Tensor:
        x = self.spatial_downsample(x)
        if self.temporal_compress:
            x = self.temporal_downsample(x, cache)
        return self.linear(x)


class CachedPXSUpsample(nn.Module):
    def __init__(self, in_channels: int, compress_time: bool, factor: int = 2) -> None:
        super().__init__()
        self.temporal_compress = compress_time
        self.factor = factor
        self.spatial_conv = SafeConv3d(
            in_channels,
            in_channels,
            kernel_size=(1, 3, 3),
            stride=(1, 1, 1),
            padding=(0, 1, 1),
            padding_mode="reflect",
        )
        if compress_time:
            self.temporal_conv = CachedCausalConv3d(
                in_channels, in_channels, kernel_size=(3, 1, 1), stride=(1, 1, 1), dilation=(1, 1, 1)
            )
        self.linear = SafeConv3d(in_channels, in_channels, kernel_size=1, stride=1)

    def temporal_upsample(self, x: torch.Tensor, cache: dict[str, Any]) -> torch.Tensor:
        factor = 1 + int(x.shape[2] > 1)
        repeated = x.repeat_interleave(factor, dim=2)
        tail = repeated[:, :, factor - 1 :] if cache["padding"] is None else repeated
        return self.temporal_conv(tail, cache) + tail

    def spatial_upsample(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        y = x.permute(0, 2, 1, 3, 4).reshape(b, t * c, h, w)
        y = F.interpolate(y, scale_factor=2, mode="nearest")
        y = y.view(b, t, c, 2 * h, 2 * w).permute(0, 2, 1, 3, 4)
        return y + self.spatial_conv(y)

    def forward(self, x: torch.Tensor, cache: dict[str, Any]) -> torch.Tensor:
        if self.temporal_compress:
            x = self.temporal_upsample(x, cache)
        y = self.spatial_upsample(x)
        return self.linear(y, write_to=torch.empty_like(y))


class CachedEncoder3DV2(nn.Module):
    def __init__(
        self,
        ch: int = 128,
        ch_mult: Sequence[int] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        in_channels: int = 3,
        z_channels: int = 16,
        temporal_compress_times: int = 4,
    ) -> None:
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.temporal_compress_level = int(math.log2(temporal_compress_times))
        self.conv_in = CachedCausalConv3d(in_channels, ch, kernel_size=3)
        self.down = nn.ModuleList()
        block_in = ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks):
                block.append(CachedResnetBlock3D(block_in, block_out, temb_channels=0))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = nn.ModuleList()
            if i_level != self.num_resolutions - 1:
                next_out = ch * ch_mult[i_level + 1]
                down.downsample = CachedPXSDownsampleV2(
                    block_in, next_out, compress_time=i_level < self.temporal_compress_level
                )
                block_in = next_out
            self.down.append(down)
        self.mid = nn.Module()
        self.mid.block_1 = CachedResnetBlock3D(block_in, block_in, temb_channels=0)
        self.mid.block_2 = CachedResnetBlock3D(block_in, block_in, temb_channels=0)
        self.norm_out = RMSNorm(block_in)
        self.conv_out = CachedCausalConv3d(block_in, 2 * z_channels, kernel_size=3)

    def forward(self, x: torch.Tensor, cache: dict[str, Any]) -> torch.Tensor:
        temb = None
        h = self.conv_in(x, cache["conv_in"])
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h, temb, cache[i_level][i_block])
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h, cache[i_level]["down"])
        h = self.mid.block_1(h, temb, cache["mid_1"])
        h = self.mid.block_2(h, temb, cache["mid_2"])
        h = self.norm_out(h, cache["norm_out"])
        return self.conv_out(nonlinearity(h), cache["conv_out"])


class CachedDecoder3D(nn.Module):
    def __init__(
        self,
        ch: int = 128,
        ch_mult: Sequence[int] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        out_ch: int = 3,
        z_channels: int = 16,
        temporal_compress_times: int = 4,
        add_conv: bool = False,
    ) -> None:
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.temporal_compress_level = int(math.log2(temporal_compress_times))
        block_in = ch * ch_mult[-1]
        self.conv_in = CachedCausalConv3d(z_channels, block_in, kernel_size=3)
        self.mid = nn.Module()
        self.mid.block_1 = CachedResnetBlock3D(block_in, block_in, zq_ch=z_channels, add_conv=add_conv)
        self.mid.block_2 = CachedResnetBlock3D(block_in, block_in, zq_ch=z_channels, add_conv=add_conv)
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks + 1):
                block.append(CachedResnetBlock3D(block_in, block_out, zq_ch=z_channels, add_conv=add_conv))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = nn.ModuleList()
            if i_level != 0:
                up.upsample = CachedPXSUpsample(
                    block_in, compress_time=i_level >= self.num_resolutions - self.temporal_compress_level
                )
            self.up.insert(0, up)
        self.norm_out = CachedSpatialNorm3D(block_in, z_channels, add_conv=add_conv)
        self.conv_out = CachedCausalConv3d(block_in, out_ch, kernel_size=3)

    def forward(self, z: torch.Tensor, cache: dict[str, Any]) -> torch.Tensor:
        temb = None
        zq = z
        h = self.conv_in(z, cache["conv_in"])
        h = self.mid.block_1(h, temb, cache["mid_1"], zq=zq)
        h = self.mid.block_2(h, temb, cache["mid_2"], zq=zq)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb, cache[i_level][i_block], zq=zq)
            if i_level != 0:
                h = self.up[i_level].upsample(h, cache[i_level]["up"])
        h = self.norm_out(h, zq, cache["norm_out"])
        return self.conv_out(nonlinearity(h), cache["conv_out"])


class KVAE2T4S8(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conf = {
            "enc": {"num_res_blocks": 2, "temporal_compress_times": 4},
            "dec": {"num_res_blocks": 2, "temporal_compress_times": 4},
        }
        self.encoder = CachedEncoder3DV2()
        self.decoder = CachedDecoder3D()

    def _make_empty_cache(self, block: str) -> dict[str, Any]:
        num_res_blocks = self.conf[block]["num_res_blocks"]

        def conv() -> dict[str, Any]:
            return {"padding": None}

        def norm_enc() -> dict[str, Any]:
            return {"mean": None, "var": None}

        def norm_dec() -> dict[str, Any]:
            return {"norm": norm_enc(), "add_conv": conv()}

        def resblock(kind: str) -> dict[str, Any]:
            return {
                "norm1": norm_enc() if kind == "enc" else norm_dec(),
                "norm2": norm_enc() if kind == "enc" else norm_dec(),
                "conv1": conv(),
                "conv2": conv(),
                "conv_shortcut": conv(),
            }

        cache: dict[str, Any] = {
            "conv_in": conv(),
            "mid_1": resblock(block),
            "mid_2": resblock(block),
            "norm_out": norm_enc() if block == "enc" else norm_dec(),
            "conv_out": conv(),
        }
        for i in range(4):
            p = num_res_blocks if block == "enc" else num_res_blocks + 1
            cache[i] = {"down": [conv(), conv()], "up": conv()}
            for j in range(p):
                cache[i][j] = resblock(block)
        return cache

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x, self._make_empty_cache("enc"))
        mean, _ = torch.chunk(h, 2, dim=1)
        return mean

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z, self._make_empty_cache("dec"))


def state_dict_report(model: nn.Module, state_dict: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    model_state = model.state_dict()
    model_keys = set(model_state)
    ckpt_keys = set(state_dict)
    shape_mismatches = []
    for key in sorted(model_keys & ckpt_keys):
        if tuple(model_state[key].shape) != tuple(state_dict[key].shape):
            shape_mismatches.append((key, tuple(model_state[key].shape), tuple(state_dict[key].shape)))
    return {
        "model_keys": len(model_keys),
        "checkpoint_keys": len(ckpt_keys),
        "missing": sorted(model_keys - ckpt_keys),
        "unexpected": sorted(ckpt_keys - model_keys),
        "shape_mismatches": shape_mismatches,
    }


def load_kvae2_t4s8(
    weights_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
    strict: bool = True,
    freeze: bool = True,
) -> KVAE2T4S8:
    device = torch.device(device)
    dtype = dtype or (torch.bfloat16 if device.type == "cuda" else torch.float32)
    model = KVAE2T4S8()
    state = load_file(str(weights_path), device="cpu")
    report = state_dict_report(model, state)
    if strict and (report["missing"] or report["unexpected"] or report["shape_mismatches"]):
        lines = [
            "KVAE-2 t4s8 state dict mismatch:",
            f"missing={len(report['missing'])}",
            f"unexpected={len(report['unexpected'])}",
            f"shape_mismatches={len(report['shape_mismatches'])}",
        ]
        for key in report["missing"][:10]:
            lines.append(f"  missing: {key}")
        for key in report["unexpected"][:10]:
            lines.append(f"  unexpected: {key}")
        for key, want, got in report["shape_mismatches"][:10]:
            lines.append(f"  shape: {key} model={want} ckpt={got}")
        raise RuntimeError("\n".join(lines))
    model.load_state_dict(state, strict=strict)
    model.to(device=device, dtype=dtype)
    model.eval()
    if freeze:
        model.requires_grad_(False)
    return model
