"""FlashVSR's TCDecoder: a tiny, streaming, decoder-only autoencoder.

The "tiny-long" FlashVSR pipeline decodes latents with a pruned Tiny
AutoEncoder (`TAEHV`) instead of the full Wan2.1 VAE. It decodes one short
clip window at a time and carries a small per-layer memory between windows
(`MemBlock`/`TPool`/`TGrow`), so an arbitrarily long video is decoded
incrementally in bounded VRAM rather than all at once.

Like `_build_lq_proj` in `flashvsr.py`, this lives in FlashVSR's example
scripts (`examples/WanVSR/utils/TCDecoder.py`, Apache-2.0) rather than in
the installable `diffsynth` package, so it is vendored here and attributed.
The architecture is reproduced faithfully so FlashVSR's `TCDecoder.ckpt`
loads into it (the pipeline loads the checkpoint with `strict=False`); the
upstream code paths the tiny-long pipeline never exercises -- the parallel
(non-streaming) decode, the in-`__init__` checkpoint loading and its
`TGrow` weight patching, and the diffusers wrapper -- are dropped.
"""

from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.init as init
from tqdm.auto import tqdm

TWorkItem = namedtuple("TWorkItem", ("input_tensor", "block_index"))


class IdentityConv2d(nn.Conv2d):
    """Same-shape, bias-free Conv2d initialised to identity (Dirac), used to
    deepen the decoder without changing its function at init."""

    def __init__(self, channels, kernel_size=3):
        super().__init__(channels, channels, kernel_size, padding=kernel_size // 2, bias=False)
        with torch.no_grad():
            init.dirac_(self.weight)


def _conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3


class MemBlock(nn.Module):
    """Residual conv block that also consumes the previous timestep's input
    (the streaming "memory") concatenated on the channel axis."""

    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(
            _conv(n_in * 2, n_out), nn.ReLU(inplace=True),
            _conv(n_out, n_out), nn.ReLU(inplace=True),
            _conv(n_out, n_out),
        )
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, past):
        return self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))


class TGrow(nn.Module):
    """Temporal upsampling: 1x1 conv to `stride` copies, then unfold back onto
    the time axis."""

    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f, n_f * stride, 1, bias=False)

    def forward(self, x):
        _NT, C, H, W = x.shape
        x = self.conv(x)
        return x.reshape(-1, C, H, W)


class PixelShuffle3d(nn.Module):
    """Space-to-depth over (F, H, W) folded onto the channel axis, returned as
    an NTCHW tensor. Left-pads the time axis to a multiple of `ff` by repeating
    the first frame. (`einops.rearrange` is avoided, as in `_build_lq_proj`, so
    this module needs no extra dependency.)"""

    def __init__(self, ff, hh, ww):
        super().__init__()
        self.ff, self.hh, self.ww = ff, hh, ww

    def forward(self, x):
        b, c, F_, H_, W_ = x.shape
        ff, hh, ww = self.ff, self.hh, self.ww
        if F_ % ff != 0:
            first = x[:, :, 0:1, :, :].repeat(1, 1, ff - F_ % ff, 1, 1)
            x = torch.cat([first, x], dim=2)
            F_ = x.shape[2]
        f, h, w = F_ // ff, H_ // hh, W_ // ww
        x = x.view(b, c, f, ff, h, hh, w, ww)
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(b, c * ff * hh * ww, f, h, w)
        return x.transpose(1, 2)


def apply_model_with_memblocks(model, x, show_progress_bar, mem):
    """Run a `nn.Sequential` of blocks over an NTCHW tensor one timestep at a
    time, threading the streaming memory (`mem`) through `MemBlock`/`TPool`/
    `TGrow` layers. Sequential (O(1)-memory) decode only, which is the mode
    FlashVSR's streaming pipeline uses. Returns `(NTCHW output, mem)`.
    """
    assert x.ndim == 5, f"TAEHV operates on NTCHW tensors, but got {x.ndim}-dim tensor"
    N, T, C, H, W = x.shape
    out = []
    work_queue = [TWorkItem(xt, 0) for xt in x.reshape(N, T * C, H, W).chunk(T, dim=1)]
    progress_bar = tqdm(range(T), disable=not show_progress_bar)
    while work_queue:
        xt, i = work_queue.pop(0)
        if i == 0:
            progress_bar.update(1)
        if i == len(model):
            out.append(xt)
            continue
        b = model[i]
        if isinstance(b, MemBlock):
            if mem[i] is None:
                xt_new = b(xt, xt * 0)
                mem[i] = xt
            else:
                xt_new = b(xt, mem[i])
                mem[i].copy_(xt)
            work_queue.insert(0, TWorkItem(xt_new, i + 1))
        elif isinstance(b, TGrow):
            xt = b(xt)
            _NT, C_, H_, W_ = xt.shape
            for xt_next in reversed(xt.view(N, b.stride * C_, H_, W_).chunk(b.stride, 1)):
                work_queue.insert(0, TWorkItem(xt_next, i + 1))
        else:
            xt = b(xt)
            work_queue.insert(0, TWorkItem(xt, i + 1))
    progress_bar.close()
    return torch.stack(out, 1), mem


class TAEHV(nn.Module):
    """Decoder-only Tiny AutoEncoder (pruned from TAEHV for Hunyuan Video).

    Deepening (`IdentityConv2d` + `ReLU` after every `ReLU`) is baked into the
    decoder skeleton. Call `decode_video`, not `forward`. Weights are loaded
    externally with `load_state_dict(..., strict=False)`.
    """

    image_channels = 3

    def __init__(
        self,
        decoder_time_upscale=(True, True),
        decoder_space_upscale=(True, True, True),
        channels=(256, 128, 64, 64),
        latent_channels=16,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        n_f = channels
        self.frames_to_trim = 2 ** sum(decoder_time_upscale) - 1

        base_decoder = nn.Sequential(
            Clamp(), _conv(self.latent_channels, n_f[0]), nn.ReLU(inplace=True),

            MemBlock(n_f[0], n_f[0]), MemBlock(n_f[0], n_f[0]), MemBlock(n_f[0], n_f[0]),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[0] else 1),
            TGrow(n_f[0], 1),
            _conv(n_f[0], n_f[1], bias=False),

            MemBlock(n_f[1], n_f[1]), MemBlock(n_f[1], n_f[1]), MemBlock(n_f[1], n_f[1]),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[1] else 1),
            TGrow(n_f[1], 2 if decoder_time_upscale[0] else 1),
            _conv(n_f[1], n_f[2], bias=False),

            MemBlock(n_f[2], n_f[2]), MemBlock(n_f[2], n_f[2]), MemBlock(n_f[2], n_f[2]),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[2] else 1),
            TGrow(n_f[2], 2 if decoder_time_upscale[1] else 1),
            _conv(n_f[2], n_f[3], bias=False),

            nn.ReLU(inplace=True), _conv(n_f[3], TAEHV.image_channels),
        )

        self.decoder = self._apply_identity_deepen(base_decoder, how_many_each=1, k=3)
        self.pixel_shuffle = PixelShuffle3d(4, 8, 8)
        self.mem = [None] * len(self.decoder)

    @staticmethod
    def _apply_identity_deepen(decoder, how_many_each=1, k=3):
        """Insert `how_many_each` (IdentityConv2d(k) + ReLU) pairs after every
        top-level `nn.ReLU`. In this decoder every such ReLU follows a Conv2d,
        so the channel count is read from that conv."""
        new_layers = []
        for b in decoder:
            new_layers.append(b)
            if isinstance(b, nn.ReLU) and isinstance(new_layers[-2], nn.Conv2d):
                channels = new_layers[-2].out_channels
                for _ in range(how_many_each):
                    new_layers.append(IdentityConv2d(channels, kernel_size=k))
                    new_layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*new_layers)

    def decode_video(self, x, parallel=False, show_progress_bar=False, cond=None):
        """Decode NTCHW latents to NTCHW RGB in ~[0, 1]. `cond` is the
        pixel-space LQ conditioning, space-to-depth folded and prepended.
        `parallel` is accepted for call-compatibility; only the sequential
        streaming decode FlashVSR uses is implemented."""
        trim_flag = self.mem[-8] is None
        if cond is not None:
            x = torch.cat([self.pixel_shuffle(cond), x], dim=2)
        x, self.mem = apply_model_with_memblocks(self.decoder, x, show_progress_bar, self.mem)
        if trim_flag:
            return x[:, self.frames_to_trim:]
        return x

    def clean_mem(self):
        self.mem = [None] * len(self.decoder)


def build_tcdecoder(
    new_channels=(512, 256, 128, 128),
    device="cuda",
    dtype=torch.bfloat16,
    new_latent_channels=None,
):
    """Build the (wider) FlashVSR TCDecoder. Mirrors upstream's
    `build_tcdecoder`; deepening is handled inside `TAEHV`. The caller loads
    `TCDecoder.ckpt` into it separately."""
    kwargs = {"channels": new_channels}
    if new_latent_channels is not None:
        kwargs["latent_channels"] = new_latent_channels
    model = TAEHV(**kwargs).to(device).to(dtype)
    model.clean_mem()
    return model
