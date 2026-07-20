import torch

from vide.models import _flashvsr_tcdecoder as tc

# Tiny decoder config: narrow channels and the 16 latent + 768 LQ-conditioning
# channels the tiny-long pipeline feeds in, so decode paths run for real on CPU.
CHANNELS = (8, 4, 4, 4)
LATENT_CHANNELS = 16 + 768


def _build():
    return tc.build_tcdecoder(
        new_channels=CHANNELS, device="cpu", dtype=torch.float32,
        new_latent_channels=LATENT_CHANNELS,
    )


def test_pixel_shuffle_pad_and_no_pad():
    shuffle = tc.PixelShuffle3d(4, 8, 8)

    no_pad = shuffle(torch.randn(1, 3, 8, 8, 8))
    assert no_pad.shape == (1, 2, 768, 1, 1)

    # 6 frames is not a multiple of ff=4, so the first frame is repeated to pad.
    padded = shuffle(torch.randn(1, 3, 6, 8, 8))
    assert padded.shape == (1, 2, 768, 1, 1)


def test_memblock_changes_channels_via_skip_conv():
    block = tc.MemBlock(2, 3)
    x = torch.randn(1, 2, 4, 4)

    out = block(x, torch.zeros_like(x))

    assert out.shape == (1, 3, 4, 4)


def test_identity_conv_is_initialised_to_identity():
    layer = tc.IdentityConv2d(4, kernel_size=3)
    x = torch.randn(1, 4, 8, 8)

    assert torch.allclose(layer(x), x, atol=1e-5)


def test_build_tcdecoder_defaults_latent_channels():
    model = tc.build_tcdecoder(new_channels=CHANNELS, device="cpu", dtype=torch.float32)

    assert model.latent_channels == 16
    assert model.mem == [None] * len(model.decoder)


def test_decode_video_with_cond_streams_and_trims():
    model = _build()
    x = torch.randn(1, 2, 16, 1, 1)
    cond = torch.randn(1, 3, 8, 8, 8)

    # First window: memory is empty, so the leading warm-up frames are trimmed.
    first = model.decode_video(x, cond=cond)
    assert first.shape[0] == 1 and first.shape[2] == 3

    # Second window: memory is primed, so nothing is trimmed and it decodes
    # more frames than the first.
    second = model.decode_video(x, cond=cond)
    assert second.shape[1] > first.shape[1]


def test_decode_video_without_cond():
    model = _build()
    x = torch.randn(1, 2, LATENT_CHANNELS, 1, 1)

    out = model.decode_video(x)

    assert out.shape[0] == 1 and out.shape[2] == 3


def test_clean_mem_resets_state():
    model = _build()
    model.decode_video(torch.randn(1, 2, LATENT_CHANNELS, 1, 1))
    assert any(m is not None for m in model.mem)

    model.clean_mem()

    assert model.mem == [None] * len(model.decoder)
