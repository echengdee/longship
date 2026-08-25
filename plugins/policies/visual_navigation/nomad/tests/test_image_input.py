"""Tests for tensor-only image ingress."""

import pytest
import torch

from nomad_runtime import (
    ImageTensorSpec,
    ObservationBuffer,
    canonicalize_image,
)


def test_canonicalizes_hwc_bgr_uint8() -> None:
    image = torch.tensor(
        [[[10, 20, 30], [40, 50, 60]]],
        dtype=torch.uint8,
    )
    result = canonicalize_image(
        image,
        ImageTensorSpec(layout="hwc", channel_order="bgr"),
    )

    expected = torch.tensor(
        [
            [[30, 60]],
            [[20, 50]],
            [[10, 40]],
        ],
        dtype=torch.float32,
    ) / 255.0
    torch.testing.assert_close(result, expected)
    assert result.shape == (3, 1, 2)
    assert result.dtype == torch.float32
    assert result.is_contiguous()


def test_canonicalizes_explicit_float_byte_range() -> None:
    image = torch.full((3, 2, 2), 127.5)

    result = canonicalize_image(
        image,
        ImageTensorSpec(value_range="byte"),
    )

    torch.testing.assert_close(result, torch.full((3, 2, 2), 0.5))


@pytest.mark.parametrize(
    "image",
    [
        torch.full((3, 2, 2), 1.01),
        torch.full((3, 2, 2), float("nan")),
    ],
)
def test_rejects_invalid_float_image_values(image: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        canonicalize_image(image)


def test_buffer_builds_chronological_four_frame_context() -> None:
    buffer = ObservationBuffer()

    for index in range(5):
        image = torch.full((2, 3, 3), index, dtype=torch.uint8)
        buffer.append(
            image,
            timestamp_s=10.0 + index,
            spec=ImageTensorSpec(layout="hwc"),
        )

    context = buffer.snapshot(now_s=14.1, max_age_s=0.2)
    assert buffer.ready
    assert buffer.size == 4
    assert context.images.shape == (4, 3, 2, 3)
    assert context.timestamps_s == (11.0, 12.0, 13.0, 14.0)
    assert context.latest_timestamp_s == 14.0
    torch.testing.assert_close(
        context.images[:, 0, 0, 0],
        torch.tensor([1.0, 2.0, 3.0, 4.0]) / 255.0,
    )


def test_buffer_owns_frame_memory() -> None:
    buffer = ObservationBuffer(context_frames=1)
    image = torch.zeros((3, 2, 2))
    buffer.append(image, timestamp_s=1.0)

    image.fill_(1.0)

    assert torch.count_nonzero(buffer.snapshot().images) == 0


def test_buffer_rejects_incomplete_out_of_order_and_resolution_change() -> None:
    buffer = ObservationBuffer(context_frames=2)
    buffer.append(torch.zeros((3, 2, 2)), timestamp_s=1.0)

    with pytest.raises(RuntimeError, match="not ready"):
        buffer.snapshot()
    with pytest.raises(ValueError, match="strictly increasing"):
        buffer.append(torch.zeros((3, 2, 2)), timestamp_s=1.0)
    with pytest.raises(ValueError, match="resolution changed"):
        buffer.append(torch.zeros((3, 3, 2)), timestamp_s=2.0)


def test_buffer_rejects_stale_latest_frame() -> None:
    buffer = ObservationBuffer(context_frames=1)
    buffer.append(torch.zeros((3, 2, 2)), timestamp_s=1.0)

    with pytest.raises(RuntimeError, match="stale"):
        buffer.snapshot(now_s=1.2, max_age_s=0.1)


def test_buffer_selects_context_as_of_request_time() -> None:
    buffer = ObservationBuffer(context_frames=2, history_frames=4)
    for timestamp in (1.0, 2.0, 3.0, 4.0):
        buffer.append(torch.zeros(3, 3, 4), timestamp_s=timestamp)

    context = buffer.snapshot(now_s=3.5, max_age_s=1.0)

    assert context.timestamps_s == (2.0, 3.0)


def test_clear_allows_camera_resolution_change() -> None:
    buffer = ObservationBuffer(context_frames=1)
    buffer.append(torch.zeros((3, 2, 2)), timestamp_s=1.0)

    buffer.clear()
    buffer.append(torch.zeros((3, 3, 4)), timestamp_s=2.0)

    assert buffer.snapshot().images.shape == (1, 3, 3, 4)
