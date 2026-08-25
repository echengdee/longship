from __future__ import annotations

import torch
from typing import Sequence


class ConcatBatchTensor:
    """Considering some buffers might contains a batch of unequal length tensors, they have to be
    accessed all at once. We concatenate them into a single tensor and store the original shapes.
    Thus, the original tensors can be recovered from the concatenated tensor.
    NOTE: by concatenating the tensors where will be a batch_idx and data_idx to identify the original tensor and data index.
    """

    def __init__(
        self,
        tensors: list[torch.Tensor] | None = None,
        batch_sizes: Sequence[int] | None = None,
        data_shape: tuple | None = None,
        dtype=torch.float32,
        device: torch.device = torch.device("cpu"),
    ):
        """Concatenate the tensors into a single tensor. Assuming the tensors' shapes are [batch, ...],
        where only the first dimension is different.
        """
        if tensors is None:
            assert (
                batch_sizes is not None and data_shape is not None
            ), "Either tensors or batch_lengths and data_shape should be provided."
            self._batch_sizes = torch.as_tensor(batch_sizes).to(torch.int64).to(device)
            self._data_shape = torch.Size(data_shape)
            self._concatenated_tensor = torch.empty([sum(batch_sizes)] + list(data_shape), dtype=dtype, device=device)
        else:
            self._batch_sizes = torch.as_tensor([t.shape[0] for t in tensors]).to(torch.int64).to(device)
            self._data_shape = tensors[0].shape[1:]
            for t in tensors:
                assert (
                    t.shape[1:] == self._data_shape
                ), "All tensors should have the same shape except the batch dimension."
            self._concatenated_tensor = torch.cat(tensors, dim=0)
        _batch_cuts = torch.cumsum(self._batch_sizes, dim=0)
        self._batch_starts = torch.zeros_like(_batch_cuts)
        self._batch_starts[1:] = _batch_cuts[:-1]
        self._batch_ends = _batch_cuts.to(torch.int64).to(device)

    def __getitem__(self, idx):
        """Get the data tensor at the given batch index."""
        if isinstance(idx, tuple):
            batch_idx, data_idx = idx
            return self._getitem_from_batch_data_idx(batch_idx, data_idx)
        elif isinstance(idx, int):
            return self._getitem_from_batch_idx(idx)
        else:
            raise NotImplementedError("Only support indexing with (batch_idx, data_idx) or batch_idx.")

    def _getitem_from_batch_data_idx(self, batch_idx, data_idx):
        assert len(batch_idx) == len(data_idx), "Batch index and data index should have the same length."
        assert (data_idx < self._batch_sizes[batch_idx]).all(), (
            f"Data index out of range at {torch.where(data_idx >= self._batch_sizes[batch_idx])},"
            + f" indexing {data_idx[torch.where(data_idx >= self._batch_sizes[batch_idx])]}"
            + f" of size {self._batch_sizes[batch_idx[torch.where(data_idx >= self._batch_sizes[batch_idx])]]}."
        )
        start = self._batch_starts[batch_idx]
        return self._concatenated_tensor[start + data_idx]  # shape (len(batch_idx), *data_shape)

    def _getitem_from_batch_idx(self, batch_idx):
        start = self._batch_starts[batch_idx]
        end = self._batch_ends[batch_idx]
        return self._concatenated_tensor[start:end]  # shape (batch_size, *data_shape)

    def __setitem__(self, idx, data):
        """Set the data tensor at the given batch index.
        TODO: some other types of indexing methods remain to be implemented.
        """
        if isinstance(idx, tuple):
            batch_idx, data_idx = idx
            self._setitem_by_batch_data_idx(batch_idx, data_idx, data)
        elif isinstance(idx, int):
            self._setitem_by_batch_idx(idx, data)

    def _setitem_by_batch_data_idx(self, batch_idx, data_idx, data):
        if isinstance(data_idx, slice):
            start = data_idx.start if data_idx.start is not None else 0
            stop = data_idx.stop if data_idx.stop is not None else len(data)
            step = data_idx.step if data_idx.step is not None else 1
            data_idx = torch.arange(start, stop, step).to(torch.int64).to(data.device)
        if isinstance(batch_idx, int):
            batch_idx = torch.tensor([batch_idx] * len(data_idx)).to(torch.int64).to(data.device)
        assert len(batch_idx) == len(data_idx), "Batch index and data index should have the same length."
        # batch_idx = batch_idx % len(self._batch_sizes)
        assert (data_idx < self._batch_sizes[batch_idx]).all(), "Data index out of range."  # type: ignore
        start = self._batch_starts[batch_idx]
        self._concatenated_tensor[start + data_idx] = data

    def _setitem_by_batch_idx(self, batch_idx, data):
        start = self._batch_starts[batch_idx]
        end = self._batch_ends[batch_idx]
        self._concatenated_tensor[start:end] = data

    def fill_data(self, data: torch.Tensor):
        """Fill all the data dimension with the given tensor.
        The data tensor should NOT have the batch dimension.
        """
        assert (
            data.shape == self._data_shape
        ), f"Data tensor shape does not match the expected shape, expect {self._data_shape} got {data.shape}."
        self._concatenated_tensor[:] = data.unsqueeze(0)

    def __len__(self):
        return len(self._batch_sizes)

    def contiguous(self):
        """Return a contiguous version of the concatenated tensor."""
        return self._concatenated_tensor.contiguous()

    def unwarp_flattened_idx(self, flattened_idx: torch.Tensor):
        """Unwarp the batch index on _concatenated_tensor to the original batch index and data index.
        Recommend the flattened_idx to have only 1 dimension.
        E.g.
            batch_starts =  [0, 3, 6, 9]
            batch_ends =    [3, 6, 9, 12]
            flattened_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            batch_idx =     [0, 0, 0, 1, 1, 1, 2, 2, 2, 3]
            data_idx =      [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
        """
        batch_idx = torch.searchsorted(self._batch_ends, flattened_idx, side="right")
        data_idx = flattened_idx - self._batch_starts[batch_idx]
        return batch_idx, data_idx

    @property
    def shape(self):
        return self._concatenated_tensor.shape
