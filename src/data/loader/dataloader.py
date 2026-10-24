import torch

from .dataclasses import DataSample
from .dataset import UPDEDataset, MultiUPDEDataset

class UPDEDataloader(torch.utils.data.DataLoader):
    def __init__(self, dataset: UPDEDataset | MultiUPDEDataset, *args, **kwargs):
        """A dataloader class for `UPDEDataset` and `MultiUPDEDataset` instances with custom collate function.

        Args:
            dataset (UPDEDataset | MultiUPDEDataset): Dataset instance to load data from
            *args: Additional positional arguments for `torch.utils.data.DataLoader`
            **kwargs: Additional keyword arguments for `torch.utils.data.DataLoader`
        """
        super().__init__(dataset, *args, **kwargs)
        self.collate_fn = self._custom_collate_fn

    def _custom_collate_fn(self, samples: list[DataSample]) -> DataSample:
        """Custom collate function to handle batching of samples of `UPDEDataset` instances.

        Args:
            samples (list[DataSample]): List of samples (DataSample instances) to be collated into a batch

        Returns:
            DataSample: Collated batch of samples
        """
        sample_dicts = [sample.to_dict() if hasattr(sample, "to_dict") else sample for sample in samples]
        batched = self._collate_dicts(sample_dicts)
        return DataSample.from_dict(batched)
    
    def _collate_dicts(self, sample_dicts: list[dict]) -> dict:
        batch = {}
        for key, value in sample_dicts[0].items():
            # Collect values for the current key across all samples
            values = [sample[key] for sample in sample_dicts]

            if isinstance(value, torch.Tensor):
                # Pad tensors to same size
                tensors_padded = self._pad_tensors_to_max_shape(values)

                # Stack tensors along a new batch dimension
                batch[key] = torch.stack(tensors_padded, dim=0)
            elif isinstance(value, dict):
                # Recursively collate nested dictionaries
                batch[key] = self._collate_dicts(values)
            elif isinstance(value, list):
                # Pad lists to same length
                lists_padded = self._pad_lists_to_max_length(values)
                batch[key] = lists_padded
            else:
                # For other data types, collate into a list directly
                batch[key] = values
        return batch
    
    def _pad_tensors_to_max_shape(self, tensors: list[torch.Tensor]) -> list[torch.Tensor]:
        """Pad tensors in the list to the maximum shape along each dimension.

        Args:
            tensors (list[torch.Tensor]): List of tensors to be padded
        Returns:
            list[torch.Tensor]: List of padded tensors
        """

        shapes = [t.shape for t in tensors]
        max_shape = tuple(max(s) for s in zip(*shapes))
        tensors_padded = []
        for t in tensors:
            # Circumventing issue with torch.nn.functional.pad and 0-size dimensions
            # TODO: Remove when fixed
            if any(dim == 0 for dim in t.shape):
                tensors_padded.append(torch.empty(max_shape).fill_(float("nan")))
                continue

            # Get padding needed for each dimension
            padding_needed = tuple(target - current for target, current in zip(max_shape, t.shape))
            # Bring padding_needed to the format expected by torch.nn.functional.pad
            padding_needed = sum(((0, p) for p in reversed(padding_needed)), ())
            # Apply padding
            # NOTE: Error of torch.nn.functional.pad when padding dimensions of size 0 with (0, 0) padding
            # Expected to be fixed in upcoming torch versions (see "https://github.com/pytorch/pytorch/issues/168071")
            tensors_padded.append(torch.nn.functional.pad(t, pad=padding_needed, mode="constant", value=torch.nan))

        return tensors_padded

    def _pad_lists_to_max_length(self, lists: list[list]) -> list[list]:
        """Pad lists in the input list to the maximum length.

        Args:
            lists (list[list]): List of lists to be padded
        Returns:
            list[list]: List of padded lists
        """
        max_length = max(len(lst) for lst in lists)
        lists_padded = [lst + [None] * (max_length - len(lst)) for lst in lists]
        return lists_padded
