# UPDE Dataloader & Dataset Reference

Reference of tensor shapes and metadata for batches and samples.

All tensors are PyTorch tensors with dtype `torch.float32` unless otherwise specified.

## Dataset — Sample Structure

### Top-Level Keys

| Key | Type | Shape | Notes |
| --- | --- | --- | --- |
| `positions` | tensor | (T, D, N) or (N, T, D) | Time-major or point-major layout.
| `values` | tensor | (T, C, N) or (N, T, C) | Per-point values.
| `attributes` | tensor | (T, A, N) or (N, T, A) | Additional per-point attributes.
| `num_points` | tensor (long) | () | Number of valid points in the sample (not ordered in N).
| `physical_metadata` | dict | — | See table below.
| `loading_metadata` | dict | — | See table below.

### `physical_metadata`

| Key | Type | Shape |
| --- | --- | --- |
| `min_domain` | tensor | (D) |
| `max_domain` | tensor | (D) |
| `hyperparams` | tensor | (P) |
| `num_hparams` | tensor (long) | () | 
| `num_attrs` | tensor (long) | () | 
| `dt` | tensor | () |

### `loading_metadata`

| Key | Type | Shape |
| --- | --- | --- |
| `name` | str | — |
| `values_name` | list(str) | (C) |
| `attrs_name` | list(str) | (A) |
| `hparams_name` | list(str) | (P) |
| `grid_dims` | tuple(int, ...) | () | Original point grid dimensions, for reshaping (if needed).
| `norm_std_pos` | tensor | (T, D) |
| `norm_mean_pos` | tensor | (T, D) |
| `norm_std_val` | tensor | (T, C) |
| `norm_mean_val` | tensor | (T, C) |

## Dataloader — Batch Structure

### Top-Level Keys

Note: Tensors are padded for batching.

| Key | Type | Shape | Notes |
| --- | --- | --- | --- |
| `positions` | tensor | (B, T, D, N) or (B, N, T, D) | Time-major or point-major layout.
| `values` | tensor | (B, T, C, N) or (B, N, T, C) | Per-point values.
| `attributes` | tensor | (B, T, A, N) or (B, N, T, A) | Additional per-point attributes.
| `num_points` | tensor | (B) | Valid Points per batch item.
| `physical_metadata` | dict | — | See table below.
| `loading_metadata` | dict | — | See table below.

### `physical_metadata`

| Key | Type | Shape |
| --- | --- | --- |
| `min_domain` | tensor | (B, D) |
| `max_domain` | tensor | (B, D) |
| `hyperparams` | tensor | (B, P) |
| `num_hparams` | tensor (long) | (B) | number of valid hyperparameters per sample
| `num_attrs` | tensor (long) | (B) | number of valid attributes per sample
| `dt` | tensor | (B) |

### `loading_metadata`

| Key | Type | Shape |
| --- | --- | --- |
| `name` | list(str) | (B) |
| `values_name` | list(list(str)) | (B, C) |
| `attrs_name` | list(list(str)) | (B, A) |
| `hparams_name` | list(list(str)) | (B, P) |
| `grid_dims` | list(tuple(int, ...)) | (B) | Original point grid dimensions, for reshaping (if needed).
| `norm_std_pos` | tensor | (B, T, D) |
| `norm_mean_pos` | tensor | (B, T, D) |
| `norm_std_val` | tensor | (B, T, C) |
| `norm_mean_val` | tensor | (B, T, C) |

## Dimension Legend

- B: batch size
- T: frames
- N: number of points per sample
- D: spatial dimensions
- C: value channels/fields
- A: attribute channels
- P: number of hyperparameters

---

## Non-Used Keys

If a key is not used in a specific dataset, the fileloader may set it's entries to `None` or some other placeholder.
For example, if no attributes are used, the `attributes` tensor may be set to `None`, and the corresponding entries in `loading_metadata` may be empty lists.

IMPORTANT: The presence of such an entry may interfere with the collation in the dataloader if other samples do have that key populated properly.
It is recommended to either all samples have the keys' values set to `None` or all samples have valid, collateable data (e.g. tensors with 0-size dimensions, empty list, ...) for those keys.

PRELIMINARY FIX: An issue in PyTorch's `torch.nn.functional.pad` function when padding dimensions of size 0 with (0, 0) padding is present in the current code.
This error interferes with the collation in the dataloader if such zero-sized dimensions are present.
As a temporary workaround when such a 0-dim tensor is encountered during collation, a new tensor (filled with padding value `nan`) is created instead of the old one padded. 
Once the issue is fixed in upcoming PyTorch versions (expected in 2.10) this workaround can be removed.



