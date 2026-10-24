import dataclasses as _dataclasses
from typing import Optional, Any, Tuple

import torch
import torch.nn as nn

from src.models.dataclasses import UnstructuredData
from .transolver_pp import Model as TransolverPPModel


class TransolverPPWrapper(TransolverPPModel):
    """
    Adapter: UnstructuredData -> Transolver++ Model -> UnstructuredData

    - Accepts values/pos in the same layout used elsewhere in the repo:
      - non-channels-last: values (B, T, C, N), pos (B, T, D, N)
      - channels-last: values (B, N, T, C), pos (B, N, T, D)
    - Supports T==1. Returns values in the same layout as input (keeps channels_last flag).
    """

    def _forward(self, data: UnstructuredData) -> UnstructuredData:

        vals = data.values
        pos = data.pos
        context = data.global_params

        B, T, C, N = vals.shape
        if T != 1:
            raise NotImplementedError("TransolverPPWrapper supports T==1 only")

        x = vals[:, 0, :, :].permute(0, 2, 1).contiguous()  # (B, N, C)
        p = pos[:, 0, :, :].permute(0, 2, 1).contiguous()    # (B, N, D)

        # no extra conditioning provided here
        cond = context

        x = torch.cat([x, p], dim=-1)  # (B, N, C+D)

        # model expects a tuple (x, pos, condition)
        values_out = self.forward((x, p, cond)).permute((0, 2, 1)).unsqueeze(1)  # (B, 1, C, N)

        new_data = _dataclasses.replace(data, values=values_out)
        
        return new_data
