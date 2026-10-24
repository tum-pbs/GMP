from .CrashDataset import CrashDataset
from .CrashSteady import CrashSteady
from .DrivAerMLSVDataset import DrivAerMLSVDataset
from .EllipseDataset import EllipseDataset
from .H5Dataset import H5Dataset
from .KSDataset import KSDataset
from .ShiftWingSVDataset import ShiftWingSVDataset

# Aliases for backwards compatibility
DrivAerMLDataset = DrivAerMLSVDataset
ShiftWingDataset = ShiftWingSVDataset

__all__ = [
    "CrashDataset",
    "CrashSteady",
    "DrivAerMLSVDataset",
    "DrivAerMLDataset",
    "EllipseDataset",
    "H5Dataset",
    "KSDataset",
    "ShiftWingSVDataset",
    "ShiftWingDataset",
]