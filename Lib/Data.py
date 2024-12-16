import numpy.typing as npt
from enum import Enum
from typing import Tuple, List, Dict
from dataclasses import dataclass


class OCRStatus(Enum):
    SUCCESS = 0
    FAILED = 1


class TPSMode(Enum):
    GLOBAL = 0
    LOCAL = 1


class LineMerge(Enum):
    Merge = 0
    Stack = 1


@dataclass
class LineXMLData:
    id: str
    points: npt.NDArray
    label: str


@dataclass
class BBox:
    x: int
    y: int
    w: int
    h: int


@dataclass
class Line:
    contour: npt.NDArray
    bbox: BBox
    center: Tuple[int, int]
    angle: float


@dataclass
class LineData:
    image: npt.NDArray
    prediction: npt.NDArray
    angle: float
    lines: List[Line]


@dataclass
class LayoutData:
    image: npt.NDArray
    rotation: float
    images: List[BBox]
    text_bboxes: List[BBox]
    lines: List[Line]
    captions: List[BBox]
    margins: List[BBox]
    predictions: Dict[str, npt.NDArray]


@dataclass
class LineDetectionConfig:
    model_file: str
    patch_size: int


@dataclass
class LayoutDetectionConfig:
    model_file: str
    patch_size: int
    classes: List[str]


@dataclass
class OCRConfig:
    model_file: str
    input_width: int
    input_height: int
    input_layer: str
    output_layer: str
    squeeze_channel: bool
    swap_hw: bool
    charset: List[str]


@dataclass
class OCRREsult:
    text: List[str]
    lines: List[Line]
    line_images: List[npt.NDArray]
    angle: float
