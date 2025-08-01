# models.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum

class CardType(Enum):
    FRONT_CARD = "front_card"
    REAR_CARD = "rear_card"
    FRONT_LONG = "front_long"

@dataclass
class BoundingBox:
    """Represents a bounding box with coordinates and confidence"""
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    confidence: float = 0.0
    
    def center(self) -> Tuple[float, float]:
        """Calculate center point of the bounding box"""
        return (self.xmin + (self.xmax - self.xmin) / 2, 
                self.ymin + (self.ymax - self.ymin) / 2)
    
    def area(self) -> int:
        """Calculate area of the bounding box"""
        return (self.xmax - self.xmin) * (self.ymax - self.ymin)
    
    def width(self) -> int:
        return self.xmax - self.xmin
    
    def height(self) -> int:
        return self.ymax - self.ymin
    
    def is_horizontal(self) -> bool:
        return self.width() > self.height()

@dataclass
class DetectionResult:
    """Represents a detected object with its properties"""
    name: str
    bbox: BoundingBox
    confidence: float

@dataclass
class OCRResult:
    """Represents OCR extracted text with metadata"""
    text: str
    confidence: float
    bbox: BoundingBox

@dataclass
class ProcessingMetrics:
    """Stores processing performance metrics"""
    processing_time: float = 0.0
    yolo_time: float = 0.0
    ocr_time: float = 0.0
    angle: float = 0.0

@dataclass
class AadhaarData:
    """Structured representation of Aadhaar card data"""
    id_number: str = "Nan"
    name_on_card: str = "Nan"
    date_of_birth: str = "Nan"
    gender: str = "Nan"
    address: str = "Nan"
    pin_code: str = "Nan"
    state: str = "Nan"
    parent_name: str = "Nan"
    
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    yolo_scores: Dict[str, float] = field(default_factory=dict)
    bounding_boxes: Dict[str, str] = field(default_factory=dict)
    metrics: ProcessingMetrics = field(default_factory=ProcessingMetrics)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            'ocr_text': {
                'ID_NUMBER': self.id_number,
                'NAME_ON_CARD': self.name_on_card,
                'DATE_OF_BIRTH': self.date_of_birth,
                'GENDER': self.gender,
                'ADDRESS': self.address,
                'PIN_CODE': self.pin_code,
                'STATE': self.state,
                'PARENT_NAME': self.parent_name
            },
            'confidence': self.confidence_scores,
            'yolo_scores': self.yolo_scores,
            'processing_time': self.metrics.processing_time,
            'yolo_time': self.metrics.yolo_time,
            'ocr_time': self.metrics.ocr_time
        }
