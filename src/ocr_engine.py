# ocr_engine.py
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import List, Tuple
from paddleocr import PaddleOCR
from data_models import BoundingBox, OCRResult

class BaseOCREngine(ABC):
    """Abstract base class for OCR engines"""
    
    @abstractmethod
    def extract_text(self, image: np.ndarray) -> List[OCRResult]:
        """Extract text from image"""
        pass

class PaddleOCREngine(BaseOCREngine):
    """PaddleOCR implementation"""
    
    def __init__(self, use_angle_cls: bool = True, lang: str = 'en'):
        self.logger = logging.getLogger(__name__)
        try:
            self.ocr = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang)
            self.logger.info("PaddleOCR engine initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize PaddleOCR: {e}")
            raise
    
    def extract_text(self, image: np.ndarray) -> List[OCRResult]:
        """Extract text using PaddleOCR"""
        try:
            if image is None:
                self.logger.error("Input image is None")
                return []
            
            # Perform OCR
            results = self.ocr.ocr(image, cls=True)
            
            if not results or len(results) == 0:
                self.logger.warning("No OCR results returned")
                return []
            
            ocr_results = []
            for line in results[0]:
                # Extract bounding box points
                box_points = line[0]
                text = line[1][0]
                confidence = line[1][1]
                
                # Convert to numpy array and get min/max coordinates
                box = np.array(box_points).astype(np.int32)
                bbox = BoundingBox(
                    xmin=int(np.min(box[:, 0])),
                    ymin=int(np.min(box[:, 1])),
                    xmax=int(np.max(box[:, 0])),
                    ymax=int(np.max(box[:, 1])),
                    confidence=confidence
                )
                
                ocr_results.append(OCRResult(
                    text=text,
                    confidence=confidence,
                    bbox=bbox
                ))
            
            self.logger.info(f"OCR extraction completed: {len(ocr_results)} text regions found")
            return ocr_results
        except Exception as e:
            self.logger.error(f"OCR extraction failed: {e}")
            return []
