# image_processor.py
import cv2
import numpy as np
import logging
from typing import Tuple, Optional, List
from data_models import BoundingBox, DetectionResult, CardType
import pandas as pd

class GeometryUtils:
    """Utility class for geometric operations"""
    
    @staticmethod
    def calculate_iou(bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """Calculate Intersection over Union between two bounding boxes"""
        # Intersection coordinates
        x1 = max(bbox1.xmin, bbox2.xmin)
        y1 = max(bbox1.ymin, bbox2.ymin)
        x2 = min(bbox1.xmax, bbox2.xmax)
        y2 = min(bbox1.ymax, bbox2.ymax)
        
        # Calculate intersection area
        intersection_width = max(0, x2 - x1)
        intersection_height = max(0, y2 - y1)
        intersection_area = intersection_width * intersection_height
        
        # Calculate union area
        area1 = bbox1.area()
        area2 = bbox2.area()
        union_area = area1 + area2 - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0

class ImageProcessor:
    """Handles image processing operations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def resize_image(self, image: np.ndarray, target_size: Tuple[int, int] = (600, 400)) -> Tuple[np.ndarray, Tuple[float, float]]:
        """Resize image and return scaling factors"""
        try:
            if image is None or image.size == 0:
                raise ValueError("Invalid input image")
            
            original_height, original_width = image.shape[:2]
            target_width, target_height = target_size
            
            # Calculate scaling factors
            width_ratio = original_width / target_width
            height_ratio = original_height / target_height
            
            resized_image = cv2.resize(image, target_size)
            
            self.logger.debug(f"Image resized from {original_width}x{original_height} to {target_width}x{target_height}")
            
            return resized_image, (width_ratio, height_ratio)
        except Exception as e:
            self.logger.error(f"Image resize failed: {e}")
            raise
    
    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by specified angle"""
        try:
            if image is None:
                raise ValueError("Input image is None")
            
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            
            # Get rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Calculate new dimensions
            cos_val = abs(rotation_matrix[0, 0])
            sin_val = abs(rotation_matrix[0, 1])
            
            new_width = int((height * sin_val) + (width * cos_val))
            new_height = int((height * cos_val) + (width * sin_val))
            
            # Adjust translation
            rotation_matrix[0, 2] += (new_width / 2) - center[0]
            rotation_matrix[1, 2] += (new_height / 2) - center[1]
            
            # Apply rotation
            rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))
            
            self.logger.debug(f"Image rotated by {angle} degrees")
            return rotated_image
        except Exception as e:
            self.logger.error(f"Image rotation failed: {e}")
            raise
    
    def apply_mask(self, image: np.ndarray, detections: List[DetectionResult], 
                   card_type: CardType, exclusions: List[str] = None) -> np.ndarray:
        """Apply mask to image based on detections"""
        try:
            if exclusions is None:
                exclusions = []
            
            height, width = image.shape[:2]
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            mask = np.zeros((height, width), dtype=np.uint8)
            
            for detection in detections:
                if (detection.name == card_type.value or 
                    detection.name in exclusions):
                    continue
                
                bbox = detection.bbox
                mask[bbox.ymin:bbox.ymax, bbox.xmin:bbox.xmax] = 1
            
            masked_image = cv2.bitwise_and(image, image, mask=mask)
            
            self.logger.debug(f"Mask applied for card type: {card_type.value}")
            return masked_image
        except Exception as e:
            self.logger.error(f"Mask application failed: {e}")
            raise
    
    def crop_image(self, image: np.ndarray, bbox: BoundingBox) -> np.ndarray:
        """Crop image based on bounding box"""
        try:
            if image is None:
                raise ValueError("Input image is None")
            
            cropped = image[bbox.ymin:bbox.ymax, bbox.xmin:bbox.xmax]
            
            self.logger.debug(f"Image cropped to region: ({bbox.xmin}, {bbox.ymin}, {bbox.xmax}, {bbox.ymax})")
            return cropped
        except Exception as e:
            self.logger.error(f"Image cropping failed: {e}")
            raise
