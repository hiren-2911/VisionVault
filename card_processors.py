# card_processors.py
import numpy as np
import pandas as pd
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
from data_models import (
    DetectionResult, BoundingBox, OCRResult, AadhaarData, 
    ProcessingMetrics, CardType
)
from image_processor import ImageProcessor, GeometryUtils
from ocr_engine import BaseOCREngine
from text_parsers import TextParserFactory

class BaseCardProcessor(ABC):
    """Abstract base class for card processors"""
    
    def __init__(self, ocr_engine: BaseOCREngine, image_processor: ImageProcessor,
                 pin_data: Dict[str, str]):
        self.ocr_engine = ocr_engine
        self.image_processor = image_processor
        self.pin_data = pin_data
        self.logger = logging.getLogger(__name__)
        
        # Initialize parsers
        self.date_parser = TextParserFactory.create_date_parser()
        self.gender_parser = TextParserFactory.create_gender_parser()
        self.address_parser = TextParserFactory.create_address_parser(pin_data)
    
    @abstractmethod
    def process(self, image: np.ndarray, detections: List[DetectionResult]) -> AadhaarData:
        """Process card and extract information"""
        pass
    
    def _get_highest_confidence_detection(self, detections: List[DetectionResult], 
                                        name: str) -> Optional[DetectionResult]:
        """Get detection with highest confidence for given name"""
        filtered = [d for d in detections if d.name == name]
        if not filtered:
            return None
        return max(filtered, key=lambda x: x.confidence)
    
    def _match_ocr_with_yolo(self, ocr_results: List[OCRResult], 
                           yolo_detection: DetectionResult) -> Optional[OCRResult]:
        """Match OCR result with YOLO detection using IoU"""
        best_match = None
        best_iou = 0
        
        for ocr_result in ocr_results:
            iou = GeometryUtils.calculate_iou(ocr_result.bbox, yolo_detection.bbox)
            if iou > best_iou:
                best_iou = iou
                best_match = ocr_result
        
        return best_match if best_iou > 0 else None

class FrontCardProcessor(BaseCardProcessor):
    """Processes front side of Aadhaar card"""
    
    def process(self, image: np.ndarray, detections: List[DetectionResult]) -> AadhaarData:
        """Process front card"""
        try:
            self.logger.info("Processing front card")
            
            # Get card detection
            card_detection = self._get_highest_confidence_detection(detections, CardType.FRONT_CARD.value)
            if not card_detection:
                self.logger.warning("No front card detection found")
                return AadhaarData()
            
            # Extract and process image
            processed_image = self._preprocess_image(image, detections, CardType.FRONT_CARD)
            
            # Extract text using OCR
            ocr_results = self.ocr_engine.extract_text(processed_image)
            
            # Create result object
            result = AadhaarData()
            
            # Process each field
            self._process_id_number(result, detections, ocr_results)
            self._process_name(result, detections, ocr_results)
            self._process_date_of_birth(result, detections, ocr_results)
            self._process_gender(result, detections, ocr_results)
            
            self.logger.info("Front card processing completed")
            return result
            
        except Exception as e:
            self.logger.error(f"Front card processing failed: {e}")
            return AadhaarData()
    
    def _preprocess_image(self, image: np.ndarray, detections: List[DetectionResult], 
                         card_type: CardType) -> np.ndarray:
        """Preprocess image for OCR"""
        try:
            # Apply mask
            exclusions = ['face_photo', 'aadhaar_qr', 'footer', 'header_front_card']
            masked_image = self.image_processor.apply_mask(image, detections, card_type, exclusions)
            
            # Get card bounding box and crop
            card_detection = self._get_highest_confidence_detection(detections, card_type.value)
            if card_detection:
                cropped_image = self.image_processor.crop_image(masked_image, card_detection.bbox)
                return cropped_image
            
            return masked_image
        except Exception as e:
            self.logger.error(f"Image preprocessing failed: {e}")
            return image
    
    def _process_id_number(self, result: AadhaarData, detections: List[DetectionResult], 
                          ocr_results: List[OCRResult]):
        """Process ID number field"""
        try:
            detection = self._get_highest_confidence_detection(detections, 'id_masked')
            if detection:
                ocr_match = self._match_ocr_with_yolo(ocr_results, detection)
                if ocr_match:
                    result.id_number = ocr_match.text
                    result.confidence_scores['ID_NUMBER'] = ocr_match.confidence
                    result.yolo_scores['ID_NUMBER'] = detection.confidence
        except Exception as e:
            self.logger.error(f"ID number processing failed: {e}")
    
    def _process_name(self, result: AadhaarData, detections: List[DetectionResult], 
                     ocr_results: List[OCRResult]):
        """Process name field"""
        try:
            detection = self._get_highest_confidence_detection(detections, 'aadhaar_name')
            if detection:
                ocr_match = self._match_ocr_with_yolo(ocr_results, detection)
                if ocr_match:
                    result.name_on_card = ocr_match.text
                    result.confidence_scores['NAME_ON_CARD'] = ocr_match.confidence
                    result.yolo_scores['NAME_ON_CARD'] = detection.confidence
        except Exception as e:
            self.logger.error(f"Name processing failed: {e}")
    
    def _process_date_of_birth(self, result: AadhaarData, detections: List[DetectionResult], 
                              ocr_results: List[OCRResult]):
        """Process date of birth field"""
        try:
            detection = self._get_highest_confidence_detection(detections, 'aadhaar_dob')
            if detection:
                ocr_match = self._match_ocr_with_yolo(ocr_results, detection)
                if ocr_match:
                    parsed_date = self.date_parser.parse(ocr_match.text)
                    result.date_of_birth = parsed_date
                    result.confidence_scores['DATE_OF_BIRTH'] = ocr_match.confidence
                    result.yolo_scores['DATE_OF_BIRTH'] = detection.confidence
        except Exception as e:
            self.logger.error(f"Date of birth processing failed: {e}")
    
    def _process_gender(self, result: AadhaarData, detections: List[DetectionResult], 
                       ocr_results: List[OCRResult]):
        """Process gender field"""
        try:
            detection = self._get_highest_confidence_detection(detections, 'aadhaar_gender')
            if detection:
                ocr_match = self._match_ocr_with_yolo(ocr_results, detection)
                if ocr_match:
                    parsed_gender = self.gender_parser.parse(ocr_match.text)
                    result.gender = parsed_gender
                    result.confidence_scores['GENDER'] = ocr_match.confidence
                    result.yolo_scores['GENDER'] = detection.confidence
        except Exception as e:
            self.logger.error(f"Gender processing failed: {e}")

class RearCardProcessor(BaseCardProcessor):
    """Processes rear side of Aadhaar card"""
    
    def process(self, image: np.ndarray, detections: List[DetectionResult]) -> AadhaarData:
        """Process rear card"""
        try:
            self.logger.info("Processing rear card")
            
            # Get card detection
            card_detection = self._get_highest_confidence_detection(detections, CardType.REAR_CARD.value)
            if not card_detection:
                self.logger.warning("No rear card detection found")
                return AadhaarData()
            
            # Extract and process image
            processed_image = self._preprocess_image(image, detections, CardType.REAR_CARD)
            
            # Extract text using OCR
            ocr_results = self.ocr_engine.extract_text(processed_image)
            
            # Combine all OCR text for address processing
            address_text = self._combine_ocr_text(ocr_results)
            
            # Parse address
            pin_code, state, parent_name = self.address_parser.parse(address_text)
            
            # Create result
            result = AadhaarData()
            result.address = address_text
            result.pin_code = pin_code
            result.state = state
            result.parent_name = parent_name
            
            # Calculate average confidence
            if ocr_results:
                avg_confidence = sum(ocr.confidence for ocr in ocr_results) / len(ocr_results)
                result.confidence_scores['ADDRESS'] = avg_confidence
            
            self.logger.info("Rear card processing completed")
            return result
            
        except Exception as e:
            self.logger.error(f"Rear card processing failed: {e}")
            return AadhaarData()
    
    def _preprocess_image(self, image: np.ndarray, detections: List[DetectionResult], 
                         card_type: CardType) -> np.ndarray:
        """Preprocess image for OCR"""
        try:
            # Get card bounding box and crop
            card_detection = self._get_highest_confidence_detection(detections, card_type.value)
            if card_detection:
                cropped_image = self.image_processor.crop_image(image, card_detection.bbox)
                return cropped_image
            
            return image
        except Exception as e:
            self.logger.error(f"Image preprocessing failed: {e}")
            return image
    
    def _combine_ocr_text(self, ocr_results: List[OCRResult]) -> str:
        """Combine OCR results into address text"""
        if not ocr_results:
            return ""
        
        combined_text = ""
        for idx, ocr in enumerate(ocr_results):
            if idx == 0:
                combined_text = ocr.text
            else:
                combined_text += "," + ocr.text
        
        return combined_text

class CardProcessorFactory:
    """Factory for creating card processors"""
    
    @staticmethod
    def create_processor(card_type: CardType, ocr_engine: BaseOCREngine, 
                        image_processor: ImageProcessor, pin_data: Dict[str, str]) -> BaseCardProcessor:
        """Create appropriate card processor"""
        if card_type == CardType.FRONT_CARD:
            return FrontCardProcessor(ocr_engine, image_processor, pin_data)
        elif card_type == CardType.REAR_CARD:
            return RearCardProcessor(ocr_engine, image_processor, pin_data)
        else:
            raise ValueError(f"Unsupported card type: {card_type}")
