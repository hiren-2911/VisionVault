# aadhaar_processor.py
import base64
import time
import json
import logging
from typing import Dict, List, Tuple, Any
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageSequence
from io import BytesIO
import torch

from data_models import (
    DetectionResult, BoundingBox, AadhaarData, 
    ProcessingMetrics, CardType
)
from config_manager import ConfigManager
from image_processor import ImageProcessor
from ocr_engine import PaddleOCREngine
from card_processors import CardProcessorFactory

class DocumentConverter:
    """Handles conversion of different document formats"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def convert_pdf_to_images(self, pdf_b64: str, filename: str) -> List[Tuple[str, np.ndarray]]:
        """Convert PDF to list of images"""
        try:
            from pdf2image import convert_from_bytes
            
            self.logger.info(f"Converting PDF: {filename}")
            
            # Decode base64
            pdf_data = base64.b64decode(pdf_b64.encode('ascii'))
            
            # Convert to images
            pdf_pages = convert_from_bytes(pdf_data)
            
            images = []
            base_name = Path(filename).stem
            
            for idx, page in enumerate(pdf_pages):
                # Convert to BGR for OpenCV
                img_array = np.array(page)
                bgr_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                img_name = f"{base_name}_pdf_page_{idx}.jpg"
                images.append((img_name, bgr_img))
            
            self.logger.info(f"PDF converted to {len(images)} images")
            return images
        except Exception as e:
            self.logger.error(f"PDF conversion failed: {e}")
            return []
    
    def convert_tiff_to_images(self, tiff_b64: str, filename: str) -> List[Tuple[str, np.ndarray]]:
        """Convert TIFF to list of images"""
        try:
            self.logger.info(f"Converting TIFF: {filename}")
            
            # Decode base64
            decoded_data = base64.b64decode(tiff_b64.encode('ascii'))
            tiff_file = BytesIO(decoded_data)
            
            # Open TIFF
            tiff_img = Image.open(tiff_file)
            
            images = []
            base_name = Path(filename).stem
            
            for idx, page in enumerate(ImageSequence.Iterator(tiff_img)):
                # Convert to BGR for OpenCV
                img_array = np.array(page.convert("RGB"))
                bgr_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                img_name = f"{base_name}_tiff_page_{idx}.jpg"
                images.append((img_name, bgr_img))
            
            self.logger.info(f"TIFF converted to {len(images)} images")
            return images
        except Exception as e:
            self.logger.error(f"TIFF conversion failed: {e}")
            return []
    
    def convert_image_from_base64(self, img_b64: str, filename: str) -> Tuple[str, np.ndarray]:
        """Convert base64 image to OpenCV format"""
        try:
            self.logger.info(f"Converting image: {filename}")
            
            # Decode base64
            decoded_data = base64.b64decode(img_b64.encode('ascii'))
            img_arr = np.frombuffer(decoded_data, dtype=np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("Failed to decode image")
            
            # Convert grayscale to RGB if needed
            if len(img.shape) < 3:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
            return filename, img
        except Exception as e:
            self.logger.error(f"Image conversion failed: {e}")
            raise

class AadhaarProcessor:
    """Main Aadhaar processing pipeline"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.image_processor = ImageProcessor()
        self.document_converter = DocumentConverter()
        self.ocr_engine = PaddleOCREngine()
        
        # Load PIN data
        self.pin_data = self._load_pin_data()
        
        # Load detection model
        self.detection_model = self._load_detection_model()
    
    def _load_pin_data(self) -> Dict[str, str]:
        """Load PIN code to state mapping"""
        try:
            pin_file = self.config.get_config('pin_json_path',"")
            with open(pin_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load PIN data: {e}")
            return {}
    
    def _load_detection_model(self):
        """Load YOLO detection model"""
        try:
            model_path = self.config.get_config('aadhar_yolo_weights',"")
            yolo_repo = self.config.get_config('yolo_repo',"")
            model = torch.hub.load("E:/OCR/aadhar_ocr/yolov5", "custom", source='local', path=model_path)
            model = model.to(self.config.device)
            self.logger.info(f"Detection model loaded on {self.config.device}")
            return model
        except Exception as e:
            self.logger.error(f"Model loading failed: {e}")
            raise
    
    def process_document(self, ref_id: str, filename: str, b64_content: str) -> Dict[str, Any]:
        """Main document processing method"""
        try:
            self.logger.info(f"[{ref_id}] Processing document: {filename}")
            
            # Convert to images
            images = self._convert_to_images(filename, b64_content)
            
            # Process each image
            results = []
            for page_idx, (img_name, img_array) in enumerate(images):
                result = self._process_single_image(img_array, img_name, page_idx)
                results.append(result)
            
            # Format output
            output = self._format_output(results)
            
            self.logger.info(f"[{ref_id}] Document processing completed")
            return output
            
        except Exception as e:
            self.logger.error(f"[{ref_id}] Document processing failed: {e}")
            raise
    
    def _convert_to_images(self, filename: str, b64_content: str) -> List[Tuple[str, np.ndarray]]:
        """Convert document to images based on file type"""
        file_ext = Path(filename).suffix.lower()
        
        if file_ext == '.pdf':
            return self.document_converter.convert_pdf_to_images(b64_content, filename)
        elif file_ext in ['.tif', '.tiff']:
            return self.document_converter.convert_tiff_to_images(b64_content, filename)
        elif file_ext in ['.jpg', '.jpeg', '.png']:
            img_name, img_array = self.document_converter.convert_image_from_base64(b64_content, filename)
            return [(img_name, img_array)]
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def _process_single_image(self, image: np.ndarray, image_name: str, page_idx: int) -> Dict[str, Any]:
        """Process a single image"""
        try:
            start_time = time.time()
            
            # Object detection
            yolo_start = time.time()
            detections = self._detect_objects(image)
            ocr_start = time.time()
            
            # Process based on detected card types
            result = self._process_detected_cards(image, detections)
            
            end_time = time.time()
            
            # Add timing information
            result.metrics.processing_time = end_time - start_time
            result.metrics.yolo_time = ocr_start - yolo_start
            result.metrics.ocr_time = end_time - ocr_start
            
            # Format result
            output = result.to_dict()
            output['image_name'] = image_name
            output['page_count'] = page_idx + 1
            output['device'] = self.config.device
            
            return output
            
        except Exception as e:
            self.logger.error(f"Single image processing failed: {e}")
            return {}
    
    def _detect_objects(self, image: np.ndarray) -> List[DetectionResult]:
        """Detect objects using YOLO model"""
        try:
            results = self.detection_model(image)
            results_df = results.pandas().xyxy[0]
            
            # Filter by confidence
            results_df = results_df[results_df['confidence'] >= self.config.yolo_threshold]
            
            detections = []
            for _, row in results_df.iterrows():
                bbox = BoundingBox(
                    xmin=int(row['xmin']),
                    ymin=int(row['ymin']),
                    xmax=int(row['xmax']),
                    ymax=int(row['ymax']),
                    confidence=row['confidence']
                )
                
                detection = DetectionResult(
                    name=row['name'],
                    bbox=bbox,
                    confidence=row['confidence']
                )
                detections.append(detection)
            
            self.logger.info(f"Detected {len(detections)} objects")
            return detections
            
        except Exception as e:
            self.logger.error(f"Object detection failed: {e}")
            return []
    
    def _process_detected_cards(self, image: np.ndarray, detections: List[DetectionResult]) -> AadhaarData:
        """Process detected cards"""
        try:
            detected_names = {d.name for d in detections}
            
            # Determine processing strategy
            if {"front_card", "rear_card"}.issubset(detected_names):
                return self._process_front_and_rear(image, detections)
            elif {"front_card", "front_long"}.issubset(detected_names):
                return self._process_front_and_long(image, detections)
            elif "front_card" in detected_names:
                return self._process_front_only(image, detections)
            elif "rear_card" in detected_names:
                return self._process_rear_only(image, detections)
            else:
                self.logger.warning("No recognized card types detected")
                return AadhaarData()
                
        except Exception as e:
            self.logger.error(f"Card processing failed: {e}")
            return AadhaarData()
    
    def _process_front_and_rear(self, image: np.ndarray, detections: List[DetectionResult]) -> AadhaarData:
        """Process front and rear cards"""
        # Process front card
        front_processor = CardProcessorFactory.create_processor(
            CardType.FRONT_CARD, self.ocr_engine, self.image_processor, self.pin_data
        )
        front_result = front_processor.process(image, detections)
        
        # Process rear card
        rear_processor = CardProcessorFactory.create_processor(
            CardType.REAR_CARD, self.ocr_engine, self.image_processor, self.pin_data
        )
        rear_result = rear_processor.process(image, detections)
        
        # Merge results
        return self._merge_results(front_result, rear_result)
    
    def _process_front_and_long(self, image: np.ndarray, detections: List[DetectionResult]) -> AadhaarData:
        """Process front card and long format"""
        # Process front card
        front_processor = CardProcessorFactory.create_processor(
            CardType.FRONT_CARD, self.ocr_engine, self.image_processor, self.pin_data
        )
        front_result = front_processor.process(image, detections)
        
        # Process long format (similar to rear card processing)
        rear_processor = CardProcessorFactory.create_processor(
            CardType.REAR_CARD, self.ocr_engine, self.image_processor, self.pin_data
        )
        long_result = rear_processor.process(image, detections)
        
        return self._merge_results(front_result, long_result)
    
    def _process_front_only(self, image: np.ndarray, detections: List[DetectionResult]) -> AadhaarData:
        """Process front card only"""
        processor = CardProcessorFactory.create_processor(
            CardType.FRONT_CARD, self.ocr_engine, self.image_processor, self.pin_data
        )
        return processor.process(image, detections)
    
    def _process_rear_only(self, image: np.ndarray, detections: List[DetectionResult]) -> AadhaarData:
        """Process rear card only"""
        processor = CardProcessorFactory.create_processor(
            CardType.REAR_CARD, self.ocr_engine, self.image_processor, self.pin_data
        )
        return processor.process(image, detections)
    
    def _merge_results(self, front_result: AadhaarData, rear_result: AadhaarData) -> AadhaarData:
        """Merge front and rear card results"""
        # Start with front card data
        merged = front_result
        
        # Add rear card address information
        if rear_result.address != "Nan":
            merged.address = rear_result.address
        if rear_result.pin_code != "Nan":
            merged.pin_code = rear_result.pin_code
        if rear_result.state != "Nan":
            merged.state = rear_result.state
        if rear_result.parent_name != "Nan":
            merged.parent_name = rear_result.parent_name
        
        # Merge confidence scores
        merged.confidence_scores.update(rear_result.confidence_scores)
        
        return merged
    
    def _format_output(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format final output"""
        output = {}
        
        for idx, result in enumerate(results):
            image_key = f"Image_{idx + 1}"
            
            # Normalize keys to lowercase
            normalized_result = self._normalize_keys(result)
            output[image_key] = normalized_result
        
        return output
    
    def _normalize_keys(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize dictionary keys to lowercase"""
        if isinstance(data, dict):
            return {k.lower(): self._normalize_keys(v) for k, v in data.items()}
        return data

# Main API function
def process_aadhaar_document(ref_id: str, filename: str, b64_content: str, 
                           config_manager:ConfigManager) -> Dict[str, Any]:
    """
    Main API function to process Aadhaar documents
    
    Args:
        ref_id: Reference ID for tracking
        filename: Input filename
        b64_content: Base64 encoded file content
        config_path: Path to configuration file
    
    Returns:
        Dictionary containing processed results
    """
    try:
        # Initialize components
        #config_manager = config_manager#ConfigManager(config_path, logger_config_path="E:/OCR/aadhar_ocr/config/logger.json" ,application_name= "aadhar_ocr", console_log=True)
        processor = AadhaarProcessor(config_manager)
        
        # Process document
        result = processor.process_document(ref_id, filename, b64_content)
        
        return result
        
    except Exception as e:
        logging.error(f"[{ref_id}] Processing failed: {e}")
        raise

# Example usage
# if __name__ == "__main__":
#     # Example usage
#     with open("E:/OCR/aadhar_ocr/sample_inputs/aadhar/AP3111TW0022349_550374016_input.json", "r") as f:
#         input_data = json.load(f)
    
#     result = process_aadhaar_document(
#         ref_id="TEST001",
#         filename=input_data["filename"],
#         b64_content=input_data["input_b64"]
#     )
    
#     print(json.dumps(result, indent=2))
