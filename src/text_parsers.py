# text_parsers.py
import re
import logging
from typing import Dict, Tuple
from difflib import get_close_matches

class TextParserBase:
    """Base class for text parsers"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)

class DateParser(TextParserBase):
    """Handles date parsing from OCR text"""
    
    def __init__(self):
        super().__init__()
        self.character_map = {
            'O': '0', 'Q': '0', 'T': '1', 'S': '5', 'Z': '2', 
            'I': '1', 'J': '7', 'G': '6', 'B': '8'
        }
    
    def _rectify_characters(self, text: str) -> str:
        """Replace commonly misrecognized characters with correct digits"""
        rectified = ""
        for char in text:
            if char in self.character_map:
                rectified += self.character_map[char]
                self.logger.debug(f"Replaced '{char}' with '{self.character_map[char]}'")
            else:
                rectified += char
        return rectified
    
    def _is_valid_date_format(self, date_string: str) -> bool:
        """Check if string matches date pattern MM/DD/YYYY or MM/DD/YY"""
        pattern = r"\d{1,2}/\d{1,2}/\d{2,4}"
        return bool(re.fullmatch(pattern, date_string))
    
    def parse(self, ocr_text: str) -> str:
        """Parse date from OCR text"""
        try:
            self.logger.debug(f"Parsing date from: '{ocr_text}'")
            
            # Find pivot points
            colon_idx = ocr_text.find(":")
            space_idx = ocr_text.find(" ")
            pivot = max(colon_idx, space_idx)
            
            if pivot > 0:
                ocr_text = ocr_text[pivot + 1:]
            
            # Process based on length and content
            if len(ocr_text) > 4:
                if '/' not in ocr_text:
                    ocr_text = ocr_text[-4:]  # Take last 4 characters (year)
                elif len(ocr_text) >= 10:
                    ocr_text = ocr_text[-10:]  # Take last 10 characters (full date)
                
                # Validate extracted date
                if not self._is_valid_date_format(ocr_text):
                    self.logger.warning(f"Invalid date format: '{ocr_text}', using last 4 characters")
                    ocr_text = ocr_text[-4:]
            
            # Rectify characters
            rectified = self._rectify_characters(ocr_text)
            
            self.logger.info(f"Date parsed successfully: '{rectified}'")
            return rectified
        except Exception as e:
            self.logger.error(f"Date parsing failed: {e}")
            return ocr_text

class GenderParser(TextParserBase):
    """Handles gender parsing from OCR text"""
    
    def parse(self, ocr_text: str) -> str:
        """Parse gender from OCR text"""
        try:
            self.logger.debug(f"Parsing gender from: '{ocr_text}'")
            
            if not ocr_text:
                return ocr_text
            
            # Check for male indicators
            if ocr_text[0].lower() == 'm':
                self.logger.info("Gender detected as Male (starts with M)")
                return "Male"
            
            # Check for slash separator
            slash_idx = ocr_text.find("/")
            if slash_idx == -1:
                self.logger.info("No slash found, defaulting to Male")
                return "Male"
            
            # Check character after slash
            if (slash_idx + 1 < len(ocr_text) and 
                ocr_text[slash_idx + 1].lower() == 'm'):
                self.logger.info("Gender detected as Male (M after slash)")
                return "Male"
            
            self.logger.info("Gender detected as Female")
            return "Female"
        except Exception as e:
            self.logger.error(f"Gender parsing failed: {e}")
            return ocr_text

class AddressParser(TextParserBase):
    """Handles address parsing to extract PIN, state, and parent name"""
    
    def __init__(self, pin_data: Dict[str, str]):
        super().__init__()
        self.pin_data = pin_data
    
    def _find_best_pincode(self, pin_code: str) -> str:
        """Find the best matching PIN code using fuzzy matching"""
        if pin_code in self.pin_data:
            return pin_code
        
        # Try fuzzy matching
        matches = get_close_matches(pin_code, list(self.pin_data.keys()), 
                                   n=1, cutoff=0.6)
        if matches:
            corrected = matches[0]
            self.logger.info(f"PIN code '{pin_code}' corrected to '{corrected}'")
            return corrected
        
        return pin_code
    
    def _extract_parent_name(self, text: str) -> str:
        """Extract parent/spouse name from text"""
        prefixes = ["C/O", "S/O", "W/O", "D/O", "F/O", "SO", "SIO", "WO", "CO", "DO"]
        
        text = text.strip().upper()
        
        # Remove known prefixes
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
                break
        
        # Remove leading non-alphabetic characters
        while text and not text[0].isalpha():
            text = text[1:]
        
        # Extract only alphabetic characters and spaces
        name = ""
        for char in text:
            if char.isalpha() or char.isspace():
                name += char
            else:
                break
        
        return name.strip().title()
    
    def parse(self, address: str) -> Tuple[str, str, str]:
        """Parse address to extract PIN code, state, and parent name"""
        try:
            self.logger.debug(f"Parsing address: '{address}'")
            
            pin_code = state = parent_name = "NOT FOUND"
            
            # Preprocess address
            address = address.replace("-", ",")
            address_chunks = address.split(",")
            
            # Find numeric sequences (potential PIN codes)
            numeric_sequences = re.findall(r'\d+', address)
            
            # Identify PIN code
            for seq in numeric_sequences:
                if len(seq) in [5, 6]:
                    pin_code = seq
                    break
            
            # Find best matching PIN code
            if pin_code != "NOT FOUND":
                pin_code = self._find_best_pincode(pin_code)
                
                # Get state from PIN code
                if pin_code in self.pin_data:
                    state = self.pin_data[pin_code]
                    self.logger.info(f"State found for PIN {pin_code}: {state}")
            
            # Extract parent name
            name_indicators = ["/", "C/O", "W/O", "S/O", "D/O", "SO", "WO", "CO", "DO"]
            
            for idx, chunk in enumerate(address_chunks):
                if idx <= 3:  # Only check first few chunks
                    for indicator in name_indicators:
                        if indicator in chunk:
                            indicator_idx = chunk.find(indicator)
                            if len(chunk) - indicator_idx > 3:
                                parent_name = self._extract_parent_name(
                                    chunk[indicator_idx + len(indicator):]
                                )
                            elif idx + 1 < len(address_chunks):
                                parent_name = self._extract_parent_name(
                                    address_chunks[idx + 1]
                                )
                            
                            if parent_name and len(parent_name) > 1:
                                break
                    
                    if parent_name != "NOT FOUND":
                        break
            
            self.logger.info(f"Address parsed: PIN={pin_code}, State={state}, Parent={parent_name}")
            return pin_code, state, parent_name
        except Exception as e:
            self.logger.error(f"Address parsing failed: {e}")
            return "NOT FOUND", "NOT FOUND", "NOT FOUND"

class TextParserFactory:
    """Factory for creating text parsers"""
    
    @staticmethod
    def create_date_parser() -> DateParser:
        return DateParser()
    
    @staticmethod
    def create_gender_parser() -> GenderParser:
        return GenderParser()
    
    @staticmethod
    def create_address_parser(pin_data: Dict[str, str]) -> AddressParser:
        return AddressParser(pin_data)
