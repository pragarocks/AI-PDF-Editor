"""
🔍 OCR EDITOR MODULE
===================
Advanced OCR text detection and real-time editing with live preview
"""

import cv2
import numpy as np
import easyocr
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Tuple, Optional, Any
import uuid
import re
import logging


class OCRTextElement:
    """
    Represents a detected OCR text element with editing capabilities
    """
    
    def __init__(
        self, 
        text: str, 
        bbox: List[float], 
        confidence: float,
        page_num: int = 0,
        element_id: str = None
    ):
        self.id = element_id or str(uuid.uuid4())
        self.original_text = text
        self.current_text = text
        self.bbox = bbox  # [x1, y1, x2, y2] normalized coordinates
        self.confidence = confidence
        self.page_num = page_num
        
        # Editing state
        self.is_selected = False
        self.is_editing = False
        self.is_modified = False
        self.is_deleted = False
        
        # Visual properties
        self.highlight_color = "#FFE4B5"  # Light orange for OCR text
        self.edit_color = "#87CEEB"       # Sky blue for editing
        self.selection_color = "#FF6347"  # Tomato for selection
        
        # Font properties (auto-detected from OCR)
        self.estimated_font_size = self._estimate_font_size()
        self.font_family = "Arial"  # Default, can be improved with font detection
        
    def _estimate_font_size(self) -> int:
        """Estimate font size based on bounding box height with better scaling"""
        height = self.bbox[3] - self.bbox[1]
        width = self.bbox[2] - self.bbox[0]
        
        # Calculate text density (characters per unit area)
        text_length = len(self.original_text.strip())
        if text_length == 0:
            return 16
        
        # More sophisticated size estimation based on text density and bounding box
        area = height * width
        density = text_length / max(area, 0.001)  # Avoid division by zero
        
        # Adjust scale factor based on text density
        if density > 100:  # Dense text (small font)
            scale_factor = 3000
        elif density > 50:  # Medium density
            scale_factor = 4000
        else:  # Sparse text (large font)
            scale_factor = 6000
        
        # Convert normalized height to approximate pixel size
        estimated_size = max(16, min(250, int(height * scale_factor)))
        return estimated_size
    
    def get_center(self) -> Tuple[float, float]:
        """Get center point of the text element"""
        x = (self.bbox[0] + self.bbox[2]) / 2
        y = (self.bbox[1] + self.bbox[3]) / 2
        return (x, y)
    
    def get_width(self) -> float:
        """Get width of the text element"""
        return self.bbox[2] - self.bbox[0]
    
    def get_height(self) -> float:
        """Get height of the text element"""
        return self.bbox[3] - self.bbox[1]
    
    def contains_point(self, x: float, y: float, tolerance: float = 0.01) -> bool:
        """Check if a point is within this text element"""
        return (self.bbox[0] - tolerance <= x <= self.bbox[2] + tolerance and
                self.bbox[1] - tolerance <= y <= self.bbox[3] + tolerance)
    
    def update_text(self, new_text: str):
        """Update the text content"""
        if new_text != self.current_text:
            self.current_text = new_text
            self.is_modified = True
    
    def reset_text(self):
        """Reset to original text"""
        self.current_text = self.original_text
        self.is_modified = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "original_text": self.original_text,
            "current_text": self.current_text,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "page_num": self.page_num,
            "is_selected": self.is_selected,
            "is_editing": self.is_editing,
            "is_modified": self.is_modified,
            "is_deleted": self.is_deleted,
            "estimated_font_size": self.estimated_font_size,
            "font_family": self.font_family
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OCRTextElement':
        """Create from dictionary"""
        element = cls(
            text=data["original_text"],
            bbox=data["bbox"],
            confidence=data["confidence"],
            page_num=data.get("page_num", 0),
            element_id=data.get("id")
        )
        
        # Apply all properties
        for key, value in data.items():
            if hasattr(element, key):
                setattr(element, key, value)
        
        return element


class OCREditor:
    """
    Advanced OCR text detection and editing system with real-time preview
    """
    
    def __init__(self, languages=['en']):
        self.ocr_reader = None
        self.languages = languages
        self.text_elements: Dict[str, OCRTextElement] = {}
        self.selected_element_id: Optional[str] = None
        self.editing_element_id: Optional[str] = None
        
        # OCR settings
        self.confidence_threshold = 0.3
        self.merge_threshold = 0.02  # Merge nearby text elements
        
        # Initialize OCR
        self._initialize_ocr()
        
        # Font cache for rendering
        self.font_cache: Dict[str, ImageFont.FreeTypeFont] = {}
        
    def _initialize_ocr(self):
        """Initialize EasyOCR reader"""
        try:
            self.ocr_reader = easyocr.Reader(self.languages, gpu=False, verbose=False)
            logging.info("✅ EasyOCR initialized successfully!")
        except Exception as e:
            logging.error(f"❌ Failed to initialize EasyOCR: {str(e)}")
            self.ocr_reader = None
    
    def detect_text(self, image: np.ndarray, page_num: int = 0) -> List[OCRTextElement]:
        """
        Detect text in image using OCR
        
        Args:
            image: Input image as numpy array
            page_num: Page number
            
        Returns:
            List of detected text elements
        """
        if self.ocr_reader is None:
            logging.warning("OCR reader not initialized")
            return []
        
        try:
            # Convert image for OCR
            if len(image.shape) == 3:
                ocr_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                ocr_image = image
            
            # Run OCR
            results = self.ocr_reader.readtext(ocr_image)
            
            # Convert results to OCRTextElement objects
            text_elements = []
            img_height, img_width = image.shape[:2]
            
            for result in results:
                bbox_coords, text, confidence = result
                
                # Skip low confidence detections
                if confidence < self.confidence_threshold:
                    continue
                
                # Convert bbox to normalized coordinates
                bbox_array = np.array(bbox_coords)
                x_coords = bbox_array[:, 0] / img_width
                y_coords = bbox_array[:, 1] / img_height
                
                normalized_bbox = [
                    float(np.min(x_coords)),  # x1
                    float(np.min(y_coords)),  # y1
                    float(np.max(x_coords)),  # x2
                    float(np.max(y_coords))   # y2
                ]
                
                element = OCRTextElement(
                    text=text.strip(),
                    bbox=normalized_bbox,
                    confidence=confidence,
                    page_num=page_num
                )
                
                text_elements.append(element)
                self.text_elements[element.id] = element
                
                logging.info(f"Added OCR text: '{text}' with confidence {confidence:.2f}")
            
            # Post-process: merge nearby text elements if needed
            text_elements = self._merge_nearby_elements(text_elements)
            
            logging.info(f"OCR found {len(text_elements)} text elements on page {page_num}")
            return text_elements
            
        except Exception as e:
            logging.error(f"Error during OCR detection: {str(e)}")
            return []
    
    def _merge_nearby_elements(self, elements: List[OCRTextElement]) -> List[OCRTextElement]:
        """
        Merge nearby text elements that likely belong together
        """
        if len(elements) <= 1:
            return elements
        
        merged = []
        used_indices = set()
        
        for i, elem1 in enumerate(elements):
            if i in used_indices:
                continue
                
            # Find nearby elements to merge
            merge_candidates = [elem1]
            used_indices.add(i)
            
            for j, elem2 in enumerate(elements):
                if j in used_indices or j == i:
                    continue
                
                # Check if elements are close enough to merge
                if self._should_merge_elements(elem1, elem2):
                    merge_candidates.append(elem2)
                    used_indices.add(j)
            
            # Merge if we have multiple candidates
            if len(merge_candidates) > 1:
                merged_element = self._merge_elements(merge_candidates)
                merged.append(merged_element)
            else:
                merged.append(elem1)
        
        return merged
    
    def _should_merge_elements(self, elem1: OCRTextElement, elem2: OCRTextElement) -> bool:
        """
        Determine if two elements should be merged based on proximity
        """
        # Check vertical alignment and horizontal proximity
        y1_center = (elem1.bbox[1] + elem1.bbox[3]) / 2
        y2_center = (elem2.bbox[1] + elem2.bbox[3]) / 2
        
        # Check if they're on roughly the same line
        y_diff = abs(y1_center - y2_center)
        avg_height = (elem1.get_height() + elem2.get_height()) / 2
        
        if y_diff > avg_height * 0.5:  # Not on same line
            return False
        
        # Check horizontal proximity
        x_gap = min(abs(elem1.bbox[2] - elem2.bbox[0]), 
                   abs(elem2.bbox[2] - elem1.bbox[0]))
        
        return x_gap < self.merge_threshold
    
    def _merge_elements(self, elements: List[OCRTextElement]) -> OCRTextElement:
        """
        Merge multiple text elements into one
        """
        # Sort by x position (left to right)
        elements.sort(key=lambda e: e.bbox[0])
        
        # Combine text with spaces
        combined_text = " ".join(elem.original_text for elem in elements)
        
        # Calculate combined bounding box
        min_x = min(elem.bbox[0] for elem in elements)
        min_y = min(elem.bbox[1] for elem in elements)
        max_x = max(elem.bbox[2] for elem in elements)
        max_y = max(elem.bbox[3] for elem in elements)
        
        # Use highest confidence
        max_confidence = max(elem.confidence for elem in elements)
        
        # Create merged element
        merged = OCRTextElement(
            text=combined_text,
            bbox=[min_x, min_y, max_x, max_y],
            confidence=max_confidence,
            page_num=elements[0].page_num
        )
        
        return merged
    
    def get_element_at_position(self, x: float, y: float, page_num: int) -> Optional[str]:
        """
        Find text element at given position
        
        Args:
            x: X coordinate (normalized 0-1)
            y: Y coordinate (normalized 0-1)
            page_num: Page number
            
        Returns:
            Element ID if found, None otherwise
        """
        for element_id, element in self.text_elements.items():
            if (element.page_num == page_num and 
                not element.is_deleted and
                element.contains_point(x, y)):
                return element_id
        return None
    
    def select_element(self, element_id: str) -> bool:
        """
        Select a text element for editing
        
        Args:
            element_id: Element ID
            
        Returns:
            True if successful, False if element not found
        """
        if element_id not in self.text_elements:
            return False
        
        # Deselect previous element
        if self.selected_element_id:
            self.text_elements[self.selected_element_id].is_selected = False
        
        # Select new element
        self.selected_element_id = element_id
        self.text_elements[element_id].is_selected = True
        
        return True
    
    def start_editing(self, element_id: str) -> bool:
        """
        Start editing a text element
        
        Args:
            element_id: Element ID
            
        Returns:
            True if successful, False if element not found
        """
        if element_id not in self.text_elements:
            return False
        
        # Stop previous editing
        if self.editing_element_id:
            self.text_elements[self.editing_element_id].is_editing = False
        
        # Start editing
        self.editing_element_id = element_id
        self.text_elements[element_id].is_editing = True
        self.text_elements[element_id].is_selected = True
        self.selected_element_id = element_id
        
        return True
    
    def stop_editing(self) -> bool:
        """
        Stop editing current element
        
        Returns:
            True if was editing, False otherwise
        """
        if self.editing_element_id:
            self.text_elements[self.editing_element_id].is_editing = False
            self.editing_element_id = None
            return True
        return False
    
    def update_element_text(self, element_id: str, new_text: str) -> bool:
        """
        Update text content of an element
        
        Args:
            element_id: Element ID
            new_text: New text content
            
        Returns:
            True if successful, False if element not found
        """
        if element_id not in self.text_elements:
            return False
        
        self.text_elements[element_id].update_text(new_text)
        return True
    
    def delete_element(self, element_id: str) -> bool:
        """
        Mark element as deleted
        
        Args:
            element_id: Element ID
            
        Returns:
            True if successful, False if element not found
        """
        if element_id not in self.text_elements:
            return False
        
        self.text_elements[element_id].is_deleted = True
        
        # Clear selection if this element was selected
        if self.selected_element_id == element_id:
            self.selected_element_id = None
        
        if self.editing_element_id == element_id:
            self.editing_element_id = None
        
        return True
    
    def restore_element(self, element_id: str) -> bool:
        """
        Restore a deleted element
        
        Args:
            element_id: Element ID
            
        Returns:
            True if successful, False if element not found
        """
        if element_id not in self.text_elements:
            return False
        
        self.text_elements[element_id].is_deleted = False
        return True
    
    def get_elements_by_page(self, page_num: int, include_deleted: bool = False) -> List[OCRTextElement]:
        """
        Get all text elements for a specific page
        
        Args:
            page_num: Page number
            include_deleted: Whether to include deleted elements
            
        Returns:
            List of text elements
        """
        elements = []
        for element in self.text_elements.values():
            if element.page_num == page_num:
                if include_deleted or not element.is_deleted:
                    elements.append(element)
        
        # Sort by position (top to bottom, left to right)
        elements.sort(key=lambda e: (e.bbox[1], e.bbox[0]))
        return elements
    
    def render_ocr_overlay(self, image: np.ndarray, page_num: int = 0) -> np.ndarray:
        """
        Render OCR text overlay on image with real-time editing
        
        Args:
            image: Input image as numpy array
            page_num: Page number to render
            
        Returns:
            Image with OCR overlay
        """
        # Convert numpy array to PIL Image
        if len(image.shape) == 3:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(image)
        
        # Create overlay for drawing
        overlay = Image.new('RGBA', pil_image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Get elements for this page
        elements = self.get_elements_by_page(page_num)
        
        for element in elements:
            self._render_single_ocr_element(draw, pil_image, element)
        
        # Composite overlay onto original image
        pil_image = pil_image.convert('RGBA')
        final_image = Image.alpha_composite(pil_image, overlay)
        
        # Convert back to numpy array
        if len(image.shape) == 3:
            return cv2.cvtColor(np.array(final_image), cv2.COLOR_RGBA2BGR)
        else:
            return np.array(final_image.convert('L'))
    
    def render_clean_text_replacement(self, image: np.ndarray, page_num: int = 0) -> np.ndarray:
        """
        Render edited text as overlays on top of the original text
        
        Args:
            image: Input image as numpy array
            page_num: Page number to render
            
        Returns:
            Image with edited text overlays
        """
        # Convert numpy array to PIL Image
        if len(image.shape) == 3:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(image)
        
        # Create drawing object
        draw = ImageDraw.Draw(pil_image)
        
        # Get elements for this page
        elements = self.get_elements_by_page(page_num)
        
        # Debug: Log how many elements we're checking
        modified_count = 0
        for element in elements:
            # Only render modified elements (text that has been changed)
            if element.current_text != element.original_text:
                modified_count += 1
                logging.info(f"🔍 Rendering modified text: '{element.original_text}' -> '{element.current_text}'")
                self._render_clean_text_replacement(draw, pil_image, element)
        
        logging.info(f"🔍 Total modified elements rendered: {modified_count}")
        
        # Convert back to numpy array
        if len(image.shape) == 3:
            return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        else:
            return np.array(pil_image.convert('L'))

    def _render_clean_text_replacement(self, draw: ImageDraw.Draw, image: Image.Image, element: OCRTextElement):
        """Render edited text as a perfect replacement that completely removes original text and replaces it seamlessly"""
        # Calculate position in pixels
        img_x1 = int(element.bbox[0] * image.width)
        img_y1 = int(element.bbox[1] * image.height)
        img_x2 = int(element.bbox[2] * image.width)
        img_y2 = int(element.bbox[3] * image.height)
        
        # Calculate bounding box dimensions
        bbox_width = img_x2 - img_x1
        bbox_height = img_y2 - img_y1
        
        # STEP 1: Completely erase the original text area
        # Create a white rectangle to cover the entire original text area
        erase_padding = 2  # Slight padding to ensure complete coverage
        erase_x1 = max(0, img_x1 - erase_padding)
        erase_y1 = max(0, img_y1 - erase_padding)
        erase_x2 = min(image.width, img_x2 + erase_padding)
        erase_y2 = min(image.height, img_y2 + erase_padding)
        
        # Draw white rectangle to completely erase the original text
        draw.rectangle([erase_x1, erase_y1, erase_x2, erase_y2], fill=(255, 255, 255))
        
        # STEP 2: Detect the original text color from surrounding areas
        # Sample colors from areas around the text to get the document's text color
        original_color = self._detect_text_color_from_surroundings(image, element.bbox)
        
        # STEP 3: Calculate perfect font size to match original text
        # Use the original bounding box height as the target
        target_height = bbox_height * 0.85  # 85% of bounding box height for text
        
        # Start with a reasonable font size and adjust
        font_size = max(8, min(int(target_height), 150))
        
        # Get font
        try:
            font = self._get_font(element.font_family, font_size)
        except:
            font = ImageFont.load_default()
        
        # Calculate text dimensions with current font
        text_bbox = draw.textbbox((0, 0), element.current_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Fine-tune font size to match original text height precisely
        if text_height > bbox_height * 0.95:
            # Text too large, reduce font size
            while text_height > bbox_height * 0.95 and font_size > 6:
                font_size -= 1
                try:
                    font = self._get_font(element.font_family, font_size)
                    text_bbox = draw.textbbox((0, 0), element.current_text, font=font)
                    text_height = text_bbox[3] - text_bbox[1]
                except:
                    break
        elif text_height < bbox_height * 0.6:
            # Text too small, increase font size
            while text_height < bbox_height * 0.6 and font_size < 150:
                font_size += 1
                try:
                    font = self._get_font(element.font_family, font_size)
                    text_bbox = draw.textbbox((0, 0), element.current_text, font=font)
                    text_height = text_bbox[3] - text_bbox[1]
                except:
                    break
        
        # Recalculate final text dimensions
        text_bbox = draw.textbbox((0, 0), element.current_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # STEP 4: Position text to match the original text position exactly
        # Center horizontally and vertically within the original bounding box
        text_x = img_x1 + (bbox_width - text_width) // 2
        text_y = img_y1 + (bbox_height - text_height) // 2
        
        # Ensure text doesn't go outside image bounds
        text_x = max(0, min(text_x, image.width - text_width))
        text_y = max(0, min(text_y, image.height - text_height))
        
        # STEP 5: Draw the new text with perfect color matching
        # Use the detected original color for perfect matching
        if original_color == (255, 255, 255) or original_color == (0, 0, 0):
            # Fallback to a dark color that should match most documents
            text_color = (30, 30, 30)  # Very dark gray, almost black
        else:
            text_color = original_color
        
        # Draw the text directly on the erased area
        draw.text((text_x, text_y), element.current_text, fill=text_color, font=font)
        
        # Log the rendering details for debugging
        logging.info(f"🎨 Perfect text replacement: '{element.current_text}' at ({text_x}, {text_y}) with color {text_color}, font size {font_size}")
    
    def _detect_text_color_from_surroundings(self, image: Image.Image, bbox: List[float]) -> Tuple[int, int, int]:
        """
        Detect text color from surrounding areas to get the document's text color
        """
        try:
            # Convert normalized coordinates to pixel coordinates
            img_x1 = int(bbox[0] * image.width)
            img_y1 = int(bbox[1] * image.height)
            img_x2 = int(bbox[2] * image.width)
            img_y2 = int(bbox[3] * image.height)
            
            # Sample colors from areas around the text (not from the text itself)
            colors = []
            
            # Sample from left side
            if img_x1 > 20:
                for y in range(img_y1, img_y2, max(1, (img_y2 - img_y1) // 5)):
                    try:
                        pixel = image.getpixel((img_x1 - 10, y))
                        if isinstance(pixel, tuple) and len(pixel) >= 3:
                            r, g, b = pixel[:3]
                            if r < 240 or g < 240 or b < 240:  # Not white
                                colors.append((r, g, b))
                    except:
                        continue
            
            # Sample from right side
            if img_x2 < image.width - 20:
                for y in range(img_y1, img_y2, max(1, (img_y2 - img_y1) // 5)):
                    try:
                        pixel = image.getpixel((img_x2 + 10, y))
                        if isinstance(pixel, tuple) and len(pixel) >= 3:
                            r, g, b = pixel[:3]
                            if r < 240 or g < 240 or b < 240:  # Not white
                                colors.append((r, g, b))
                    except:
                        continue
            
            # Sample from above
            if img_y1 > 20:
                for x in range(img_x1, img_x2, max(1, (img_x2 - img_x1) // 5)):
                    try:
                        pixel = image.getpixel((x, img_y1 - 10))
                        if isinstance(pixel, tuple) and len(pixel) >= 3:
                            r, g, b = pixel[:3]
                            if r < 240 or g < 240 or b < 240:  # Not white
                                colors.append((r, g, b))
                    except:
                        continue
            
            # Sample from below
            if img_y2 < image.height - 20:
                for x in range(img_x1, img_x2, max(1, (img_x2 - img_x1) // 5)):
                    try:
                        pixel = image.getpixel((x, img_y2 + 10))
                        if isinstance(pixel, tuple) and len(pixel) >= 3:
                            r, g, b = pixel[:3]
                            if r < 240 or g < 240 or b < 240:  # Not white
                                colors.append((r, g, b))
                    except:
                        continue
            
            if not colors:
                return (30, 30, 30)  # Default dark gray
            
            # Find the most common color
            color_counts = {}
            for color in colors:
                color_counts[color] = color_counts.get(color, 0) + 1
            
            # Return the most frequent color
            dominant_color = max(color_counts.items(), key=lambda x: x[1])[0]
            return dominant_color
            
        except Exception as e:
            logging.warning(f"Failed to detect text color from surroundings: {str(e)}")
            return (30, 30, 30)  # Default dark gray

    def _render_single_ocr_element(self, draw: ImageDraw.Draw, image: Image.Image, element: OCRTextElement):
        """Render a single OCR text element with appropriate styling"""
        # Calculate position in pixels
        img_x1 = int(element.bbox[0] * image.width)
        img_y1 = int(element.bbox[1] * image.height)
        img_x2 = int(element.bbox[2] * image.width)
        img_y2 = int(element.bbox[3] * image.height)
        
        # Choose colors based on state
        if element.is_editing:
            bg_color = self._hex_to_rgba(element.edit_color, 0.4)
            border_color = element.edit_color
            border_width = 3
        elif element.is_selected:
            bg_color = self._hex_to_rgba(element.selection_color, 0.3)
            border_color = element.selection_color
            border_width = 2
        elif element.is_modified:
            bg_color = self._hex_to_rgba("#90EE90", 0.3)  # Light green for modified
            border_color = "#32CD32"  # Lime green
            border_width = 2
        else:
            bg_color = self._hex_to_rgba(element.highlight_color, 0.2)
            border_color = "#DEB887"  # Burlywood
            border_width = 1
        
        # Draw background
        draw.rectangle([img_x1, img_y1, img_x2, img_y2], fill=bg_color)
        
        # Draw border
        for i in range(border_width):
            draw.rectangle(
                [img_x1 - i, img_y1 - i, img_x2 + i, img_y2 + i],
                outline=border_color
            )
        
        # Draw text if modified or editing (show current text)
        if element.is_modified or element.is_editing:
            try:
                # Calculate font size using the same improved scaling logic
                bbox_height = img_y2 - img_y1
                base_font_size = element.estimated_font_size
                scale_factor = bbox_height / (base_font_size * 0.6)  # More aggressive scaling
                font_size = max(16, min(int(base_font_size * scale_factor), 200))
                
                font = self._get_font(element.font_family, font_size)
                
                # Calculate text position (centered in bbox)
                text_bbox = draw.textbbox((0, 0), element.current_text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                text_x = img_x1 + (img_x2 - img_x1 - text_width) // 2
                text_y = img_y1 + (img_y2 - img_y1 - text_height) // 2
                
                # Draw text with outline for visibility
                text_color = "#000000"  # Black text
                outline_color = "#FFFFFF"  # White outline
                
                # Draw text outline
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx != 0 or dy != 0:
                            draw.text((text_x + dx, text_y + dy), element.current_text, 
                                    fill=outline_color, font=font)
                
                # Draw main text
                draw.text((text_x, text_y), element.current_text, fill=text_color, font=font)
                
            except Exception as e:
                logging.warning(f"Failed to render text for element {element.id}: {str(e)}")
        
        # Add confidence indicator for low confidence elements
        if element.confidence < 0.7:
            confidence_text = f"{element.confidence:.1f}"
            small_font = self._get_font("Arial", 10)
            draw.text((img_x1 + 2, img_y1 + 2), confidence_text, 
                     fill="#FF0000", font=small_font)
    
    def _get_font(self, font_family: str, font_size: int) -> ImageFont.FreeTypeFont:
        """Get font object with caching"""
        cache_key = f"{font_family}_{font_size}"
        
        if cache_key in self.font_cache:
            return self.font_cache[cache_key]
        
        try:
            # Try to load better system fonts in order of preference
            font_paths = [
                "arial.ttf",  # Windows Arial
                "Arial.ttf",  # Windows Arial (capitalized)
                "DejaVuSans.ttf",  # Linux DejaVu Sans
                "LiberationSans-Regular.ttf",  # Linux Liberation Sans
                "Helvetica.ttf",  # macOS Helvetica
            ]
            
            font = None
            for font_path in font_paths:
                try:
                    font = ImageFont.truetype(font_path, font_size)
                    break
                except:
                    continue
            
            # Fallback to default font if no system fonts found
            if font is None:
                font = ImageFont.load_default()
            
            self.font_cache[cache_key] = font
            return font
            
        except Exception as e:
            logging.warning(f"Failed to load font {font_family} size {font_size}: {str(e)}")
            # Ultimate fallback
            font = ImageFont.load_default()
            self.font_cache[cache_key] = font
            return font
    
    def _hex_to_rgba(self, hex_color: str, alpha: float = 1.0) -> Tuple[int, int, int, int]:
        """Convert hex color to RGBA tuple"""
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return rgb + (int(alpha * 255),)
    
    def get_editor_state(self) -> Dict[str, Any]:
        """Get complete editor state for frontend"""
        return {
            "text_elements": [elem.to_dict() for elem in self.text_elements.values() if not elem.is_deleted],
            "selected_element_id": self.selected_element_id,
            "editing_element_id": self.editing_element_id,
            "total_elements": len([e for e in self.text_elements.values() if not e.is_deleted]),
            "modified_elements": len([e for e in self.text_elements.values() if e.is_modified and not e.is_deleted]),
            "confidence_threshold": self.confidence_threshold
        }
    
    def clear_all(self):
        """Clear all text elements"""
        self.text_elements.clear()
        self.selected_element_id = None
        self.editing_element_id = None


# Example usage and testing
if __name__ == "__main__":
    editor = OCREditor()
    
    print("🔍 OCR Editor Module - Example Usage")
    print(f"OCR initialized: {editor.ocr_reader is not None}")
    print(f"Languages: {editor.languages}")
    print(f"Confidence threshold: {editor.confidence_threshold}")
    
    # Mock some text elements for testing
    elem1 = OCRTextElement(
        text="Sample Invoice Text",
        bbox=[0.1, 0.1, 0.4, 0.15],
        confidence=0.95,
        page_num=0
    )
    
    elem2 = OCRTextElement(
        text="Total Amount: $123.45",
        bbox=[0.6, 0.8, 0.9, 0.85],
        confidence=0.88,
        page_num=0
    )
    
    editor.text_elements[elem1.id] = elem1
    editor.text_elements[elem2.id] = elem2
    
    # Test editing workflow
    print(f"\n📝 Testing Editing Workflow:")
    print(f"Element 1 text: '{elem1.current_text}'")
    
    # Select and edit
    editor.select_element(elem1.id)
    editor.start_editing(elem1.id)
    editor.update_element_text(elem1.id, "Modified Invoice Text")
    
    print(f"Modified text: '{elem1.current_text}'")
    print(f"Is modified: {elem1.is_modified}")
    print(f"Is editing: {elem1.is_editing}")
    
    # Get state
    state = editor.get_editor_state()
    print(f"\n📊 Editor State:")
    print(f"Total elements: {state['total_elements']}")
    print(f"Modified elements: {state['modified_elements']}")
    print(f"Selected: {state['selected_element_id'] is not None}")
    print(f"Editing: {state['editing_element_id'] is not None}") 