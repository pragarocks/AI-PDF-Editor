"""
📝 TEXT EDITOR MODULE
====================
Complete text editing functionality with styling, positioning, and management
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Tuple, Optional, Any
import uuid
import os


class TextElement:
    """
    Represents a single text element with all its properties
    """
    
    def __init__(
        self, 
        text: str, 
        x: float, 
        y: float, 
        page_num: int = 0,
        element_id: str = None
    ):
        self.id = element_id or str(uuid.uuid4())
        self.text = text
        self.x = x  # Normalized coordinates (0-1)
        self.y = y
        self.page_num = page_num
        
        # Text styling properties
        self.font_family = "Arial"
        self.font_size = 16
        self.font_weight = "normal"  # normal, bold
        self.font_style = "normal"   # normal, italic
        self.color = "#000000"       # Hex color
        self.background_color = None # None for transparent, hex for highlight
        self.text_decoration = "none" # none, underline, strikethrough
        self.text_align = "left"     # left, center, right
        self.opacity = 1.0           # 0.0 to 1.0
        
        # Additional properties
        self.rotation = 0            # Degrees
        self.border_color = None     # None or hex color
        self.border_width = 0        # Border thickness
        self.padding = 4             # Padding around text
        self.shadow = False          # Text shadow
        self.shadow_color = "#666666"
        self.shadow_offset = (2, 2)  # (x, y) offset
        
        # State
        self.is_selected = False
        self.is_editing = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert text element to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "text": self.text,
            "x": self.x,
            "y": self.y,
            "page_num": self.page_num,
            "font_family": self.font_family,
            "font_size": self.font_size,
            "font_weight": self.font_weight,
            "font_style": self.font_style,
            "color": self.color,
            "background_color": self.background_color,
            "text_decoration": self.text_decoration,
            "text_align": self.text_align,
            "opacity": self.opacity,
            "rotation": self.rotation,
            "border_color": self.border_color,
            "border_width": self.border_width,
            "padding": self.padding,
            "shadow": self.shadow,
            "shadow_color": self.shadow_color,
            "shadow_offset": self.shadow_offset,
            "is_selected": self.is_selected,
            "is_editing": self.is_editing
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TextElement':
        """Create text element from dictionary"""
        element = cls(
            text=data["text"],
            x=data["x"],
            y=data["y"],
            page_num=data.get("page_num", 0),
            element_id=data.get("id")
        )
        
        # Apply all properties
        for key, value in data.items():
            if hasattr(element, key):
                setattr(element, key, value)
        
        return element


class TextEditor:
    """
    Professional text editor with comprehensive styling and management
    """
    
    def __init__(self):
        self.text_elements: Dict[str, TextElement] = {}
        self.selected_element_id: Optional[str] = None
        self.font_cache: Dict[str, ImageFont.FreeTypeFont] = {}
        
        # Default fonts directory
        self.fonts_dir = self._get_fonts_directory()
        
        # Available fonts
        self.available_fonts = [
            "Arial", "Times New Roman", "Courier New", "Comic Sans MS",
            "Impact", "Georgia", "Trebuchet MS", "Verdana", "Tahoma"
        ]
        
        # Available colors
        self.preset_colors = [
            "#000000", "#FF0000", "#00FF00", "#0000FF", "#FFFF00",
            "#FF00FF", "#00FFFF", "#800000", "#008000", "#000080",
            "#808080", "#C0C0C0", "#800080", "#008080", "#FFA500"
        ]
        
        # Available sizes
        self.preset_sizes = [8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 48, 72]
    
    def _get_fonts_directory(self) -> str:
        """Get system fonts directory"""
        import platform
        system = platform.system()
        
        if system == "Windows":
            return "C:/Windows/Fonts/"
        elif system == "Darwin":  # macOS
            return "/System/Library/Fonts/"
        else:  # Linux
            return "/usr/share/fonts/"
    
    def _get_font(self, font_family: str, font_size: int, bold: bool = False, italic: bool = False) -> ImageFont.FreeTypeFont:
        """Get font object with caching"""
        cache_key = f"{font_family}_{font_size}_{bold}_{italic}"
        
        if cache_key in self.font_cache:
            return self.font_cache[cache_key]
        
        try:
            # Try to load system font
            font_filename = self._get_font_filename(font_family, bold, italic)
            font_path = os.path.join(self.fonts_dir, font_filename)
            
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
            else:
                # Fallback to default font
                font = ImageFont.load_default()
            
            self.font_cache[cache_key] = font
            return font
            
        except Exception:
            # Ultimate fallback
            font = ImageFont.load_default()
            self.font_cache[cache_key] = font
            return font
    
    def _get_font_filename(self, font_family: str, bold: bool, italic: bool) -> str:
        """Get font filename based on family and style"""
        font_map = {
            "Arial": {
                (False, False): "arial.ttf",
                (True, False): "arialbd.ttf",
                (False, True): "ariali.ttf",
                (True, True): "arialbi.ttf"
            },
            "Times New Roman": {
                (False, False): "times.ttf",
                (True, False): "timesbd.ttf",
                (False, True): "timesi.ttf",
                (True, True): "timesbi.ttf"
            },
            "Courier New": {
                (False, False): "cour.ttf",
                (True, False): "courbd.ttf",
                (False, True): "couri.ttf",
                (True, True): "courbi.ttf"
            }
        }
        
        family_fonts = font_map.get(font_family, font_map["Arial"])
        return family_fonts.get((bold, italic), family_fonts[(False, False)])
    
    def add_text_element(
        self, 
        text: str, 
        x: float, 
        y: float, 
        page_num: int = 0,
        **styling_options
    ) -> str:
        """
        Add a new text element
        
        Args:
            text: Text content
            x: X coordinate (normalized 0-1)
            y: Y coordinate (normalized 0-1)
            page_num: Page number
            **styling_options: Font styling options
        
        Returns:
            Element ID
        """
        element = TextElement(text, x, y, page_num)
        
        # Apply styling options
        for key, value in styling_options.items():
            if hasattr(element, key):
                setattr(element, key, value)
        
        self.text_elements[element.id] = element
        return element.id
    
    def update_text_element(self, element_id: str, **updates) -> bool:
        """
        Update an existing text element
        
        Args:
            element_id: Element ID
            **updates: Properties to update
        
        Returns:
            True if successful, False if element not found
        """
        if element_id not in self.text_elements:
            return False
        
        element = self.text_elements[element_id]
        for key, value in updates.items():
            if hasattr(element, key):
                setattr(element, key, value)
        
        return True
    
    def delete_text_element(self, element_id: str) -> bool:
        """
        Delete a text element
        
        Args:
            element_id: Element ID
        
        Returns:
            True if successful, False if element not found
        """
        if element_id in self.text_elements:
            del self.text_elements[element_id]
            if self.selected_element_id == element_id:
                self.selected_element_id = None
            return True
        return False
    
    def select_element(self, element_id: str) -> bool:
        """
        Select a text element for editing
        
        Args:
            element_id: Element ID
        
        Returns:
            True if successful, False if element not found
        """
        if element_id in self.text_elements:
            # Deselect previous element
            if self.selected_element_id:
                self.text_elements[self.selected_element_id].is_selected = False
            
            # Select new element
            self.selected_element_id = element_id
            self.text_elements[element_id].is_selected = True
            return True
        return False
    
    def get_selected_element(self) -> Optional[TextElement]:
        """Get the currently selected element"""
        if self.selected_element_id:
            return self.text_elements.get(self.selected_element_id)
        return None
    
    def get_elements_by_page(self, page_num: int) -> List[TextElement]:
        """Get all text elements for a specific page"""
        return [elem for elem in self.text_elements.values() if elem.page_num == page_num]
    
    def render_text_on_image(self, image: np.ndarray, page_num: int = 0) -> np.ndarray:
        """
        Render all text elements for a page onto an image
        
        Args:
            image: Input image as numpy array
            page_num: Page number to render
        
        Returns:
            Image with text rendered
        """
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        # Get elements for this page
        elements = self.get_elements_by_page(page_num)
        
        # Sort by creation order (render older elements first)
        elements.sort(key=lambda x: x.id)
        
        for element in elements:
            self._render_single_element(draw, pil_image, element)
        
        # Convert back to numpy array
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    def _render_single_element(self, draw: ImageDraw.Draw, image: Image.Image, element: TextElement):
        """Render a single text element"""
        # Calculate position
        img_x = int(element.x * image.width)
        img_y = int(element.y * image.height)
        
        # Get font
        is_bold = element.font_weight == "bold"
        is_italic = element.font_style == "italic"
        font = self._get_font(element.font_family, element.font_size, is_bold, is_italic)
        
        # Calculate text dimensions
        bbox = draw.textbbox((0, 0), element.text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Apply text alignment
        if element.text_align == "center":
            img_x -= text_width // 2
        elif element.text_align == "right":
            img_x -= text_width
        
        # Draw background/highlight if specified
        if element.background_color:
            bg_rect = [
                img_x - element.padding,
                img_y - element.padding,
                img_x + text_width + element.padding,
                img_y + text_height + element.padding
            ]
            draw.rectangle(bg_rect, fill=element.background_color)
        
        # Draw border if specified
        if element.border_color and element.border_width > 0:
            border_rect = [
                img_x - element.padding,
                img_y - element.padding,
                img_x + text_width + element.padding,
                img_y + text_height + element.padding
            ]
            for i in range(element.border_width):
                draw.rectangle(
                    [border_rect[0] - i, border_rect[1] - i, border_rect[2] + i, border_rect[3] + i],
                    outline=element.border_color
                )
        
        # Draw shadow if enabled
        if element.shadow:
            shadow_x = img_x + element.shadow_offset[0]
            shadow_y = img_y + element.shadow_offset[1]
            draw.text((shadow_x, shadow_y), element.text, fill=element.shadow_color, font=font)
        
        # Draw main text
        text_color = element.color
        if element.opacity < 1.0:
            # Apply opacity (simplified - would need more complex implementation for true alpha)
            r, g, b = self._hex_to_rgb(text_color)
            text_color = f"#{int(r * element.opacity):02x}{int(g * element.opacity):02x}{int(b * element.opacity):02x}"
        
        draw.text((img_x, img_y), element.text, fill=text_color, font=font)
        
        # Draw text decorations
        if element.text_decoration == "underline":
            line_y = img_y + text_height
            draw.line([(img_x, line_y), (img_x + text_width, line_y)], fill=text_color, width=1)
        elif element.text_decoration == "strikethrough":
            line_y = img_y + text_height // 2
            draw.line([(img_x, line_y), (img_x + text_width, line_y)], fill=text_color, width=1)
        
        # Draw selection indicator if selected
        if element.is_selected:
            selection_rect = [
                img_x - 2,
                img_y - 2,
                img_x + text_width + 2,
                img_y + text_height + 2
            ]
            draw.rectangle(selection_rect, outline="#0066FF", width=2)
    
    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
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
        # This would need more sophisticated collision detection
        # For now, simple distance-based detection
        for element in self.get_elements_by_page(page_num):
            distance = ((element.x - x) ** 2 + (element.y - y) ** 2) ** 0.5
            if distance < 0.05:  # Within 5% of image size
                return element.id
        return None
    
    def get_editor_state(self) -> Dict[str, Any]:
        """Get complete editor state for frontend"""
        return {
            "text_elements": [elem.to_dict() for elem in self.text_elements.values()],
            "selected_element_id": self.selected_element_id,
            "available_fonts": self.available_fonts,
            "preset_colors": self.preset_colors,
            "preset_sizes": self.preset_sizes
        }


# Example usage and testing
if __name__ == "__main__":
    editor = TextEditor()
    
    # Add some test text elements
    elem1_id = editor.add_text_element(
        "Hello World!", 
        0.5, 0.2, 
        font_size=24, 
        font_weight="bold", 
        color="#FF0000"
    )
    
    elem2_id = editor.add_text_element(
        "Styled Text", 
        0.3, 0.5, 
        font_family="Times New Roman",
        font_style="italic",
        background_color="#FFFF00",
        text_decoration="underline"
    )
    
    # Test selection
    editor.select_element(elem1_id)
    
    # Test updates
    editor.update_text_element(elem1_id, color="#0000FF", font_size=32)
    
    print("📝 Text Editor Module - Example Usage")
    print(f"Created elements: {list(editor.text_elements.keys())}")
    print(f"Selected element: {editor.selected_element_id}")
    
    # Test state export
    state = editor.get_editor_state()
    print(f"Available fonts: {len(state['available_fonts'])}")
    print(f"Available colors: {len(state['preset_colors'])}")
    print(f"Total elements: {len(state['text_elements'])}") 