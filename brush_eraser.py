"""
🖌️ BRUSH ERASER MODULE
======================
Perfect working brush erase functionality with accurate preview
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any


class BrushEraser:
    """
    Professional brush eraser with accurate preview
    """
    
    def __init__(self):
        self.default_brush_size = 20
        self.background_color = [255, 255, 255]  # White background
    
    def apply_brush_erase(
        self, 
        image: np.ndarray, 
        x: float, 
        y: float, 
        brush_size: int = None
    ) -> np.ndarray:
        """
        Apply brush erase to the image at specified coordinates
        
        Args:
            image: Input image as numpy array
            x: X coordinate (normalized 0-1)
            y: Y coordinate (normalized 0-1)
            brush_size: Brush size (default: 20)
        
        Returns:
            Modified image with erased area
        """
        if brush_size is None:
            brush_size = self.default_brush_size
        
        # Scale coordinates to image space
        img_x = int(x * image.shape[1])
        img_y = int(y * image.shape[0])
        img_brush_size = brush_size * 2  # Match frontend calculation
        
        # Create mask for erasing
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (img_x, img_y), img_brush_size, 255, -1)
        
        # Apply erasing
        img_erased = image.copy()
        img_erased[mask > 0] = self.background_color
        
        return img_erased
    
    def get_preview_size(self, brush_size: int = None) -> int:
        """
        Get the preview circle size that matches the actual erase area
        
        Args:
            brush_size: Brush size (default: 20)
        
        Returns:
            Circle diameter for preview
        """
        if brush_size is None:
            brush_size = self.default_brush_size
        
        # Backend calculation: img_brush_size = brush_size * 2 (radius)
        # Preview diameter = radius * 2 = brush_size * 4
        return brush_size * 4
    
    def validate_coordinates(self, x: float, y: float) -> bool:
        """
        Validate if coordinates are within valid range
        
        Args:
            x: X coordinate (0-1)
            y: Y coordinate (0-1)
        
        Returns:
            True if valid, False otherwise
        """
        return 0 <= x <= 1 and 0 <= y <= 1
    
    def get_brush_info(self, brush_size: int = None) -> Dict[str, Any]:
        """
        Get brush information for debugging/display
        
        Args:
            brush_size: Brush size (default: 20)
        
        Returns:
            Dictionary with brush information
        """
        if brush_size is None:
            brush_size = self.default_brush_size
        
        return {
            "brush_size": brush_size,
            "actual_radius": brush_size * 2,
            "preview_diameter": brush_size * 4,
            "background_color": self.background_color,
            "description": f"Brush size {brush_size} -> Radius {brush_size * 2} -> Preview diameter {brush_size * 4}"
        }


# Example usage and testing
if __name__ == "__main__":
    # Create brush eraser instance
    eraser = BrushEraser()
    
    # Test brush info
    info = eraser.get_brush_info(25)
    print("🖌️ Brush Info:")
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    # Test coordinate validation
    print(f"\n✅ Valid coordinates (0.5, 0.5): {eraser.validate_coordinates(0.5, 0.5)}")
    print(f"❌ Invalid coordinates (1.5, 0.5): {eraser.validate_coordinates(1.5, 0.5)}")
    
    # Test preview size calculation
    for size in [10, 20, 30]:
        preview = eraser.get_preview_size(size)
        print(f"📐 Brush size {size} -> Preview diameter {preview}") 