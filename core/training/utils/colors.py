"""
Color utilities for QModel Trainer

Provides color conversion and default color palette functions
for visualization, reporting, and class coloring.
"""

from typing import Tuple


# Default color palette for semantic segmentation classes
DEFAULT_COLOR_PALETTE_RGB = [
    (0, 0, 0),       # 0: Background (black)
    (0, 255, 0),     # 1: Green
    (255, 0, 0),     # 2: Red
    (0, 0, 255),     # 3: Blue
    (255, 255, 0),   # 4: Yellow
    (255, 0, 255),   # 5: Magenta
    (0, 255, 255),   # 6: Cyan
    (255, 165, 0),   # 7: Orange
    (128, 0, 128),   # 8: Purple
    (255, 192, 203), # 9: Pink
]


def get_default_color_rgb(index: int) -> Tuple[int, int, int]:
    """
    Get default RGB color for a class by index.
    
    Uses a predefined palette for common classes (0-9),
    then generates colors procedurally for higher indices.
    
    Args:
        index: Class index (0-based)
        
    Returns:
        RGB color tuple (R, G, B) with values 0-255
        
    Examples:
        >>> get_default_color_rgb(0)
        (0, 0, 0)  # Black (background)
        >>> get_default_color_rgb(1)
        (0, 255, 0)  # Green
    """
    if index < len(DEFAULT_COLOR_PALETTE_RGB):
        return DEFAULT_COLOR_PALETTE_RGB[index]
    else:
        # Generate color procedurally using prime number offsets
        # This ensures colors are distributed across RGB space
        r = (index * 137) % 256
        g = (index * 197) % 256
        b = (index * 257) % 256
        return (r, g, b)


def get_default_color_hex(index: int) -> str:
    """
    Get default hex color for a class by index.
    
    Args:
        index: Class index (0-based)
        
    Returns:
        Hex color string in format '#RRGGBB'
        
    Examples:
        >>> get_default_color_hex(0)
        '#000000'
        >>> get_default_color_hex(1)
        '#00ff00'
    """
    r, g, b = get_default_color_rgb(index)
    return f'#{r:02x}{g:02x}{b:02x}'


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """
    Convert hex color string to RGB tuple.
    
    Args:
        hex_color: Hex color string ('#RRGGBB' or 'RRGGBB')
        
    Returns:
        RGB color tuple (R, G, B) with values 0-255
        
    Raises:
        ValueError: If hex_color is not a valid hex color
        
    Examples:
        >>> hex_to_rgb('#ff0000')
        (255, 0, 0)
        >>> hex_to_rgb('00ff00')
        (0, 255, 0)
    """
    # Remove '#' prefix if present
    hex_color = hex_color.lstrip('#')
    
    if len(hex_color) != 6:
        raise ValueError(f"Invalid hex color: {hex_color} (must be 6 characters)")
    
    try:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return (r, g, b)
    except ValueError as e:
        raise ValueError(f"Invalid hex color: {hex_color}") from e


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """
    Convert RGB tuple to hex color string.
    
    Args:
        rgb: RGB color tuple (R, G, B) with values 0-255
        
    Returns:
        Hex color string in format '#RRGGBB'
        
    Examples:
        >>> rgb_to_hex((255, 0, 0))
        '#ff0000'
        >>> rgb_to_hex((0, 255, 0))
        '#00ff00'
    """
    r, g, b = rgb
    return f'#{r:02x}{g:02x}{b:02x}'


def parse_color(color: str | Tuple[int, int, int]) -> Tuple[int, int, int]:
    """
    Parse color from string or tuple to RGB tuple.
    
    Accepts:
    - Hex color string ('#RRGGBB' or 'RRGGBB')
    - RGB tuple (R, G, B)
    
    Args:
        color: Color as hex string or RGB tuple
        
    Returns:
        RGB color tuple (R, G, B) with values 0-255
        
    Examples:
        >>> parse_color('#ff0000')
        (255, 0, 0)
        >>> parse_color((0, 255, 0))
        (0, 255, 0)
    """
    if isinstance(color, str):
        return hex_to_rgb(color)
    elif isinstance(color, (tuple, list)) and len(color) == 3:
        return tuple(color)
    else:
        raise ValueError(f"Invalid color format: {color}")