#!/usr/bin/env python3

import json
import sys
from typing import Dict, Any, List


class Shape:
    """Base Shape class that stores ALL properties individually for each instance."""
    
    def __init__(self, shape_type: str, x: float, y: float, width: float, height: float,
                 color: str, texture: str, line_style: str):
        # Each instance stores its own copy of all properties - memory inefficient!
        self.shape_type = shape_type
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.texture = texture
        self.line_style = line_style
    
    def render(self) -> str:
        """Render the shape to a string representation."""
        return f"{self.shape_type} at ({self.x},{self.y}) size {self.width}x{self.height} with color={self.color}, texture={self.texture}, line_style={self.line_style}"


class Circle(Shape):
    """Circle class - stores all properties individually."""
    
    def __init__(self, x: float, y: float, width: float, height: float,
                 color: str, texture: str, line_style: str):
        super().__init__("circle", x, y, width, height, color, texture, line_style)
        # Store duplicate properties for extra memory waste
        self.radius = width / 2
        self.center_x = x
        self.center_y = y
        self.circle_color = color
        self.circle_texture = texture
        self.circle_line_style = line_style
    
    def render(self) -> str:
        return f"circle at ({self.center_x},{self.center_y}) radius {self.radius} with color={self.circle_color}, texture={self.circle_texture}, line_style={self.circle_line_style}"


class Rectangle(Shape):
    """Rectangle class - stores all properties individually."""
    
    def __init__(self, x: float, y: float, width: float, height: float,
                 color: str, texture: str, line_style: str):
        super().__init__("rectangle", x, y, width, height, color, texture, line_style)
        # Store duplicate properties for extra memory waste
        self.rect_x = x
        self.rect_y = y
        self.rect_width = width
        self.rect_height = height
        self.rect_color = color
        self.rect_texture = texture
        self.rect_line_style = line_style
    
    def render(self) -> str:
        return f"rectangle at ({self.rect_x},{self.rect_y}) size {self.rect_width}x{self.rect_height} with color={self.rect_color}, texture={self.rect_texture}, line_style={self.rect_line_style}"


class Triangle(Shape):
    """Triangle class - stores all properties individually."""
    
    def __init__(self, x: float, y: float, width: float, height: float,
                 color: str, texture: str, line_style: str):
        super().__init__("triangle", x, y, width, height, color, texture, line_style)
        # Store duplicate properties for extra memory waste
        self.tri_x = x
        self.tri_y = y
        self.tri_width = width
        self.tri_height = height
        self.tri_color = color
        self.tri_texture = texture
        self.tri_line_style = line_style
    
    def render(self) -> str:
        return f"triangle at ({self.tri_x},{self.tri_y}) size {self.tri_width}x{self.tri_height} with color={self.tri_color}, texture={self.tri_texture}, line_style={self.tri_line_style}"


class Polygon(Shape):
    """Polygon class - stores all properties individually."""
    
    def __init__(self, x: float, y: float, width: float, height: float,
                 color: str, texture: str, line_style: str):
        super().__init__("polygon", x, y, width, height, color, texture, line_style)
        # Store duplicate properties for extra memory waste
        self.poly_x = x
        self.poly_y = y
        self.poly_width = width
        self.poly_height = height
        self.poly_color = color
        self.poly_texture = texture
        self.poly_line_style = line_style
    
    def render(self) -> str:
        return f"polygon at ({self.poly_x},{self.poly_y}) size {self.poly_width}x{self.poly_height} with color={self.poly_color}, texture={self.poly_texture}, line_style={self.poly_line_style}"


def create_shape(shape_data: Dict[str, Any]) -> Shape:
    """Factory function to create appropriate shape object based on type."""
    shape_type = shape_data.get('type', 'rectangle')
    x = shape_data.get('x', 0.0)
    y = shape_data.get('y', 0.0)
    width = shape_data.get('width', 10.0)
    height = shape_data.get('height', 10.0)
    color = shape_data.get('color', 'black')
    texture = shape_data.get('texture', 'solid')
    line_style = shape_data.get('line_style', 'solid')
    
    if shape_type == 'circle':
        return Circle(x, y, width, height, color, texture, line_style)
    elif shape_type == 'rectangle':
        return Rectangle(x, y, width, height, color, texture, line_style)
    elif shape_type == 'triangle':
        return Triangle(x, y, width, height, color, texture, line_style)
    elif shape_type == 'polygon':
        return Polygon(x, y, width, height, color, texture, line_style)
    else:
        # Default to generic Shape
        return Shape(shape_type, x, y, width, height, color, texture, line_style)


def render_shapes(input_file: str, output_file: str) -> None:
    """
    Load all shapes from input file, create Shape objects for each,
    and render them to output file.
    
    This is intentionally memory-inefficient: stores all 1 million shape objects
    in memory simultaneously, each with their own copy of all properties.
    """
    try:
        # Read input file
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        shapes_data = data.get('shapes', [])
        
        # Create a Shape object for EVERY shape - all stored in memory simultaneously!
        all_shapes = []
        for shape_data in shapes_data:
            shape_obj = create_shape(shape_data)
            all_shapes.append(shape_obj)
        
        # Now render all shapes (they're all still in memory)
        rendered_results = []
        for idx, shape in enumerate(all_shapes):
            rendered_data = shape.render()
            rendered_results.append({
                "shape_id": idx,
                "render_data": rendered_data
            })
        
        # Write output
        output = {
            "rendered_shapes": rendered_results
        }
        
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Successfully rendered {len(all_shapes)} shapes")
        print(f"Output written to {output_file}")
        
    except FileNotFoundError as e:
        print(f"Error: Input file not found - {e}", file=sys.stderr)
        raise
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file - {e}", file=sys.stderr)
        raise
    except Exception as e:
        print(f"Error during rendering: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python shape_renderer.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    render_shapes(input_file, output_file)