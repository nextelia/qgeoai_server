"""
HTML report generation module
"""

from .html_generator import HTMLReportGenerator
from .yolo_html_generator import YOLOHTMLReportGenerator

__all__ = [
    "HTMLReportGenerator",
    "YOLOHTMLReportGenerator"
]