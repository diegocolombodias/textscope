"""TextScope — local, private, calibratable prose analysis."""
from .stylometry import analyze_style, split_sentences, StyleReport
from .calibration import build_reference, z_scores, interpret, ReferenceStats

__version__ = "0.1.0"
__all__ = [
    "analyze_style", "split_sentences", "StyleReport",
    "build_reference", "z_scores", "interpret", "ReferenceStats",
]
