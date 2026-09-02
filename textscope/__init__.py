"""TextScope — local, private, calibratable prose analysis."""
from .stylometry import analyze_style, split_sentences, StyleReport
from .calibration import build_reference, z_scores, interpret, ReferenceStats
from .rewrite import rewrite_text, RewriteResult, render_rewrite_text

__version__ = "0.1.0"
__all__ = [
    "analyze_style", "split_sentences", "StyleReport",
    "build_reference", "z_scores", "interpret", "ReferenceStats",
    "rewrite_text", "RewriteResult", "render_rewrite_text",
]
