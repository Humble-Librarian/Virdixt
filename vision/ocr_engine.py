"""
Virdixt High-Performance OCR Engine
Lightweight (<180MB RAM), ONNX-powered, air-gapped OCR engine with OpenCV image restoration
and geometric reading-order table reconstruction.
"""

import os
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
from PIL import Image

from vision.image_enhancer import DocumentImageEnhancer


class BaseOCREngine(ABC):
    """Abstract base class for OCR engines."""

    @abstractmethod
    def extract_text(self, image_input: Union[str, np.ndarray, Image.Image], enhance: bool = True) -> str:
        """Extracts structured text from an image input."""
        pass


class RapidOCREngine(BaseOCREngine):
    """Production-grade RapidOCR engine using ONNX Runtime.
    
    Features:
    - Lazy loading: Zero memory allocation until first OCR call.
    - OpenCV optical pre-enhancement for blurry / degraded scans.
    - Geometric line & column sorting to maintain table integrity.
    - Total memory footprint: ~150 MB (safe for 4 GB RAM systems).
    """

    def __init__(self, text_score_threshold: float = 0.5):
        self.text_score_threshold = text_score_threshold
        self._engine = None

    def _get_engine(self):
        """Lazy loader for RapidOCR instance."""
        if self._engine is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
                # Initialize ONNX runtime with standard CPU provider
                self._engine = RapidOCR()
            except ImportError:
                raise ImportError(
                    "rapidocr_onnxruntime is required for OCR. "
                    "Install it via: pip install rapidocr_onnxruntime"
                )
        return self._engine

    @staticmethod
    def _sort_bounding_boxes(ocr_results: List[List[Any]], line_grouping_tol: float = 0.5) -> List[Dict[str, Any]]:
        """Groups detected bounding boxes into natural reading order lines (top-down, left-to-right).
        
        Args:
            ocr_results: List of [dt_boxes, text, score]
            line_grouping_tol: Vertical overlap tolerance ratio
            
        Returns:
            Sorted list of dicts with text, bbox, and line_idx
        """
        if not ocr_results:
            return []

        parsed_boxes = []
        for item in ocr_results:
            bbox = item[0]  # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            text = str(item[1]).strip()
            score = float(item[2])

            pts = np.array(bbox)
            min_x = np.min(pts[:, 0])
            max_x = np.max(pts[:, 0])
            min_y = np.min(pts[:, 1])
            max_y = np.max(pts[:, 1])
            center_y = (min_y + max_y) / 2.0
            height = max(max_y - min_y, 1.0)

            parsed_boxes.append({
                "text": text,
                "score": score,
                "bbox": bbox,
                "min_x": min_x,
                "max_x": max_x,
                "min_y": min_y,
                "max_y": max_y,
                "center_y": center_y,
                "height": height
            })

        # Sort primarily by vertical center Y
        parsed_boxes.sort(key=lambda b: b["center_y"])

        # Group into lines based on vertical overlap
        lines: List[List[Dict[str, Any]]] = []
        for box in parsed_boxes:
            placed = False
            for line in lines:
                ref_box = line[0]
                # If vertical center is within tolerance of reference box height
                avg_height = (box["height"] + ref_box["height"]) / 2.0
                if abs(box["center_y"] - ref_box["center_y"]) < avg_height * line_grouping_tol:
                    line.append(box)
                    placed = True
                    break
            if not placed:
                lines.append([box])

        # Sort lines vertically from top to bottom
        lines.sort(key=lambda line: np.mean([b["center_y"] for b in line]))

        # Sort each line horizontally from left to right
        sorted_output = []
        for line_idx, line in enumerate(lines):
            line.sort(key=lambda b: b["min_x"])
            for b in line:
                b["line_idx"] = line_idx
                sorted_output.append(b)

        return sorted_output

    def extract_text_and_boxes(
        self, 
        image_input: Union[str, np.ndarray, Image.Image], 
        enhance: bool = True
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Extracts text along with structured bounding boxes."""
        engine = self._get_engine()

        # Step 1: Optical Image Restoration (OpenCV)
        if enhance:
            img = DocumentImageEnhancer.enhance_for_ocr(image_input)
        else:
            img = DocumentImageEnhancer.to_cv2_image(image_input)

        # Step 2: Run ONNX OCR Inference
        raw_result, _ = engine(img)

        if not raw_result:
            return "", []

        # Filter by confidence score
        valid_results = [r for r in raw_result if float(r[2]) >= self.text_score_threshold]
        if not valid_results:
            return "", []

        # Step 3: Geometric Reading-Order Sorting
        sorted_boxes = self._sort_bounding_boxes(valid_results)

        # Step 4: Reconstruct structured document text
        line_texts: Dict[int, List[str]] = {}
        for box in sorted_boxes:
            l_idx = box.get("line_idx", 0)
            if l_idx not in line_texts:
                line_texts[l_idx] = []
            line_texts[l_idx].append(box["text"])

        structured_lines = []
        for l_idx in sorted(line_texts.keys()):
            # Join cells on the same horizontal line with tab or space for table preservation
            structured_lines.append("   ".join(line_texts[l_idx]))

        final_text = "\n".join(structured_lines)
        return final_text, sorted_boxes

    def extract_text(self, image_input: Union[str, np.ndarray, Image.Image], enhance: bool = True) -> str:
        """Extracts clean structured text from image input."""
        text, _ = self.extract_text_and_boxes(image_input, enhance=enhance)
        return text


# Global Singleton for Zero-Tax Ingestion
_GLOBAL_OCR_ENGINE: Optional[RapidOCREngine] = None


def get_ocr_engine(text_score_threshold: float = 0.5) -> RapidOCREngine:
    """Returns the singleton RapidOCR engine instance."""
    global _GLOBAL_OCR_ENGINE
    if _GLOBAL_OCR_ENGINE is None:
        _GLOBAL_OCR_ENGINE = RapidOCREngine(text_score_threshold=text_score_threshold)
    return _GLOBAL_OCR_ENGINE


def ocr_document_page(image_input: Union[str, np.ndarray, Image.Image], enhance: bool = True) -> str:
    """Convenience functional helper for OCR extraction."""
    engine = get_ocr_engine()
    return engine.extract_text(image_input, enhance=enhance)
