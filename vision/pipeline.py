import time
from typing import List
from .chart_detector import ChartDetector
from .chart_extractor import ChartExtractor
from .delta_calculator import parse_deplot_table, compute_deltas, format_concessive_sentence

class VisionPipeline:
    def __init__(self):
        self.detector = ChartDetector()
        self.extractor = ChartExtractor()
        
    def process_image(self, image_path: str) -> str:
        t0 = time.time()
        
        # 1. Detect
        det_result = self.detector.detect(image_path)
        if not det_result.is_chart:
            return ""
            
        # 2. Extract
        ext_result = self.extractor.extract(image_path)
        if not ext_result.raw_table:
            return ""
            
        # 3. Calculate & Format
        data = parse_deplot_table(ext_result.raw_table)
        deltas = compute_deltas(data)
        sentence = format_concessive_sentence(deltas)
        
        t_total = time.time() - t0
        print(f"[VisionPipeline] Processed image in {t_total*1000:.2f}ms")
        
        return sentence

    def process_document(self, text: str, images: List[str]) -> str:
        injections = []
        for img in images:
            sentence = self.process_image(img)
            if sentence:
                injections.append(sentence)
                
        if not injections:
            return text
            
        injection_text = " ".join(injections)
        return text + "\n\nChart Analysis: " + injection_text
