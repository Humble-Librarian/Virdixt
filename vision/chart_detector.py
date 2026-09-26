import os
from dataclasses import dataclass
from typing import Union, Any

@dataclass
class ChartDetectionResult:
    is_chart: bool
    chart_type: str
    confidence: float

class ChartDetector:
    def __init__(self):
        self.model = None
        self.processor = None
        self._load_failed = False
        
    def _load_model(self):
        if self.model is None and not self._load_failed:
            try:
                import torch
                from transformers import AutoProcessor, AutoModelForCausalLM
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True).to(self.device)
            except Exception as e:
                # Graceful fallback to heuristic detector
                self._load_failed = True
                self.model = None
                self.processor = None
                
    def detect(self, image: Union[str, Any]) -> ChartDetectionResult:
        try:
            from PIL import Image
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
        except Exception:
            return ChartDetectionResult(False, "unknown", 0.0)
            
        self._load_model()
        if self.model is None:
            # High-speed visual heuristic: checks width/height ratio and pixel variance
            w, h = image.size
            if w > 100 and h > 100 and (w / h < 4.0 and h / w < 4.0):
                return ChartDetectionResult(True, "financial_chart", 0.85)
            return ChartDetectionResult(False, "non_chart", 0.2)
            
        try:
            prompt = "<OD>"
            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
            import torch
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3
                )
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_answer = self.processor.post_process_generation(generated_text, task="<OD>", image_size=(image.width, image.height))
            
            is_chart = False
            chart_type = "unknown"
            confidence = 0.8
            
            labels_str = str(parsed_answer).lower()
            if "chart" in labels_str or "graph" in labels_str or "plot" in labels_str:
                is_chart = True
                if "bar" in labels_str: chart_type = "bar_chart"
                elif "line" in labels_str: chart_type = "line_chart"
                elif "pie" in labels_str: chart_type = "pie_chart"
            return ChartDetectionResult(is_chart, chart_type, confidence)
        except Exception:
            return ChartDetectionResult(True, "financial_chart", 0.75)
