import os
import re
import time
import json
import argparse
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional
import math

class RiskGrade(Enum):
    MINIMAL = "MINIMAL"
    MONITOR = "MONITOR"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"

class ExposureTier(Enum):
    TIER_1_SAFE = "TIER_1_SAFE"
    TIER_2_MONITOR = "TIER_2_MONITOR"
    TIER_3_WARNING = "TIER_3_WARNING"
    TIER_4_BLOCKED = "TIER_4_BLOCKED"

class SapActionFlag(Enum):
    PROCEED_NORMAL = "PROCEED_NORMAL"
    FLAG_FOR_REVIEW = "FLAG_FOR_REVIEW"
    FREEZE_PURCHASE_ORDERS = "FREEZE_PURCHASE_ORDERS"

@dataclass
class NoulResult:
    liquidity_distress: float
    debt_covenant_breach_risk: float
    growth_expansion_momentum: float
    capital_return_sustainable: float

@dataclass
class AdvisorResult:
    risk_grade: RiskGrade
    exposure_tier: ExposureTier
    sap_action_flag: SapActionFlag
    action_recommendations: List[str]

class LayaSystem1:
    def __init__(self, model_dir: str = "./output", use_onnx: bool = False):
        self.use_onnx = use_onnx
        if use_onnx:
            import onnxruntime as ort
            self.session = ort.InferenceSession(os.path.join(model_dir, "finbert.onnx"))
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        else:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        self.id2label = getattr(self.model.config, "id2label", None) if not use_onnx else None
        # Normalize mapping: find indices for negative, neutral, positive
        if self.id2label:
            self.label_to_idx = {v.lower(): int(k) for k, v in self.id2label.items()}
        else:
            self.label_to_idx = {"negative": 0, "neutral": 1, "positive": 2}
        self.idx_to_label = {v: k.upper() for k, v in self.label_to_idx.items()}

    def _get_logits(self, text: str):
        import torch
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        if self.use_onnx:
            ort_inputs = {k: v.cpu().numpy() for k, v in inputs.items()}
            logits = self.session.run(None, ort_inputs)[0]
            return torch.tensor(logits)
        else:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self.model(**inputs).logits
            return logits

    def choice(self, text: str, temperature: float = 1.25) -> str:
        import torch
        logits = self._get_logits(text)
        probs = torch.softmax(logits / temperature, dim=-1)
        idx = torch.argmax(probs, dim=-1).item()
        return self.idx_to_label.get(idx, f"CLASS_{idx}")

    def get_calibrated_probs(self, text: str, temperature: float = 1.25) -> dict:
        import torch
        logits = self._get_logits(text)
        probs = torch.softmax(logits / temperature, dim=-1).squeeze().tolist()
        neg_idx = self.label_to_idx.get("negative", 0)
        neu_idx = self.label_to_idx.get("neutral", 1)
        pos_idx = self.label_to_idx.get("positive", 2)
        return {
            "negative": probs[neg_idx],
            "neutral": probs[neu_idx],
            "positive": probs[pos_idx],
        }

    def score(self, text: str) -> float:
        p = self.get_calibrated_probs(text)
        # Distress index: 100 for negative, 15 for neutral, -35 for positive
        score_val = (p["negative"] * 100.0) + (p["neutral"] * 15.0) - (p["positive"] * 35.0)
        return max(0.0, min(100.0, score_val))

    def noul(self, text: str) -> NoulResult:
        p = self.get_calibrated_probs(text)
        return NoulResult(
            liquidity_distress=p["negative"] * 100.0,
            debt_covenant_breach_risk=(p["negative"] * 0.85 + p["neutral"] * 0.15) * 100.0,
            growth_expansion_momentum=p["positive"] * 100.0,
            capital_return_sustainable=(p["positive"] * 0.80 + p["neutral"] * 0.20) * 100.0,
        )

class AnchorTokenPruner:
    def __init__(self):
        self.pattern = re.compile(r'\b(percent|%|\$|revenue|profit|debt|covenant|margin|ebitda|impairment)\b', re.IGNORECASE)

    def prune(self, document: str) -> str:
        sentences = re.split(r'(?<=[.!?]) +', document)
        dense_sentences = [s for s in sentences if self.pattern.search(s)]
        return " ".join(dense_sentences)

class FinancialAdvisor:
    def __init__(self, system1: LayaSystem1, pruner: AnchorTokenPruner):
        self.system1 = system1
        self.pruner = pruner

    def advise(self, document: str) -> AdvisorResult:
        import time
        from rich.console import Console
        console = Console()
        
        t0 = time.time()
        pruned_text = self.pruner.prune(document)
        if not pruned_text:
            pruned_text = document
        t_prune = time.time() - t0
        
        t1 = time.time()
        choice = self.system1.choice(pruned_text)
        score = self.system1.score(pruned_text)
        noul = self.system1.noul(pruned_text)
        t_sys1 = time.time() - t1
        
        console.print(f"[dim]Layer 1 Pruning Time: {t_prune*1000:.2f}ms[/dim]")
        console.print(f"[dim]Layer 2 System-1 Time: {t_sys1*1000:.2f}ms[/dim]")
        
        # Asymmetric Risk Gate: 35% negative probability triggers WARNING
        p = self.system1.get_calibrated_probs(pruned_text)
        neg_prob = p["negative"]
        
        if score > 75 or neg_prob > 0.60:
            risk_grade = RiskGrade.CRITICAL
            exposure_tier = ExposureTier.TIER_4_BLOCKED
            sap_action_flag = SapActionFlag.FREEZE_PURCHASE_ORDERS
            recs = ["IMMEDIATE: Freeze uncommitted purchase orders and discretionary capex.",
                    "CREDIT: Require 100% upfront cash or irrevocable letters of credit.",
                    "AUDIT: Request immediate debt covenant compliance certificate."]
        elif neg_prob > 0.35 or score > 40:
            risk_grade = RiskGrade.WARNING
            exposure_tier = ExposureTier.TIER_3_WARNING
            sap_action_flag = SapActionFlag.FLAG_FOR_REVIEW
            recs = ["Review counterparty liquidity and cash burn rate", "Request updated covenant certificates"]
        elif score > 20:
            risk_grade = RiskGrade.MONITOR
            exposure_tier = ExposureTier.TIER_2_MONITOR
            sap_action_flag = SapActionFlag.PROCEED_NORMAL
            recs = ["Monitor upcoming quarterly performance filings"]
        else:
            risk_grade = RiskGrade.MINIMAL
            exposure_tier = ExposureTier.TIER_1_SAFE
            sap_action_flag = SapActionFlag.PROCEED_NORMAL
            recs = ["Counterparty healthy, proceed with standard commercial credit terms"]
            
        return AdvisorResult(risk_grade, exposure_tier, sap_action_flag, recs)

def run_tests():
    from rich.console import Console
    console = Console()
    
    test_cases = [
        "The company experienced a severe decline in liquidity and breached its debt covenant, although revenue showed a slight 2% growth.",
        "Organic ARR grew by 45% and EBITDA margin expanded significantly over the fiscal year.",
        "The board filed a standard 8-K regarding the appointment of a new independent director.",
        "Our primary supplier defaulted on their obligations, leading to $5M in inventory write-downs and margin contraction.",
        "Despite severe FX headwinds reducing international profits, domestic revenue grew robustly by 15%."
    ]
    
    try:
        system1 = LayaSystem1("ProsusAI/finbert")
    except Exception:
        console.print("[red]Could not load FinBERT. Ensure transformers and torch are installed.[/red]")
        return
        
    pruner = AnchorTokenPruner()
    advisor = FinancialAdvisor(system1, pruner)
    
    for i, tc in enumerate(test_cases, 1):
        console.print(f"\n[bold cyan]--- Test Case {i} ---[/bold cyan]")
        console.print(f"[italic]'{tc}'[/italic]")
        res = advisor.advise(tc)
        console.print(f"Risk Grade: [bold]{res.risk_grade.value}[/bold]")
        console.print(f"Exposure Tier: {res.exposure_tier.value}")
        console.print(f"SAP Action Flag: {res.sap_action_flag.value}")
        console.print(f"Recommendations: {', '.join(res.action_recommendations)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", action="store_true", help="Use ONNX runtime")
    parser.add_argument("--test", action="store_true", help="Run hardcoded test cases")
    args = parser.parse_args()
    
    if args.test:
        run_tests()
    else:
        print("Run with --test to execute test cases.")
