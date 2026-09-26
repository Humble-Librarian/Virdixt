import os
import sys
import re
import time
import json
import argparse

# Enable UTF-8 console output for Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional
import math
from document_parser import DocumentParser, ParsedDocument
from vision.pipeline import VisionPipeline
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console(force_terminal=True, legacy_windows=False)

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

class PolicyActionFlag(Enum):
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
    action_flag: PolicyActionFlag
    action_recommendations: List[str]
    sentiment_choice: str
    distress_score: float
    noul: NoulResult
    calibrated_probs: Dict[str, float]

class LayaSystem1:
    def __init__(self, model_dir: str = "./output", use_onnx: bool = False):
        self.use_onnx = use_onnx
        if use_onnx:
            import onnxruntime as ort
            onnx_file = os.path.join(model_dir, "finbert.onnx") if not model_dir.endswith(".onnx") else model_dir
            if not os.path.exists(onnx_file):
                if os.path.exists("models/finbert.onnx"):
                    onnx_file = "models/finbert.onnx"
                elif os.path.exists("models/finbert_quantized.onnx"):
                    onnx_file = "models/finbert_quantized.onnx"
            self.session = ort.InferenceSession(onnx_file)
            from transformers import AutoTokenizer
            tok_path = model_dir if os.path.exists(os.path.join(model_dir, "tokenizer.json")) else "ProsusAI/finbert"
            self.tokenizer = AutoTokenizer.from_pretrained(tok_path)
        else:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            resolved_dir = model_dir if os.path.exists(model_dir) else "ProsusAI/finbert"
            self.model = AutoModelForSequenceClassification.from_pretrained(resolved_dir).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(resolved_dir)

        self.id2label = getattr(self.model.config, "id2label", None) if not use_onnx else None
        if self.id2label:
            self.label_to_idx = {v.lower(): int(k) for k, v in self.id2label.items()}
        else:
            self.label_to_idx = {"negative": 0, "neutral": 1, "positive": 2}
        self.idx_to_label = {v: k.upper() for k, v in self.label_to_idx.items()}

    def _get_logits(self, text: str):
        import torch
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        if self.use_onnx:
            ort_inputs = {
                "input_ids": inputs["input_ids"].cpu().numpy(),
                "attention_mask": inputs["attention_mask"].cpu().numpy()
            }
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
        self.pattern = re.compile(r'\b(percent|%|\$|revenue|profit|debt|covenant|margin|ebitda|impairment|loss|cash|default|downgrade|restructuring)\b', re.IGNORECASE)

    def prune(self, document: str) -> str:
        sentences = re.split(r'(?<=[.!?]) +|\n+', document)
        dense_sentences = [s.strip() for s in sentences if self.pattern.search(s) and len(s.strip()) > 5]
        return " ".join(dense_sentences)

class FinancialAdvisor:
    def __init__(self, system1: LayaSystem1, pruner: AnchorTokenPruner):
        self.system1 = system1
        self.pruner = pruner

    def advise(self, document: str) -> AdvisorResult:
        t0 = time.time()
        pruned_text = self.pruner.prune(document)
        if not pruned_text:
            pruned_text = document
        t_prune = time.time() - t0
        
        t1 = time.time()
        choice_val = self.system1.choice(pruned_text)
        score_val = self.system1.score(pruned_text)
        noul_val = self.system1.noul(pruned_text)
        probs = self.system1.get_calibrated_probs(pruned_text)
        t_sys1 = time.time() - t1
        
        console.print(f"[dim]• Layer 1 Anchor Pruner: {t_prune*1000:.2f}ms (Compressed to {len(pruned_text)} chars)[/dim]")
        console.print(f"[dim]• Layer 2 Laya System-1: {t_sys1*1000:.2f}ms ({'ONNX Runtime' if self.system1.use_onnx else 'PyTorch'})[/dim]")
        
        neg_prob = probs["negative"]
        
        if score_val > 75 or neg_prob > 0.60:
            risk_grade = RiskGrade.CRITICAL
            exposure_tier = ExposureTier.TIER_4_BLOCKED
            action_flag = PolicyActionFlag.FREEZE_PURCHASE_ORDERS
            recs = [
                "IMMEDIATE: Freeze uncommitted purchase orders and discretionary capex.",
                "CREDIT: Require 100% upfront cash or irrevocable letters of credit.",
                "AUDIT: Request immediate debt covenant compliance certificate."
            ]
        elif neg_prob > 0.35 or score_val > 40:
            risk_grade = RiskGrade.WARNING
            exposure_tier = ExposureTier.TIER_3_WARNING
            action_flag = PolicyActionFlag.FLAG_FOR_REVIEW
            recs = [
                "Review counterparty liquidity buffer and 90-day cash burn rate.",
                "Request updated debt covenant compliance certificates from lenders.",
                "Cap maximum single transaction exposure to $25,000."
            ]
        elif score_val > 20:
            risk_grade = RiskGrade.MONITOR
            exposure_tier = ExposureTier.TIER_2_MONITOR
            action_flag = PolicyActionFlag.PROCEED_NORMAL
            recs = [
                "Maintain standard monitoring on upcoming quarterly 10-Q filing.",
                "Record counterparty risk review note in CRM."
            ]
        else:
            risk_grade = RiskGrade.MINIMAL
            exposure_tier = ExposureTier.TIER_1_SAFE
            action_flag = PolicyActionFlag.PROCEED_NORMAL
            recs = [
                "Counterparty balance sheet healthy; proceed with standard commercial credit terms.",
                "Approved for preferred volume discount and standard Net-30/Net-60 terms."
            ]
            
        return AdvisorResult(
            risk_grade=risk_grade,
            exposure_tier=exposure_tier,
            action_flag=action_flag,
            action_recommendations=recs,
            sentiment_choice=choice_val,
            distress_score=score_val,
            noul=noul_val,
            calibrated_probs=probs
        )


def display_advisor_report(res: AdvisorResult, source_title: str = "Analysis Result"):
    """Renders a structured enterprise audit report in the console."""
    color_map = {
        RiskGrade.CRITICAL: "bold red",
        RiskGrade.WARNING: "bold yellow",
        RiskGrade.MONITOR: "bold cyan",
        RiskGrade.MINIMAL: "bold green"
    }
    grade_color = color_map.get(res.risk_grade, "white")
    
    table = Table(title=f"=== VIRDIXT FINANCIAL ADVISOR & ERP AUDIT: {source_title} ===", style="blue")
    table.add_column("Decision Dimension", style="cyan", width=26)
    table.add_column("Verdict / Calibrated Value", style="white")

    table.add_row("Sentiment Choice", f"[{'red' if res.sentiment_choice=='NEGATIVE' else 'green' if res.sentiment_choice=='POSITIVE' else 'yellow'}]{res.sentiment_choice}[/]")
    table.add_row("Distress Index Score", f"{res.distress_score:.1f} / 100.0 (0=Peak Health, 100=Insolvency)")
    table.add_row("Calibrated Probabilities", f"Neg: {res.calibrated_probs['negative']*100:.1f}% | Neu: {res.calibrated_probs['neutral']*100:.1f}% | Pos: {res.calibrated_probs['positive']*100:.1f}%")
    table.add_row("Risk Grade", f"[{grade_color}]{res.risk_grade.value}[/{grade_color}]")
    table.add_row("Exposure Tier", f"[{grade_color}]{res.exposure_tier.value}[/{grade_color}]")
    table.add_row("ERP Policy Action", f"[{grade_color}]{res.action_flag.value}[/{grade_color}]")
    
    noul_str = (
        f"- Liquidity Distress Risk   : {res.noul.liquidity_distress:.1f}%\n"
        f"- Debt Covenant Breach Risk : {res.noul.debt_covenant_breach_risk:.1f}%\n"
        f"- Growth Expansion Momentum : {res.noul.growth_expansion_momentum:.1f}%\n"
        f"- Sustainable Capital Return: {res.noul.capital_return_sustainable:.1f}%"
    )
    table.add_row("NOUL Hypotheses P(True)", noul_str)
    
    recs_str = "\n".join([f"- {r}" for r in res.action_recommendations])
    table.add_row("Action Directives", recs_str)
    
    console.print()
    console.print(table)
    console.print()


def process_file_input(file_path: str, use_onnx: bool = False):
    """Universal pipeline for PDF, DOCX, and TXT files with auto visual extraction."""
    console.print(Panel(f"[bold green]Ingesting Document:[/] [cyan]{file_path}[/]", title="Universal Ingestion Pipeline"))
    
    parsed = DocumentParser.parse(file_path, extract_images=True)
    console.print(f"[bold]Detected Format:[/] {parsed.file_type} | [bold]Pages:[/] {parsed.page_count} | [bold]Extracted Images/Charts:[/] {len(parsed.image_paths)}")
    
    final_text = parsed.raw_text
    
    # Run Multimodal Vision Pipeline if visuals are detected
    if parsed.has_visuals:
        console.print("[bold yellow][!] Embedded Visuals Detected: Routing through Florence-2 + DePlot Vision Pipeline...[/]")
        vision = VisionPipeline()
        final_text = vision.process_document(parsed.raw_text, parsed.image_paths)
    else:
        console.print("[dim][+] Pure text report: Vision pipeline bypassed (0ms visual overhead).[/dim]")

    model_path = "./output" if not use_onnx else "./models"
    try:
        system1 = LayaSystem1(model_dir=model_path, use_onnx=use_onnx)
    except Exception as e:
        console.print(f"[yellow]Loading fallback model ProsusAI/finbert ({e})...[/yellow]")
        system1 = LayaSystem1("ProsusAI/finbert", use_onnx=False)
        
    pruner = AnchorTokenPruner()
    advisor = FinancialAdvisor(system1, pruner)
    
    result = advisor.advise(final_text)
    display_advisor_report(result, source_title=os.path.basename(file_path))
    parsed.cleanup()


def run_tests(use_onnx: bool = False):
    test_cases = [
        "The company experienced a severe decline in liquidity and breached its debt covenant, although revenue showed a slight 2% growth.",
        "Organic ARR grew by 45% and EBITDA margin expanded significantly over the fiscal year.",
        "The board filed a standard 8-K regarding the appointment of a new independent director.",
        "Our primary supplier defaulted on their obligations, leading to $5M in inventory write-downs and margin contraction.",
        "Despite severe FX headwinds reducing international profits, domestic revenue grew robustly by 15%."
    ]
    
    try:
        model_path = "./output" if not use_onnx else "./models"
        system1 = LayaSystem1(model_dir=model_path, use_onnx=use_onnx)
    except Exception as e:
        system1 = LayaSystem1("ProsusAI/finbert", use_onnx=False)
        
    pruner = AnchorTokenPruner()
    advisor = FinancialAdvisor(system1, pruner)
    
    for i, tc in enumerate(test_cases, 1):
        console.print(f"\n[bold cyan]--- Scenario {i} ---[/bold cyan]")
        console.print(f"[italic]'{tc}'[/italic]")
        res = advisor.advise(tc)
        display_advisor_report(res, source_title=f"Scenario {i}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Virdixt Multimodal Financial Sentiment & Laya Decision Engine")
    parser.add_argument("--file", type=str, default="", help="Path to PDF, DOCX, or TXT financial report")
    parser.add_argument("--text", type=str, default="", help="Direct text input to evaluate")
    parser.add_argument("--onnx", action="store_true", help="Execute via ONNX Runtime")
    parser.add_argument("--test", action="store_true", help="Run validation scenario benchmarks")
    args = parser.parse_args()
    
    if args.file:
        process_file_input(args.file, use_onnx=args.onnx)
    elif args.text:
        model_path = "./output" if not args.onnx else "./models"
        system1 = LayaSystem1(model_dir=model_path, use_onnx=args.onnx)
        pruner = AnchorTokenPruner()
        advisor = FinancialAdvisor(system1, pruner)
        res = advisor.advise(args.text)
        display_advisor_report(res, source_title="Direct Text Input")
    elif args.test:
        run_tests(use_onnx=args.onnx)
    else:
        console.print("[yellow]Usage options:[/yellow]")
        console.print("  python infer.py --file <report.pdf / report.docx / report.txt> [--onnx]")
        console.print("  python infer.py --text 'Financial commentary text...' [--onnx]")
        console.print("  python infer.py --test [--onnx]")
