import os
import sys
import re
import time
import argparse
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional

# Enable UTF-8 console output for Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

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


from nlp import (
    ABSAEngine,
    AspectResult,
    LinguisticHedgingDetector,
    HedgingAnalysisResult,
    RhetoricalDiscourseParser,
    DiscourseAnalysisResult,
    ForensicAccountingEngine,
    ForensicScoreResult,
)


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
    absa_results: List[AspectResult] = field(default_factory=list)
    hedging_result: Optional[HedgingAnalysisResult] = None
    discourse_result: Optional[DiscourseAnalysisResult] = None
    forensic_scores: Optional[ForensicScoreResult] = None
    inference_time_ms: float = 0.0



class LayaSystem1:
    """Zero-overhead decision engine runtime with lazy-loaded ONNX/PyTorch backends."""

    def __init__(self, model_dir: str = "./output", use_onnx: Optional[bool] = None):
        # Auto-detect fastest runtime: prefer ONNX if models/ exists
        if use_onnx is None:
            if os.path.exists("models/finbert.onnx") or os.path.exists("models/finbert_quantized.onnx"):
                self.use_onnx = True
            else:
                self.use_onnx = False
        else:
            self.use_onnx = use_onnx

        self.model_dir = model_dir
        self.session = None
        self.model = None
        self.tokenizer = None
        self._init_engine()

    def _init_engine(self):
        t0 = time.time()
        if self.use_onnx:
            import onnxruntime as ort
            onnx_candidates = [
                os.path.join(self.model_dir, "finbert.onnx"),
                "models/finbert.onnx",
                "models/finbert_quantized.onnx"
            ]
            onnx_file = next((f for f in onnx_candidates if os.path.exists(f)), None)
            if not onnx_file:
                # Fallback to PyTorch
                self.use_onnx = False
                return self._init_engine()

            # Set thread count and fast graph optimization
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 4
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(onnx_file, sess_options)

            from transformers import AutoTokenizer
            tok_path = self.model_dir if os.path.exists(os.path.join(self.model_dir, "tokenizer.json")) else ("models" if os.path.exists("models/tokenizer.json") else "ProsusAI/finbert")
            is_local = os.path.exists(tok_path)
            self.tokenizer = AutoTokenizer.from_pretrained(tok_path, local_files_only=is_local)
            self.label_to_idx = {"negative": 0, "neutral": 1, "positive": 2}
            self.idx_to_label = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
        else:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            resolved_dir = self.model_dir if os.path.exists(self.model_dir) else ("models" if os.path.exists("models") else "ProsusAI/finbert")
            is_local = os.path.exists(resolved_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(resolved_dir, local_files_only=is_local).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(resolved_dir, local_files_only=is_local)
            id2label = getattr(self.model.config, "id2label", None)
            if id2label:
                self.label_to_idx = {v.lower(): int(k) for k, v in id2label.items()}
            else:
                self.label_to_idx = {"negative": 0, "neutral": 1, "positive": 2}
            self.idx_to_label = {v: k.upper() for k, v in self.label_to_idx.items()}

        t_load = (time.time() - t0) * 1000
        # console.print(f"[dim]Initialized LayaSystem1 ({'ONNX' if self.use_onnx else 'PyTorch'}) in {t_load:.1f}ms[/dim]")

    def _get_logits(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        if self.use_onnx:
            ort_inputs = {
                "input_ids": inputs["input_ids"].cpu().numpy(),
                "attention_mask": inputs["attention_mask"].cpu().numpy()
            }
            logits = self.session.run(None, ort_inputs)[0]
            return logits[0]
        else:
            import torch
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self.model(**inputs).logits
            return logits[0].cpu().numpy()

    def get_calibrated_probs(self, text: str, temperature: float = 1.25) -> dict:
        import numpy as np
        logits = self._get_logits(text)
        scaled = logits / temperature
        exp_logits = np.exp(scaled - np.max(scaled))
        probs = exp_logits / np.sum(exp_logits)

        neg_idx = self.label_to_idx.get("negative", 0)
        neu_idx = self.label_to_idx.get("neutral", 1)
        pos_idx = self.label_to_idx.get("positive", 2)
        return {
            "negative": float(probs[neg_idx]),
            "neutral": float(probs[neu_idx]),
            "positive": float(probs[pos_idx]),
        }

    def choice(self, text: str, temperature: float = 1.25) -> str:
        probs = self.get_calibrated_probs(text, temperature)
        sorted_labels = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        return sorted_labels[0][0].upper()

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
        self.pattern = re.compile(
            r'\b(percent|%|\$|revenue|profit|debt|covenant|margin|ebitda|impairment|loss|cash|default|downgrade|restructuring|expense|expenses|cogs|operating|distress|insolvency|adverse|opinion|freeze|liquidity|burn|growth|sales|income)\b',
            re.IGNORECASE
        )

    def prune(self, document: str) -> str:
        sentences = re.split(r'(?<=[.!?]) +|\n+', document)
        dense_sentences = [s.strip() for s in sentences if self.pattern.search(s) and len(s.strip()) > 5]
        return " ".join(dense_sentences) if dense_sentences else document



class FinancialAdvisor:
    def __init__(self, system1: LayaSystem1, pruner: AnchorTokenPruner):
        self.system1 = system1
        self.pruner = pruner
        self.absa_engine = ABSAEngine(system1)
        self.hedging_detector = LinguisticHedgingDetector()
        self.discourse_parser = RhetoricalDiscourseParser()
        self.forensic_engine = ForensicAccountingEngine()

    def advise(self, document: str, financial_dict: Optional[Dict[str, float]] = None) -> AdvisorResult:
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

        # Run Advanced NLP & Computational Linguistics Audits
        absa_results = self.absa_engine.evaluate(document)
        hedging_res = self.hedging_detector.analyze(document)
        discourse_res = self.discourse_parser.parse(document)
        forensic_res = self.forensic_engine.compute(financial_dict or {})

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

        total_time_ms = (t_prune + t_sys1) * 1000.0

        return AdvisorResult(
            risk_grade=risk_grade,
            exposure_tier=exposure_tier,
            action_flag=action_flag,
            action_recommendations=recs,
            sentiment_choice=choice_val,
            distress_score=score_val,
            noul=noul_val,
            calibrated_probs=probs,
            absa_results=absa_results,
            hedging_result=hedging_res,
            discourse_result=discourse_res,
            forensic_scores=forensic_res,
            inference_time_ms=total_time_ms
        )


def display_advisor_report(res: AdvisorResult, source_title: str = "Analysis Result"):
    color_map = {
        RiskGrade.CRITICAL: "bold red",
        RiskGrade.WARNING: "bold yellow",
        RiskGrade.MONITOR: "bold cyan",
        RiskGrade.MINIMAL: "bold green"
    }
    grade_color = color_map.get(res.risk_grade, "white")

    # 1. Primary Executive Verdict Table
    table = Table(title=f"=== VIRDIXT FINANCIAL ADVISOR & ERP AUDIT: {source_title} ===", style="blue")
    table.add_column("Decision Dimension", style="cyan", width=28)
    table.add_column("Verdict / Calibrated Value", style="white")

    table.add_row("Sentiment Choice", f"[{'red' if res.sentiment_choice=='NEGATIVE' else 'green' if res.sentiment_choice=='POSITIVE' else 'yellow'}]{res.sentiment_choice}[/]")
    table.add_row("Distress Index Score", f"{res.distress_score:.1f} / 100.0 (0=Peak Health, 100=Insolvency)")
    table.add_row("Calibrated Probabilities", f"Neg: {res.calibrated_probs['negative']*100:.1f}% | Neu: {res.calibrated_probs['neutral']*100:.1f}% | Pos: {res.calibrated_probs['positive']*100:.1f}%")
    table.add_row("Risk Grade", f"[{grade_color}]{res.risk_grade.value}[/{grade_color}]")
    table.add_row("Exposure Tier", f"[{grade_color}]{res.exposure_tier.value}[/{grade_color}]")
    table.add_row("ERP Policy Action", f"[{grade_color}]{res.action_flag.value}[/{grade_color}]")
    table.add_row("Inference Latency", f"[bold green]{res.inference_time_ms:.2f} ms[/]")

    noul_str = (
        f"- Liquidity Distress Risk   : {res.noul.liquidity_distress:.1f}%\n"
        f"- Debt Covenant Breach Risk : {res.noul.debt_covenant_breach_risk:.1f}%\n"
        f"- Growth Expansion Momentum : {res.noul.growth_expansion_momentum:.1f}%\n"
        f"- Sustainable Capital Return: {res.noul.capital_return_sustainable:.1f}%"
    )
    table.add_row("NOUL Hypotheses P(True)", noul_str)

    console.print()
    console.print(table)

    # 2. Aspect-Based Financial Sentiment Table (ABSA)
    if res.absa_results:
        absa_table = Table(title="🔍 Aspect-Based Sentiment & Multi-Entity Risk (ABSA)", style="magenta")
        absa_table.add_column("Financial Aspect", style="bold cyan", width=26)
        absa_table.add_column("Sentiment", style="white", width=16)
        absa_table.add_column("Distress %", style="white", width=12)
        absa_table.add_column("Risk Level", style="white", width=14)
        absa_table.add_column("Key Grounding Evidence", style="dim white")

        for a in res.absa_results:
            if not a.detected:
                absa_table.add_row(a.aspect_name, "[dim]NOT REPORTED[/dim]", "-", "[dim]NEUTRAL[/dim]", "[dim]No specific commentary[/dim]")
                continue
            s_color = "red" if a.sentiment == "NEGATIVE" else "green" if a.sentiment == "POSITIVE" else "yellow"
            r_color = "red" if a.risk_level == "CRITICAL" else "yellow" if a.risk_level == "WARNING" else "green"
            evidence = a.key_sentences[0] if a.key_sentences else ""
            if len(evidence) > 65:
                evidence = evidence[:62] + "..."
            absa_table.add_row(
                a.aspect_name,
                f"[{s_color}]{a.sentiment}[/{s_color}]",
                f"{a.distress_score:.1f}%",
                f"[{r_color}]{a.risk_level}[/{r_color}]",
                evidence
            )
        console.print(absa_table)

    # 3. Linguistic Deception & Hedging Audit Panel
    if res.hedging_result and (res.hedging_result.hedging_score > 0 or res.hedging_result.gunning_fog_index > 0):
        h = res.hedging_result
        h_color = "red" if h.hedging_level in ["EXTREME_EVASION", "HIGH_UNCERTAINTY"] else "yellow" if h.hedging_level == "MODERATE" else "green"
        euphemism_str = ", ".join([f"'{e}'" for e in h.detected_euphemisms]) if h.detected_euphemisms else "None detected"
        hedges_str = ", ".join([f"'{m}'" for m in h.detected_hedges]) if h.detected_hedges else "None detected"
        
        hedge_panel = Panel(
            f"• [bold]Epistemic Uncertainty Level:[/] [{h_color}]{h.hedging_level}[/{h_color}] (Score: {h.hedging_score:.1f}/100)\n"
            f"• [bold]Passive Voice Evasion Score:[/] {h.passive_evasion_score:.1f}/100\n"
            f"• [bold]Gunning-Fog Obfuscation Index:[/] Grade {h.gunning_fog_index:.1f} ({h.obfuscation_risk})\n"
            f"• [bold]Flagged Corporate Euphemisms:[/] [yellow]{euphemism_str}[/yellow]\n"
            f"• [bold]Detected Modal Hedges:[/] [cyan]{hedges_str}[/cyan]",
            title="🗣️ Linguistic Deception & Executive Hedging Audit",
            style="yellow"
        )
        console.print(hedge_panel)

    # 4. Rhetorical Discourse & Concessive Nucleus Breakdown
    if res.discourse_result and res.discourse_result.has_concessive_structures:
        d = res.discourse_result
        deceptive_pairs = [p for p in d.pairs if p.is_deceptive_buffer]
        disc_text = []
        if deceptive_pairs:
            disc_text.append("[bold red][!] Rhetorical Masking Detected:[/] Superficial positive buffer clauses detected masking core distress nuclei:")
            for p in deceptive_pairs[:2]:
                disc_text.append(f"  • [green]Satellite (Buffer):[/] \"{p.satellite_clause}\"")
                disc_text.append(f"    ↳ [bold red]Nucleus (Core Reality):[/] \"{p.nucleus_clause}\"")
        else:
            disc_text.append(f"Parsed {d.total_concessive_sentences} concessive rhetorical structure(s). Core discourse nuclei aligned with accounting deltas.")

        disc_panel = Panel("\n".join(disc_text), title="⚖️ Rhetorical Structure Theory (RST) Discourse Analysis", style="cyan")
        console.print(disc_panel)

    # 5. Deterministic Forensic Accounting Suite (Altman Z + Beneish M + Piotroski F)
    if res.forensic_scores and res.forensic_scores.calculated:
        f = res.forensic_scores
        z_color = "red" if "DISTRESS" in f.altman_zone else "yellow" if "GREY" in f.altman_zone else "green"
        m_color = "red" if "HIGH" in f.beneish_manipulation_risk else "green"
        f_color = "green" if "STRONG" in f.piotroski_grade else "yellow" if "MODERATE" in f.piotroski_grade else "red"

        forensic_table = Table(title="📐 Deterministic Forensic Accounting Benchmarks (0ms ML Overhead)", style="green")
        forensic_table.add_column("Forensic Model", style="cyan", width=26)
        forensic_table.add_column("Calculated Score", style="white", width=18)
        forensic_table.add_column("Institutional Risk Interpretation", style="white")

        forensic_table.add_row("Altman Z-Score", f"{f.altman_z_score:.2f}" if f.altman_z_score is not None else "N/A", f"[{z_color}]{f.altman_zone}[/{z_color}]")
        forensic_table.add_row("Beneish M-Score", f"{f.beneish_m_score:.2f}" if f.beneish_m_score is not None else "N/A", f"[{m_color}]{f.beneish_manipulation_risk}[/{m_color}]")
        forensic_table.add_row("Piotroski F-Score", f"{f.piotroski_f_score} / 9" if f.piotroski_f_score is not None else "N/A", f"[{f_color}]{f.piotroski_grade}[/{f_color}]")
        console.print(forensic_table)

    # 6. Action Recommendations
    recs_str = "\n".join([f"• {r}" for r in res.action_recommendations])
    rec_panel = Panel(recs_str, title="🛡️ ERP Policy Directives & Commercial Actions", style=grade_color)
    console.print(rec_panel)
    console.print()


def process_file_input(file_path: str, advisor: FinancialAdvisor, deep_vision: bool = False):
    """Universal pipeline for PDF, DOCX, and TXT files with adaptive visual extraction."""
    from document_parser import DocumentParser

    t0 = time.time()
    parsed = DocumentParser.parse(file_path, extract_images=True)
    t_parse = (time.time() - t0) * 1000

    console.print(Panel(
        f"[bold green]Ingesting Document:[/] [cyan]{file_path}[/]\n"
        f"Format: [bold]{parsed.file_type}[/] | Pages: [bold]{parsed.page_count}[/] | Visual Assets: [bold]{len(parsed.image_paths)}[/] | Parse Time: [dim]{t_parse:.1f}ms[/dim]",
        title="Universal Ingestion Pipeline"
    ))

    final_text = parsed.raw_text

    if parsed.has_visuals:
        console.print(f"[bold yellow][!] {len(parsed.image_paths)} Embedded Visuals Detected: Processing via Visual Delta Pipeline...[/]")
        from vision.pipeline import VisionPipeline
        vision = VisionPipeline(deep_vision=deep_vision)
        final_text = vision.process_document(parsed.raw_text, parsed.image_paths)
    else:
        console.print("[dim][+] Pure text report: Vision pipeline bypassed (0ms visual overhead).[/dim]")

    result = advisor.advise(final_text, financial_dict=getattr(parsed, "financial_dict", None))
    display_advisor_report(result, source_title=os.path.basename(file_path))
    parsed.cleanup()
    return result


def run_interactive_mode(advisor: FinancialAdvisor, deep_vision: bool = False):
    """Warm interactive REPL loop for processing multiple files or text prompts in sub-10ms."""
    console.print(Panel(
        "[bold cyan]VIRDIXT INTERACTIVE ENGINE (WARM SESSION)[/bold cyan]\n"
        "Model is warm and resident in RAM. Instant sub-10ms evaluation.\n"
        "Type a [bold green]file path[/] (.pdf, .docx, .txt) or [bold green]raw text[/].\n"
        "Type [bold yellow]'exit'[/] or [bold yellow]'q'[/] to quit.",
        title="Interactive Mode"
    ))

    while True:
        try:
            user_input = input("\nvirdixt > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                console.print("[dim]Exiting interactive session. Goodbye![/dim]")
                break

            # Check if input is a file path
            clean_path = user_input.strip('"').strip("'")
            if os.path.exists(clean_path):
                process_file_input(clean_path, advisor, deep_vision=deep_vision)
            else:
                # Direct text evaluation
                res = advisor.advise(user_input)
                display_advisor_report(res, source_title="Interactive Prompt")
        except KeyboardInterrupt:
            console.print("\n[dim]Session terminated.[/dim]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


def run_batch_directory(dir_path: str, advisor: FinancialAdvisor, deep_vision: bool = False):
    """Scans and evaluates an entire directory of documents in a single warm session."""
    if not os.path.exists(dir_path):
        console.print(f"[red]Directory not found: {dir_path}[/red]")
        return

    valid_exts = {".pdf", ".docx", ".doc", ".csv", ".tsv", ".xlsx", ".xls", ".txt", ".md"}
    files = [os.path.join(dir_path, f) for f in sorted(os.listdir(dir_path)) if os.path.splitext(f)[1].lower() in valid_exts]

    if not files:
        console.print(f"[yellow]No supported documents found in {dir_path}[/yellow]")
        return

    console.print(f"\n[bold green]Found {len(files)} documents in {dir_path}. Running batch audit...[/bold green]")
    t_start = time.time()
    
    summary_table = Table(title=f"=== BATCH AUDIT SUMMARY: {dir_path} ===", style="green")
    summary_table.add_column("Filename", style="cyan")
    summary_table.add_column("Choice", style="white")
    summary_table.add_column("Distress Score", style="white")
    summary_table.add_column("Risk Grade", style="white")
    summary_table.add_column("Policy Action", style="white")
    summary_table.add_column("Latency", style="white")

    for fpath in files:
        from document_parser import DocumentParser
        parsed = DocumentParser.parse(fpath, extract_images=True)
        final_text = parsed.raw_text
        if parsed.has_visuals:
            from vision.pipeline import VisionPipeline
            vision = VisionPipeline(deep_vision=deep_vision)
            final_text = vision.process_document(parsed.raw_text, parsed.image_paths)
        res = advisor.advise(final_text)
        parsed.cleanup()

        color = "red" if res.risk_grade == RiskGrade.CRITICAL else "yellow" if res.risk_grade == RiskGrade.WARNING else "green"
        summary_table.add_row(
            os.path.basename(fpath),
            res.sentiment_choice,
            f"{res.distress_score:.1f}/100",
            f"[{color}]{res.risk_grade.value}[/{color}]",
            f"[{color}]{res.action_flag.value}[/{color}]",
            f"{res.inference_time_ms:.1f}ms"
        )

    t_total = time.time() - t_start
    console.print()
    console.print(summary_table)
    console.print(f"\n[bold green]Batch complete:[/] Processed {len(files)} files in {t_total*1000:.1f}ms ({t_total/len(files)*1000:.1f}ms / file average).\n")


def run_tests(advisor: FinancialAdvisor):
    test_cases = [
        "The company experienced a severe decline in liquidity and breached its debt covenant, although revenue showed a slight 2% growth.",
        "Organic ARR grew by 45% and EBITDA margin expanded significantly over the fiscal year.",
        "The board filed a standard 8-K regarding the appointment of a new independent director.",
        "Our primary supplier defaulted on their obligations, leading to $5M in inventory write-downs and margin contraction.",
        "Despite severe FX headwinds reducing international profits, domestic revenue grew robustly by 15%."
    ]

    for i, tc in enumerate(test_cases, 1):
        console.print(f"\n[bold cyan]--- Scenario {i} ---[/bold cyan]")
        console.print(f"[italic]'{tc}'[/italic]")
        res = advisor.advise(tc)
        display_advisor_report(res, source_title=f"Scenario {i}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Virdixt High-Performance Financial Sentiment & Decision Engine")
    parser.add_argument("--file", type=str, default="", help="Path to PDF, DOCX, or TXT financial report")
    parser.add_argument("--text", type=str, default="", help="Direct text input to evaluate")
    parser.add_argument("--batch", type=str, default="", help="Directory path to batch-process all documents")
    parser.add_argument("--interactive", "-i", action="store_true", help="Launch interactive warm REPL session")
    parser.add_argument("--onnx", action="store_true", help="Force ONNX Runtime (default: auto-detect)")
    parser.add_argument("--pytorch", action="store_true", help="Force PyTorch backend")
    parser.add_argument("--deep-vision", action="store_true", help="Enable deep Google DePlot token autoregression")
    parser.add_argument("--test", action="store_true", help="Run validation scenario benchmarks")
    args = parser.parse_args()

    # Determine backend: ONNX by default if available unless --pytorch is set
    use_onnx = False if args.pytorch else (True if args.onnx or os.path.exists("models/finbert.onnx") else False)

    system1 = LayaSystem1(model_dir="./output", use_onnx=use_onnx)
    pruner = AnchorTokenPruner()
    advisor = FinancialAdvisor(system1, pruner)

    if args.interactive:
        run_interactive_mode(advisor, deep_vision=args.deep_vision)
    elif args.batch:
        run_batch_directory(args.batch, advisor, deep_vision=args.deep_vision)
    elif args.file:
        process_file_input(args.file, advisor, deep_vision=args.deep_vision)
    elif args.text:
        res = advisor.advise(args.text)
        display_advisor_report(res, source_title="Direct Text Input")
    elif args.test:
        run_tests(advisor)
    else:
        console.print("[yellow]Usage options:[/yellow]")
        console.print("  python infer.py --file <report.pdf / report.docx / report.txt>")
        console.print("  python infer.py --batch data/sample_reports/")
        console.print("  python infer.py --interactive   (Sub-10ms warm REPL)")
        console.print("  python infer.py --text 'Financial commentary...'")
        console.print("  python infer.py --test")
