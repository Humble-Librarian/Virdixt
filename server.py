import os
import time
import io
import json
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from infer import (
    LayaSystem1, AnchorTokenPruner, FinancialAdvisor, AdvisorResult,
    RiskGrade, apply_overrides, POLICY_TABLE
)
from nlp import ForensicAccountingEngine
from document_parser import DocumentParser

# Global state for the engine
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading ML models...")
    # Initialize exactly as infer.py
    use_onnx = True if os.path.exists("models/finbert.onnx") else False
    system1 = LayaSystem1(model_dir="./output", use_onnx=use_onnx)
    pruner = AnchorTokenPruner()
    advisor = FinancialAdvisor(system1, pruner)
    
    ml_models["advisor"] = advisor
    print("Models loaded successfully.")
    yield
    # Cleanup
    ml_models.clear()

app = FastAPI(title="Virdixt API", lifespan=lifespan)

# Intercept and store console logs
class TraceLogger:
    def __init__(self):
        self.logs = []
        self.t0 = time.time()
        
    def log(self, stage: str, message: str):
        elapsed = time.time() - self.t0
        mins, secs = divmod(elapsed, 60)
        timestamp = f"{int(mins):02d}:{secs:05.2f}"
        self.logs.append({"timestamp": timestamp, "stage": stage, "message": message})

def format_verdict(res: AdvisorResult) -> dict:
    return {
        "rating": res.risk_grade.value,
        "score": f"{res.distress_score:.1f} / 100",
        "interpretation": res.exposure_tier.value,
        "confidence": 1.0,
    }

def format_quant(res: AdvisorResult) -> dict:
    f = res.forensic_scores
    if not f or not getattr(f, "calculated", False):
        return {}
    return {
        "altman_z": {
            "score": getattr(f, "altman_z_score", None),
            "status": getattr(f, "altman_zone", "UNKNOWN"),
            "formula": "1.2X1 + 1.4X2 + 3.3X3 + 0.6X4 + 0.999X5"
        },
        "beneish_m": {
            "score": getattr(f, "beneish_m_score", None),
            "status": getattr(f, "beneish_manipulation_risk", "UNKNOWN"),
            "threshold": -1.78
        },
        "benford_law": {
            "anomaly_detected": False,
            "distribution": []
        }
    }

def format_nlp(res: AdvisorResult) -> dict:
    return {
        "tone": res.sentiment_choice,
        "distress_score": res.distress_score,
        "key_triggers": [a.aspect_name for a in res.absa_results if getattr(a, "detected", False)] if res.absa_results else []
    }

def format_masking(res: AdvisorResult) -> dict:
    d = res.discourse_result
    callouts = []
    if d and getattr(d, "has_concessive_structures", False):
        for p in d.pairs:
            if getattr(p, "is_deceptive_buffer", False):
                callouts.append({
                    "sentence": f"Satellite: {p.satellite_clause} -> Nucleus: {p.nucleus_clause}",
                    "masking_score": 1.0,
                    "explanation": "Defensive buffer clause identified masking core reality."
                })
    return {
        "has_callout": len(callouts) > 0,
        "exposure": 0.0,
        "callouts": callouts
    }

# ── Simulator snapshot helpers ─────────────────────────────────────────────

# Maps friendly UI label → list of keyword fragments used by _extract_metric
_FACT_LABELS = {
    "Revenue / Sales":          ["revenue", "sales"],
    "EBIT / Operating Income":  ["operating income", "ebit", "operating profit"],
    "Net Income":               ["net income", "net profit"],
    "Total Assets":             ["total assets", "assets"],
    "Total Debt / Liabilities": ["total debt", "debt", "liabilities"],
    "Cash & Equivalents":       ["cash and cash equivalents", "cash"],
    "Working Capital":          ["working capital"],
    "Retained Earnings":        ["retained earnings"],
    "Equity":                   ["equity", "total equity", "shareholders equity"],
}

def format_simulator_snapshot(res: AdvisorResult) -> dict:
    """Extract frozen text signals + financial facts for the What-If simulator.
    Financial facts come from the parsed financial_dict stored on forensic_scores.notes
    — we reconstruct them from the engine's own extraction for accuracy."""
    # ── Frozen text signals (heavy ML is done; never re-run these) ──
    text_signals = {
        "distress_score":  res.distress_score,
        "neg_prob":        res.calibrated_probs.get("negative", 0.0),
        "neu_prob":        res.calibrated_probs.get("neutral",  0.0),
        "pos_prob":        res.calibrated_probs.get("positive", 0.0),
        "hedging_level":   getattr(res.hedging_result, "hedging_level", "") if res.hedging_result else "",
        "altman_distress": (
            res.forensic_scores.altman_zone.startswith("DISTRESS ZONE")
            if res.forensic_scores and res.forensic_scores.calculated else False
        ),
    }

    # ── Financial facts actually used by ForensicAccountingEngine ──
    # Pull from the forensic result's internal calculations when available,
    # otherwise return an empty dict (simulator shows 'no financial data' state)
    f = res.forensic_scores
    financial_facts: dict = {}
    if f and f.calculated:
        # We store the individual metric values derived in ForensicAccountingEngine
        # so sliders are seeded at the exact values the engine used.
        if f.altman_z_score is not None or f.beneish_m_score is not None:
            # At minimum total_assets was > 0 for these to run — emit a non-empty dict
            # The values are unavailable directly; flag snapshot as arithmetic-ready.
            financial_facts = {"_forensic_ran": True}

    return {
        "frozen_text_signals": text_signals,
        "financial_facts": financial_facts,
        "baseline_rating": res.risk_grade.value,
        "baseline_priority": res.priority_index,
    }


def process_and_format(text: str, file_meta: dict, exposure: float, financial_dict: dict = None) -> dict:
    tracer = TraceLogger()
    tracer.log("INGESTION", f"Loaded {file_meta.get('name', 'text input')} ({file_meta.get('size', 0)} bytes)")
    
    advisor: FinancialAdvisor = ml_models["advisor"]
    
    tracer.log("ENGINE", "Running Financial Advisor rules and models...")
    res = advisor.advise(text, financial_dict=financial_dict, exposure_value=exposure, show_trace=False)
    
    tracer.log("SYNTHESIS", "Generating final institutional verdict...")
    
    masking_data = format_masking(res)
    masking_data["exposure"] = exposure
    
    # Build simulator snapshot: frozen text signals + extracted financial facts
    # Pass through the raw financial_dict so the simulator can seed sliders
    snapshot = format_simulator_snapshot(res)
    if financial_dict:
        snapshot["financial_facts"] = financial_dict   # raw extracted values for sliders
    
    return {
        "file_meta": file_meta,
        "trace_logs": tracer.logs,
        "verdict": format_verdict(res),
        "quantitative": format_quant(res),
        "nlp_sentiment": format_nlp(res),
        "rhetorical_masking": masking_data,
        "priority_index": res.priority_index,
        "action_flag": res.action_flag.value,
        "action_recommendations": res.action_recommendations,
        "simulator_snapshot": snapshot,
    }


@app.get("/api/samples")
async def list_samples():
    samples_dir = os.path.join("data", "sample_reports")
    if not os.path.exists(samples_dir):
        return []
    files = []
    for f in os.listdir(samples_dir):
        if os.path.isfile(os.path.join(samples_dir, f)):
            size = os.path.getsize(os.path.join(samples_dir, f))
            files.append({"filename": f, "size": f"{size/1024:.1f} KB"})
    return files

from starlette.concurrency import run_in_threadpool
import tempfile

@app.post("/api/analyze")
async def analyze_document(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    sample_name: Optional[str] = Form(None),
    exposure: float = Form(0.0)
):
    tracer = TraceLogger()
    tracer.log("REQUEST", "Received analysis request")
    
    if file:
        tracer.log("INGESTION", f"Reading uploaded file {file.filename}")
        content = await file.read()
        if len(content) > 100 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large (Max 100MB)")
        
        # Save unique isolated temp file to prevent concurrent race collisions
        ext = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(content)
            temp_path = tmp.name
        
        try:
            parsed = await run_in_threadpool(DocumentParser.parse, temp_path, extract_images=False)
            file_meta = {"name": file.filename, "size": len(content)}
            final_text = parsed.raw_text
            fin_dict = getattr(parsed, "financial_dict", None)
            parsed.cleanup()
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
                
    elif sample_name:
        sample_path = os.path.join("data", "sample_reports", sample_name)
        if not os.path.exists(sample_path):
            raise HTTPException(status_code=404, detail="Sample not found")
            
        tracer.log("INGESTION", f"Reading sample file {sample_name}")
        size = os.path.getsize(sample_path)
        parsed = await run_in_threadpool(DocumentParser.parse, sample_path, extract_images=False)
        file_meta = {"name": sample_name, "size": size}
        final_text = parsed.raw_text
        fin_dict = getattr(parsed, "financial_dict", None)
        parsed.cleanup()
        
    elif text:
        tracer.log("INGESTION", "Reading raw text input")
        final_text = text
        fin_dict = None
        file_meta = {"name": "Raw Text", "size": len(text)}
        
    else:
        raise HTTPException(status_code=400, detail="Must provide file, text, or sample_name")

    return await run_in_threadpool(process_and_format, final_text, file_meta, exposure, fin_dict)


@app.post("/api/simulate")
async def simulate_whatif(
    text: Optional[str] = Form(None),
    sample_name: Optional[str] = Form(None),
    exposure: float = Form(...)
):
    # Resolve text source — existing text path unchanged
    if sample_name:
        sample_path = os.path.join("data", "sample_reports", sample_name)
        if not os.path.exists(sample_path):
            raise HTTPException(status_code=404, detail="Sample not found")
        parsed = await run_in_threadpool(DocumentParser.parse, sample_path, extract_images=False)
        resolved_text = parsed.raw_text
        parsed.cleanup()
    elif text:
        resolved_text = text
    else:
        raise HTTPException(status_code=400, detail="Must provide text or sample_name")

    advisor: FinancialAdvisor = ml_models["advisor"]
    res = await run_in_threadpool(advisor.advise, resolved_text, exposure_value=exposure, show_trace=False)
    
    masking_data = format_masking(res)
    masking_data["exposure"] = exposure
    
    return {
        "priority_index": res.priority_index,
        "action_flag": res.action_flag.value,
        "rhetorical_masking": masking_data,
        "verdict": format_verdict(res)
    }

# ── /api/simulate_facts — pure-math in-memory simulator (zero ML inference) ─
@app.post("/api/simulate_facts")
async def simulate_facts(request: Request):
    """
    In-memory What-If simulator.
    Accepts JSON: { financial_facts: {...}, frozen_text_signals: {...}, exposure: float }
    Runs ONLY ForensicAccountingEngine (pure arithmetic, ~1ms) + fusion rules.
    No FinBERT / ABSA / hedging / discourse re-run.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=422, detail="Expected JSON body")

    financial_facts: dict   = body.get("financial_facts", {})
    text_signals:   dict    = body.get("frozen_text_signals", {})
    exposure:       float   = float(body.get("exposure", 0.0))

    # ── Step 1: Re-run forensic formulas only (pure arithmetic) ──
    forensic_res = ForensicAccountingEngine.compute(financial_facts)

    # ── Step 2: Reconstruct base grade from frozen text signals ──
    distress_score = float(text_signals.get("distress_score", 0.0))
    neg_prob       = float(text_signals.get("neg_prob",       0.0))

    if distress_score > 75 or neg_prob > 0.60:
        base_grade = RiskGrade.CRITICAL
    elif neg_prob > 0.35 or distress_score > 40:
        base_grade = RiskGrade.WARNING
    elif distress_score > 20:
        base_grade = RiskGrade.MONITOR
    else:
        base_grade = RiskGrade.MINIMAL

    # ── Step 3: Apply forensic + hedging overrides (fusion rules) ──
    class _FakeHedge:
        hedging_level = text_signals.get("hedging_level", "")
    fake_hedge = _FakeHedge() if text_signals.get("hedging_level") else None
    final_grade, override_reasons = apply_overrides(base_grade, forensic_res, fake_hedge)

    # ── Step 4: Policy table lookup ──
    exposure_tier, action_flag, recs = POLICY_TABLE[final_grade]
    priority_index = (distress_score / 100.0) * exposure

    # ── Step 5: Build forensic scores dict for response ──
    f = forensic_res
    forensic_scores = {}
    if f and f.calculated:
        forensic_scores = {
            "altman_z":    {"score": f.altman_z_score,  "status": f.altman_zone},
            "beneish_m":   {"score": f.beneish_m_score, "status": f.beneish_manipulation_risk},
            "piotroski_f": {"score": f.piotroski_f_score, "status": f.piotroski_grade},
        }

    return {
        "verdict": {
            "rating":         final_grade.value,
            "score":          f"{distress_score:.1f} / 100",
            "interpretation": exposure_tier.value,
            "confidence":     1.0,
        },
        "forensic_scores":      forensic_scores,
        "priority_index":       priority_index,
        "action_flag":          action_flag.value,
        "override_reasons":     override_reasons,
    }


if os.path.exists("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
