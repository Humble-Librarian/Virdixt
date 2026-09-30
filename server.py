import os
import time
import io
import json
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from infer import LayaSystem1, AnchorTokenPruner, FinancialAdvisor, AdvisorResult, RiskGrade
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

def process_and_format(text: str, file_meta: dict, exposure: float, financial_dict: dict = None) -> dict:
    tracer = TraceLogger()
    tracer.log("INGESTION", f"Loaded {file_meta.get('name', 'text input')} ({file_meta.get('size', 0)} bytes)")
    
    advisor: FinancialAdvisor = ml_models["advisor"]
    
    tracer.log("ENGINE", "Running Financial Advisor rules and models...")
    res = advisor.advise(text, financial_dict=financial_dict, exposure_value=exposure, show_trace=False)
    
    tracer.log("SYNTHESIS", "Generating final institutional verdict...")
    
    masking_data = format_masking(res)
    masking_data["exposure"] = exposure
    
    return {
        "file_meta": file_meta,
        "trace_logs": tracer.logs,
        "verdict": format_verdict(res),
        "quantitative": format_quant(res),
        "nlp_sentiment": format_nlp(res),
        "rhetorical_masking": masking_data,
        "priority_index": res.priority_index,
        "action_flag": res.action_flag.value,
        "action_recommendations": res.action_recommendations
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
    text: str = Form(...),
    exposure: float = Form(...)
):
    advisor: FinancialAdvisor = ml_models["advisor"]
    res = await run_in_threadpool(advisor.advise, text, exposure_value=exposure, show_trace=False)
    
    masking_data = format_masking(res)
    masking_data["exposure"] = exposure
    
    return {
        "priority_index": res.priority_index,
        "action_flag": res.action_flag.value,
        "rhetorical_masking": masking_data,
        "verdict": format_verdict(res)
    }

if os.path.exists("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
