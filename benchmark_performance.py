"""
Virdixt Real-World Empirical Benchmark Suite
Measures actual wall-clock latencies, throughput, and memory footprint
on the host machine across all individual subsystem components.
"""

import os
import sys
import time
import tracemalloc
import numpy as np

# Set stdout to UTF-8
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from infer import LayaSystem1, AnchorTokenPruner, FinancialAdvisor
from document_parser import DocumentParser
from nlp.linguistic_hedging import LinguisticHedgingDetector
from nlp.discourse_parser import RhetoricalDiscourseParser
from nlp.forensic_accounting import ForensicAccountingEngine
from nlp.absa_engine import ABSAEngine
from vision.image_enhancer import DocumentImageEnhancer
from vision.ocr_engine import get_ocr_engine

SAMPLE_DOC = """
GLOBAL LOGISTICS HOLDINGS - Q3 COMPREHENSIVE PERFORMANCE REVIEW
Although gross revenue expanded by 18.5% to $145.2M, operating cash flows deteriorated significantly to negative ($14.8M).
Management believes that the strategic realignment might possibly be a one-off adjustment due to transitory macro headwinds.
Operating expenses increased to $48.5M driven by higher logistics and fuel costs.
Debt covenants were impacted, leaving minimal headroom under the primary credit facility.
Auditor Opinion: Going concern explanatory paragraph highlighted in the independent audit report.
"""

SAMPLE_FINANCIAL_DICT = {
    "Total_Assets": 250000000,
    "Revenue": 145200000,
    "Gross_Profit": 42000000,
    "Operating_Expenses": 48500000,
    "Operating_Income": -6500000,
    "Net_Income": -12400000,
    "Total_Debt": 130000000,
    "Cash_and_Cash_Equivalents": 8500000,
}


def run_benchmarks():
    print("=" * 80)
    print("      VIRDIXT EMPIRICAL BENCHMARK: MEASURED HARDWARE NUMBERS")
    print("=" * 80)

    tracemalloc.start()

    # ---------------------------------------------------------
    # 1. FinBERT Backbone: Single-Pass vs Multi-Pass
    # ---------------------------------------------------------
    print("\n[1] BENCHMARKING FINBERT ONNX ENGINE...")
    use_onnx = os.path.exists("models/finbert.onnx")
    system1 = LayaSystem1(model_dir="./output", use_onnx=use_onnx)
    pruner = AnchorTokenPruner()
    pruned = pruner.prune(SAMPLE_DOC)

    # Warmup
    for _ in range(3):
        system1.get_calibrated_probs(pruned)

    # Single-Pass Benchmark (100 iterations)
    N_RUNS = 50
    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        probs, choice, score, noul = system1.get_all_predictions(pruned)
    t_single = (time.perf_counter() - t0) / N_RUNS * 1000.0

    print(f"    • Single-Pass (get_all_predictions) Latency : {t_single:.2f} ms / doc")
    print(f"    • Pure FinBERT Throughput                  : {1000.0 / t_single:.1f} docs / sec")

    # ---------------------------------------------------------
    # 2. NLP Analytical Lanes
    # ---------------------------------------------------------
    print("\n[2] BENCHMARKING 5-LANE NLP SUB-COMPONENTS...")
    
    # Hedging Detector (with LRU syllable cache)
    hedging = LinguisticHedgingDetector()
    t0 = time.perf_counter()
    for _ in range(200):
        h_res = hedging.analyze(SAMPLE_DOC)
    t_hedge = (time.perf_counter() - t0) / 200 * 1000.0
    print(f"    • Linguistic Hedging & Gunning-Fog Latency : {t_hedge:.3f} ms / doc")

    # Rhetorical Discourse Parser
    discourse = RhetoricalDiscourseParser()
    t0 = time.perf_counter()
    for _ in range(200):
        d_res = discourse.parse(SAMPLE_DOC)
    t_disc = (time.perf_counter() - t0) / 200 * 1000.0
    print(f"    • RST Discourse Nucleus Parser Latency     : {t_disc:.3f} ms / doc")

    # Forensic Accounting Engine (Altman Z + Beneish M + Piotroski F)
    forensic = ForensicAccountingEngine()
    t0 = time.perf_counter()
    for _ in range(500):
        f_res = forensic.compute(SAMPLE_FINANCIAL_DICT)
    t_forensic = (time.perf_counter() - t0) / 500 * 1000.0
    print(f"    • Quantitative Forensic Accounting Latency: {t_forensic:.3f} ms / doc")

    # Aspect-Based Sentiment Analysis (ABSA)
    absa = ABSAEngine(system1)
    t0 = time.perf_counter()
    for _ in range(20):
        a_res = absa.evaluate(SAMPLE_DOC)
    t_absa = (time.perf_counter() - t0) / 20 * 1000.0
    print(f"    • 5-Aspect ABSA Full Evaluation Latency    : {t_absa:.2f} ms / doc")

    # ---------------------------------------------------------
    # 3. Document Ingestion Subsystem
    # ---------------------------------------------------------
    print("\n[3] BENCHMARKING MULTI-FORMAT DOCUMENT INGESTION...")
    
    # Test sample files if available
    samples = {
        "Excel (.xlsx)": "data/sample_reports/corporate_filing.xlsx",
        "CSV (.csv)": "data/sample_reports/narrative_audit_log.csv",
        "PDF (.pdf)": "data/sample_reports/covenant_breach.pdf",
        "DOCX (.docx)": "data/sample_reports/healthy_report.docx",
        "TXT (.txt)": "data/sample_reports/distress_report.txt",
    }
    
    for label, path in samples.items():
        if os.path.exists(path):
            t0 = time.perf_counter()
            for _ in range(10):
                doc = DocumentParser.parse(path, extract_images=False)
                doc.cleanup()
            t_ingest = (time.perf_counter() - t0) / 10 * 1000.0
            print(f"    • Ingestion: {label:<15} ({os.path.getsize(path)/1024:.1f} KB) -> {t_ingest:.2f} ms")

    # ---------------------------------------------------------
    # 4. End-to-End Full Advisor Pipeline Execution
    # ---------------------------------------------------------
    print("\n[4] BENCHMARKING FULL END-TO-END PIPELINE (All 5 Lanes + ERP Advisor)...")
    advisor = FinancialAdvisor(system1, pruner)

    # Warmup
    advisor.advise(SAMPLE_DOC, financial_dict=SAMPLE_FINANCIAL_DICT, exposure_value=500000.0)

    t0 = time.perf_counter()
    N_E2E = 25
    for _ in range(N_E2E):
        res = advisor.advise(SAMPLE_DOC, financial_dict=SAMPLE_FINANCIAL_DICT, exposure_value=500000.0)
    t_e2e = (time.perf_counter() - t0) / N_E2E * 1000.0

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"    • Full End-to-End Analysis Latency         : {t_e2e:.2f} ms / doc")
    print(f"    • Full Institutional Throughput            : {1000.0 / t_e2e:.1f} complete audits / sec")
    print(f"    • Total Memory Allocation Overhead         : {peak_mem / (1024 * 1024):.2f} MB")

    print("\n" + "=" * 80)
    print("                         SUMMARY SCOREBOARD")
    print("=" * 80)
    print(f"  • Single-Pass FinBERT Forward Pass : {t_single:>8.2f} ms")
    print(f"  • Linguistic Hedging & Fog Index   : {t_hedge:>8.3f} ms")
    print(f"  • RST Discourse Nucleus Separation : {t_disc:>8.3f} ms")
    print(f"  • Quantitative Forensic Ratios     : {t_forensic:>8.3f} ms")
    print(f"  • 5-Aspect ABSA Sub-Score Stream   : {t_absa:>8.2f} ms")
    print(f"  ────────────────────────────────────────────────────────")
    print(f"  ⚡ TOTAL END-TO-END AUDIT LATENCY   : {t_e2e:>8.2f} ms")
    print(f"  💾 PEAK RUNTIME MEMORY FOOTPRINT   : {peak_mem / (1024 * 1024):>8.2f} MB")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmarks()
