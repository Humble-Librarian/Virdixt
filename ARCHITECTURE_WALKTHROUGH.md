# 🧠 Virdixt: Complete Architecture & Codebase Walkthrough

> **Welcome to the Virdixt Engine!**  
> This guide is an interactive, visual walkthrough designed to help every teammate understand **what each file does**, **how data flows between modules**, and **how the entire system connects** from raw training data to real-time C++ inference and operational policy action flags.

---

## 🗺️ 1. The Big Picture: 30,000-Foot System Map

Virdixt is split into two clean lifecycles: **Offline Training (Python)** and **Online Real-Time Inference (Python & C++)**.

```mermaid
flowchart TD
    subgraph Offline_Training["📦 1. OFFLINE TRAINING (Python)"]
        D1[prepare_data.py] -->|data/base_real.jsonl| MDB[build_rich_dataset.py]
        D2[synthetic_builder.py] -->|data/synthetic.jsonl| MDB
        D3[complex_sentence_injector.py] -->|data/complex.jsonl| MDB
        MDB -->|data/train.jsonl & val.jsonl| TR[train.py / soup.yaml]
        TR -->|./output model weights| EXP[export_onnx.py]
        EXP -->|models/finbert.onnx| ONNX_OUT[(Static ONNX Graph)]
    end

    subgraph Document_Ingestion["📂 2. MULTI-FORMAT INGESTION (document_parser.py)"]
        DOC[PDF / DOCX / TXT Document] --> PARSER[Universal DocumentParser]
        PARSER -->|Clean Text Stream| PRUN[Anchor Token Pruner]
        PARSER -->|Embedded Visual Assets| DET[vision/chart_detector.py]
    end

    subgraph Multimodal_Vision["👁️ 3. VISION SUBSYSTEM (vision/)"]
        DET -->|If Chart Detected| EXT[vision/chart_extractor.py]
        EXT -->|Fast Heuristic / DePlot Table| CALC[vision/delta_calculator.py]
        CALC -->|Concessive Sentence Injection| VP[vision/pipeline.py]
        VP -.->|Injected Delta Text| PRUN
    end

    subgraph RealTime_Inference["⚡ 4. REAL-TIME DECISION RUNTIME (Python & C++)"]
        PRUN -->|Dense Signal Sentences| BACKBONE[FinBERT ONNX Runtime]
        ONNX_OUT -.->|Loads Model Once| BACKBONE
        BACKBONE -->|Calibrated Logits| LAYA[Laya System-1 Primitives]
        LAYA -->|Choice, Score 0-100, Noul| ERP[ERP / Policy Advisor Engine]
        ERP -->|Action Directives| ACT[Policy Action: FREEZE_PURCHASE_ORDERS / PROCEED]
    end

    style Offline_Training fill:#1e1e2e,stroke:#89b4fa,stroke-width:2px,color:#cdd6f4
    style Document_Ingestion fill:#1e1e2e,stroke:#a6adc8,stroke-width:2px,color:#cdd6f4
    style Multimodal_Vision fill:#181825,stroke:#f9e2af,stroke-width:2px,color:#cdd6f4
    style RealTime_Inference fill:#11111b,stroke:#a6e3a1,stroke-width:2px,color:#cdd6f4
```

---

## 🔄 2. End-to-End Sequence Diagram

Here is what happens when a document containing text and a financial chart enters Virdixt:

```mermaid
sequenceDiagram
    autonumber
    actor User as Client / Enterprise Application
    participant Parser as document_parser.py
    participant Pipe as vision/pipeline.py
    participant Vision as vision/chart_extractor.py & delta_calc.py
    participant Pruner as text_preprocessor / AnchorPruner
    participant Backbone as FinBERT (ONNX Runtime)
    participant Laya as laya_primitives (Choice/Score/Noul)
    participant Advisor as erp_advisor / FinancialAdvisor

    User->>Parser: Submit Document (.pdf / .docx / .txt)
    Parser->>Parser: Extract text stream & embedded visual assets
    alt Has Embedded Visual Assets
        Parser->>Pipe: Route extracted chart images
        Pipe->>Vision: Fast Heuristic / DePlot Table Extraction
        Vision-->>Pipe: "Although Revenue grew 6.7%, Margin dropped 38.9%"
        Pipe-->>Pruner: Injected Concessive Sentence
    else Pure Text Document
        Parser-->>Pruner: Direct Text Stream (0ms visual overhead)
    end
    Pruner->>Pruner: Filter out fluff; extract financial signal tokens (0.1ms)
    Pruner->>Backbone: Dense Tokens (input_ids, attention_mask)
    Backbone->>Backbone: ONNX Forward Pass (~1.5ms GPU / ~10ms CPU)
    Backbone-->>Laya: Raw Output Logits [-2.1, 0.4, 3.8]
    Laya->>Laya: Temperature Scaling (T=1.25) & Softmax
    Laya->>Laya: Compute Score (0-100) & Noul Risk Probabilities (40ns)
    Laya-->>Advisor: LayaVerdict(Choice=NEGATIVE, Score=88.5, P_covenant=0.95)
    Advisor->>Advisor: Evaluate Asymmetric Risk Gate (>35% negative)
    Advisor-->>User: PolicyAction: FREEZE_PURCHASE_ORDERS (CRITICAL / TIER 4)
```

---

## 📂 3. File-by-File Deep Dive

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ MODULE 1: DATA GENERATION & BALANCING (THE DATA FOUNDRY)                         │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### 1. `prepare_data.py`
* **What it does:** Connects to Hugging Face and downloads `zeroshot/twitter-financial-news-sentiment` (9,543 real financial market tweets and headlines). Remaps the original market labels (`0: Bearish` $\to$ `negative`, `1: Bullish` $\to$ `positive`, `2: Neutral` $\to$ `neutral`).
* **Input:** Remote Hugging Face dataset.
* **Output:** `data/base_real.jsonl` (raw real news rows).
* **Why it matters:** Ground-truth baseline of authentic market vocabulary.

### 2. `synthetic_builder.py`
* **What it does:** Generates 1,500 domain-specific corporate accounting sentences across 30 parameterized templates (10 negative, 10 positive, 10 neutral). Injects authentic accounting terms: *impairment charges, debt covenant headroom, ARR growth, EBITDA expansion, dividend suspensions*.
* **Input:** None (algorithmic generation with random numeric distributions).
* **Output:** `data/synthetic.jsonl`.
* **Why it matters:** Real news tweets often lack balance-sheet accounting disclosures; this injects institutional corporate vocabulary.

### 3. `complex_sentence_injector.py`
* **What it does:** Generates 750 multi-clause adversarial sentences using concessive conjunctions (*"Although"*, *"Despite"*, *"Notwithstanding"*, *"Even though"*).
  - *Example Negative:* *"Although revenue expanded by 14%, operating cash flow turned deeply negative."*
  - *Example Positive:* *"Despite a $110M one-off impairment charge, operating margins surged."*
* **Output:** `data/complex.jsonl`.
* **Why it matters:** **This is the secret sauce.** Standard FinBERT sees the word "growth" and guesses positive. This file teaches the model financial priority: *Cash Flow > Revenue* and *Guidance > Historical Quarter*.

### 4. `build_rich_dataset.py`
* **What it does:** The master dataset assembler. Loads `base_real.jsonl`, `synthetic.jsonl`, and `complex.jsonl`, balances them to exactly **2,100 rows per class (6,300 total)**, shuffles with deterministic seed `42`, and creates an 85/15 train/val split.
* **Output:** `data/train.jsonl` (5,355 rows) and `data/val.jsonl` (945 rows).
* **Why it matters:** Solves the **75% Problem**. Equal gradient pressure raises Negative Recall from **9.5% to 88.6%**.

---

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ MODULE 2: MODEL TRAINING, EVALUATION & EXPORT                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### 5. `soup.yaml`
* **What it does:** Declarative fine-tuning configuration for the **Soup CLI** engine. Specifies `task: classifier`, base model `ProsusAI/finbert`, 3 labels, learning rate `2e-5`, batch size `4`, gradient accumulation `4`, and `3 epochs`.
* **How to run:** `soup train --config soup.yaml`

### 6. `train.py`
* **What it does:** Standalone PyTorch fine-tuning script. Implements a `CustomTrainer` with **Class-Weighted CrossEntropyLoss**, CPU multi-threading caps (`torch.set_num_threads(4)`), and auto-detects CUDA for GPU acceleration.
* **Input:** `data/train.jsonl`, `data/val.jsonl`.
* **Output:** Fine-tuned model checkpoints saved to `./output/`.

### 7. `eval.py`
* **What it does:** Loads `./output/` and runs a full validation sweep over `data/val.jsonl`. Generates the complete Scikit-Learn `classification_report` (Precision, Recall, F1 per class) and the confusion matrix.
* **Expected Result:** **90.26% Accuracy, 0.9022 Macro-F1, 88.64% Negative Recall**.

### 8. `export_onnx.py`
* **What it does:** "Freezes" the PyTorch model into an optimized **ONNX computation graph** (`models/finbert.onnx`) with dynamic batch axes. Supports an optional `--quantize` flag to produce an **INT8 quantized model** (3x faster on Intel CPUs).
* **Output:** `models/finbert.onnx` + `models/tokenizer.json`.
* **Why it matters:** Bridges the Python training world with the bare-metal C++ inference engine.

---

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ MODULE 3: DOCUMENT INGESTION & MULTIMODAL VISION (`document_parser.py` & `vision/`)│
└──────────────────────────────────────────────────────────────────────────────────┘
```

### 9. `document_parser.py`
* **What it does:** Universal multi-format document ingestor.
  - **TXT:** Multi-encoding text reader (`utf-8`, `utf-8-sig`, `latin-1`, `cp1252`).
  - **DOCX:** Extracts text paragraphs and automatically decompresses embedded chart drawings/images from `word/media/`.
  - **PDF:** Extracts text pages and extracts high-resolution embedded raster images via `PyMuPDF` (`fitz`) / `pypdf`.
* **Output:** `ParsedDocument(raw_text, image_paths, file_type, page_count, has_visuals)`.
* **Why it matters:** Allows ingesting any corporate document format. If no images exist, bypasses the vision pipeline with **0ms visual overhead**.

### 10. `vision/chart_detector.py`
* **What it does:** Uses Microsoft's ultra-lightweight **Florence-2** model (with fast visual bounding-box heuristic fallback) to classify incoming images as *financial charts* vs. *decorative photos/logos*.
* **Output:** `ChartDetectionResult(is_chart=True/False, confidence=0.98)`.

### 11. `vision/chart_extractor.py`
* **What it does:** Extracts table data from charts.
  - **Fast-Path Heuristic (Default):** Extracts period deltas in **< 5ms**.
  - **Deep Vision (`--deep-vision`):** Runs **Google DePlot** (`google/deplot`) to generate linearized table strings.
* **Why it matters:** Eliminates 10+ second CPU wait times for standard charts while allowing deep token autoregression when explicitly requested.

### 12. `vision/delta_calculator.py`
* **What it does:** **Zero-ML, pure Python arithmetic.** Parses the linearized table, calculates exact percentage changes ($\Delta = \frac{v_2 - v_1}{v_1} \times 100$), and synthesizes a concessive sentence (*"Although Revenue grew 6.7%, Gross Margin declined 38.9%"*).
* **Execution Time:** **< 0.05 milliseconds.** Zero hallucination risk.

### 13. `vision/pipeline.py`
* **What it does:** High-level vision orchestrator. Glues Detector $\to$ Extractor $\to$ Delta Calculator together. Enriches document text streams with concessive chart analysis before passing to FinBERT.

---

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ MODULE 4: REAL-TIME DECISION ENGINE (`infer.py`)                                 │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### 14. `infer.py`
* **What it does:** The complete high-performance decision runtime containing:
  1. **`AnchorTokenPruner`:** Regex pruner extracting dense signal sentences (0.1ms).
  2. **`LayaSystem1`:** Computes `Choice`, `Score` (0-100 distress index), and `Noul` binary risk hypotheses (`liquidity_distress`, `debt_covenant_breach_risk`, `growth_momentum`, `capital_return`). Auto-detects fast ONNX Runtime.
  3. **`FinancialAdvisor`:** Applies the **Asymmetric Risk Gate (35% threshold)** and outputs strict operational action flags (`FREEZE_PURCHASE_ORDERS`, `FLAG_FOR_REVIEW`, `PROCEED_NORMAL`).
  4. **Multi-Mode Execution:**
     - **One-Shot File Evaluation:** `python infer.py --file <doc.pdf / doc.docx / doc.txt>`
     - **Batch Directory Audit:** `python infer.py --batch <folder_path>`
     - **Warm Interactive REPL:** `python infer.py --interactive` (Sub-10ms evaluation per document)

---

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ MODULE 5: HIGH-THROUGHPUT C++ RUNTIME (`cpp/`)                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### 15. `cpp/include/laya_primitives.hpp`
* **What it does:** Header-only modern C++20 math engine. Implements Temperature Softmax ($T=1.25$), continuous Distress Index $[0, 100]$, and sigmoid-calibrated Noul propositions.
* **Speed:** **~40 nanoseconds.**

### 16. `cpp/include/inference_engine.hpp`
* **What it does:** Direct C++ wrapper around Microsoft ONNX Runtime with standalone simulation fallback. Automatically initializes CPU thread pools (AVX2/VNNI vectorization) or appends the NVIDIA CUDA execution provider.

### 17. `cpp/include/text_preprocessor.hpp`
* **What it does:** C++ regex-based anchor token pruner. Compresses multi-page documents down to dense financial signal sentences before tokenization.

### 18. `cpp/include/erp_advisor.hpp`
* **What it does:** Deterministic rule engine in C++ that maps Laya risk grades into formatted audit reports and operational policy flags.

### 19. `cpp/src/main.cpp` & `cpp/CMakeLists.txt`
* **What it does:** C++ executable entry point with built-in test suite executing complex multi-clause enterprise scenarios.

---

## 🧩 4. Interactive File Dependency Matrix

Use this matrix to understand what breaks if you edit a file:

| File | Depends On (Inputs) | Consumed By (Downstream) | If you change this... |
|---|---|---|---|
| `prepare_data.py` | HuggingFace Hub | `build_rich_dataset.py` | Changes raw real-news baseline |
| `synthetic_builder.py` | None | `build_rich_dataset.py` | Changes corporate accounting vocabulary |
| `complex_sentence_injector.py` | None | `build_rich_dataset.py` | Changes adversarial concessive conjunctions |
| `build_rich_dataset.py` | `data/*.jsonl` | `train.py`, `soup.yaml` | **Changes training distribution & class balance** |
| `train.py` / `soup.yaml` | `data/train.jsonl` | `eval.py`, `export_onnx.py` | **Changes model neural weights (`./output`)** |
| `export_onnx.py` | `./output/` | `infer.py`, `cpp/` | **Updates `finbert.onnx` for Python & C++** |
| `document_parser.py` | Local files (.pdf/.docx/.txt) | `infer.py` | Changes text & visual asset extraction |
| `vision/delta_calculator.py`| Table strings | `vision/pipeline.py` | Changes mathematical delta phrasing |
| `vision/pipeline.py` | `vision/*` | `infer.py` | Changes multimodal document ingestion |
| `laya_primitives.hpp` | Raw logits | `main.cpp`, ERP Advisor | **Changes Laya score math & risk calibration** |
| `infer.py` | `./output` or `.onnx` | End Users / Enterprise Systems | Main Python entry point for live decisions |

---

## ⚡ 5. Execution Recipes for Teammates

### Recipe 1: Fast One-Shot Document Evaluation
```bash
# Evaluate any PDF, Word (.docx), or Text (.txt) report:
python infer.py --file data/sample_reports/covenant_breach.pdf
python infer.py --file data/sample_reports/healthy_report.docx
python infer.py --file data/sample_reports/distress_report.txt

# Evaluate a multimodal document with embedded charts:
python infer.py --file data/sample_reports/multimodal_report.docx

# Optional deep vision autoregression:
python infer.py --file data/sample_reports/multimodal_report.docx --deep-vision
```

### Recipe 2: Warm Interactive REPL Session (Sub-10ms per document)
```bash
python infer.py --interactive
# At the virdixt > prompt, type any filename or text commentary!
```

### Recipe 3: Multi-Document Batch Directory Audit
```bash
python infer.py --batch data/sample_reports/
```

### Recipe 4: Direct Text Analysis
```bash
python infer.py --text "Supplier defaulted on obligations leading to $5M inventory write-down."
```

### Recipe 5: Rebuilding Dataset & Fine-Tuning
```bash
# Rebuild dataset:
python prepare_data.py
python synthetic_builder.py
python complex_sentence_injector.py
python build_rich_dataset.py

# Fine-tune model:
python train.py
# OR: soup train --config soup.yaml
```

### Recipe 6: Exporting to ONNX & Quantizing
```bash
python export_onnx.py --quantize
```

### Recipe 7: Compiling & Running C++ Engine
```bash
cd cpp
cmake -B build
cmake --build build --config Release
./build/virdixt_engine
```

---

## 🎯 Summary for Teammates
* **To add support for new file formats (e.g. HTML, RTF):** Extend `document_parser.py`.
* **To improve accuracy on complex statements:** Edit `complex_sentence_injector.py`.
* **To add new corporate accounting phrases:** Edit `synthetic_builder.py`.
* **To adjust distress sensitivity & thresholds:** Edit the risk gates in `infer.py` and `cpp/include/laya_primitives.hpp`.
* **To modify ERP action flags:** Edit `cpp/include/erp_advisor.hpp` and `FinancialAdvisor` in `infer.py`.

