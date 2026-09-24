# Virdixt — Multimodal FinBERT Financial Sentiment & Laya System-1 Decision Engine

<p align="center">
  <img src="https://img.shields.io/badge/Accuracy-90.26%25-brightgreen" />
  <img src="https://img.shields.io/badge/Macro_F1-0.9022-brightgreen" />
  <img src="https://img.shields.io/badge/Model-FinBERT-blue" />
  <img src="https://img.shields.io/badge/Dataset-6%2C300_rows-blue" />
  <img src="https://img.shields.io/badge/Runtime-C%2B%2B_%2B_ONNX-purple" />
  <img src="https://img.shields.io/badge/Hardware-CPU_(i5)_%2F_GPU_(RTX_3050)-orange" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

> Fine-tuned FinBERT for financial sentiment analysis (Positive / Negative / Neutral) augmented with **Laya Open-Weights System-1 Decision Primitives (Choice, Score, Noul)**, an **Automated Financial Advisor Engine**, a **Deterministic Chart-to-Table Vision Pipeline**, and a **Zero-Overhead C++ Inference Runtime** — built from first principles for Intel CPUs and NVIDIA GPUs.

> 📖 **Teammate Guide:** For a file-by-file visual breakdown and interactive dependency map, see **[`ARCHITECTURE_WALKTHROUGH.md`](ARCHITECTURE_WALKTHROUGH.md)**.

---

## Table of Contents

1. [Project Origin & Problem Statement](#1-project-origin--problem-statement)
2. [Why FinBERT Instead of an LLM](#2-why-finbert-instead-of-an-llm)
3. [Why Soup CLI Was Chosen](#3-why-soup-cli-was-chosen)
4. [The Structured Data vs. Sentiment Challenge](#4-the-structured-data-vs-sentiment-challenge)
5. [Dataset Selection & Why](#5-dataset-selection--why)
6. [Hardware Constraints & Hardware Scaling (i5 to RTX 3050)](#6-hardware-constraints--hardware-scaling)
7. [The 75% Problem — Root Cause Analysis](#7-the-75-problem--root-cause-analysis)
8. [Architecture Battle: Senior Architect vs. Ponytail](#8-architecture-battle-senior-architect-vs-ponytail)
9. [Laya System-1 Primitives & Financial Advisor Engine](#9-laya-system-1-primitives--financial-advisor-engine)
10. [Multimodal Vision Pipeline (Florence-2 + DePlot + Math)](#10-multimodal-vision-pipeline)
11. [High-Throughput C++ Runtime Engine](#11-high-throughput-c-runtime-engine)
12. [Dataset Pipeline Design (6,300 Balanced Rows)](#12-dataset-pipeline-design)
13. [Complex Sentence Injection](#13-complex-sentence-injection)
14. [Final Results & Benchmarks](#14-final-results--benchmarks)
15. [Repository Structure](#15-repository-structure)
16. [How to Run (Python & C++)](#16-how-to-run)

---

## 1. Project Origin & Problem Statement 

This project began as an **NLP project** on **Financial Sentiment Analysis** using BERT or FinBERT with custom parameters.

The core challenge: rather than manually writing 200+ lines of training loop boilerplate (tokenization, batching, optimizer setup, scheduler, checkpointing, mixed precision, evaluation), we asked:

> *"Is there something like Soup CLI for LLMs that we could adapt for BERT to save engineering time?"*

The answer: **Soup CLI supports `task: classifier` out of the box**, which wraps `AutoModelForSequenceClassification` — exactly the Hugging Face class that BERT and FinBERT use for sequence classification. This became the foundation of the pipeline.

---

## 2. Why FinBERT Instead of an LLM

We debated whether to use a generative LLM (Llama, Qwen, etc.) or an encoder model (BERT).

**Decision: FinBERT (`ProsusAI/finbert`)**

| Factor | Generative LLM (7B+) | FinBERT (110M) |
|---|---|---|
| Task type | Generative text | Sequence classification |
| VRAM needed | 4–16 GB | Fits in 2 GB RAM / 250MB VRAM |
| Inference speed | 200–500ms/sample | ~2ms (GPU) / ~10ms (CPU) |
| Financial pre-training | Generic | Trained on 4.9B financial tokens |
| Fine-tuning time (CPU) | 10+ hours | ~25–50 minutes |

FinBERT was pre-trained specifically on Reuters, financial news, and SEC filings. For a classification task (positive/negative/neutral), an encoder is both faster and more accurate than a generative LLM.

---

## 3. Why Soup CLI Was Chosen

[Soup CLI](https://github.com/MakazhanAlpamys/Soup) is a declarative fine-tuning tool that eliminates training boilerplate. With a single YAML config and one command (`soup train`), it handles:

- Auto tokenization and data collation
- Train/validation split
- Learning rate scheduling with warmup
- Mixed precision (bf16/fp16) detection
- Checkpointing and model saving
- Live training dashboard

**The 20-line config that replaces ~300 lines of manual HuggingFace Trainer code:**

```yaml
base: ProsusAI/finbert
task: classifier
backend: transformers

data:
  train: ./data/train.jsonl
  format: auto
  max_length: 128
  val_split: 0.15

training:
  num_labels: 3
  classifier_kind: single_label
  label_names: [negative, neutral, positive]
  epochs: 2
  lr: 2e-5
  batch_size: 32
  lora:
    r: 0

output: ./output
```

---

## 4. The Structured Data vs. Sentiment Challenge

The project's original data inquiry evaluated **structured transactional business objects** (Sales Orders, Purchase Documents). After analysis, a critical issue was identified:

**Transactional data is purely numerical and has no sentiment.**

```json
{
  "sales_order": "4500012345",
  "material": "MAT-7890",
  "quantity": 500,
  "unit_price": 42.50,
  "net_value": 21250.00
}
```

**BERT tokenizes numbers as subword tokens — it does not compute math.** There is no sentiment in these records.

**Decision: Use established financial text datasets for training, and apply the model to generate sentiment analysis on financial text related to the business context (vendor news, earnings reports, market commentary, filings).**

---

## 5. Dataset Selection & Why

### 5.1 Why a Financial-Specific Dataset (Not General Sentiment)?

Generic sentiment datasets like SST-2 (movie reviews) or IMDB train models to recognize casual emotional language — *"this movie was terrible"*, *"I loved the acting"*. Financial sentiment is an entirely different domain:

- **Vocabulary is domain-specific:** Words like *"impairment"*, *"liquidity"*, *"covenant breach"*, *"basis points"*, *"EBITDA"*, *"short interest"* carry sentiment that general models cannot learn.
- **Sentiment polarity is context-dependent:** *"The company is cutting costs aggressively"* is **negative** to employees but **positive** to shareholders. A general model fails here.
- **FinBERT's pre-training advantage:** FinBERT was pre-trained on 4.9 billion tokens of Reuters financial news, SEC filings, and earnings transcripts. Fine-tuning it on a financial sentiment dataset teaches the classification boundary directly.

### 5.2 Why Twitter Financial News Sentiment Won

- **12,000 professionally annotated samples** (Bearish, Bullish, Neutral)
- **Real-world market language** — Covers earnings beats, guidance cuts, credit rating downgrades, macro events
- **Native Parquet loader** — Loads reliably without deprecated dataset loading scripts

---

## 6. Hardware Constraints & Hardware Scaling

Virdixt is architected to run across the entire hardware spectrum:

```
[ Tier 1: Intel Core i5 CPU (8GB RAM) ]  ───→ ONNX INT8 via AVX2 (~10-15ms)
[ Tier 2: RTX 3050 (6GB VRAM / 16GB RAM)] ───→ CUDA FP16 Provider (~1.5-2.5ms)
```

| Setting | Intel Core i5 (8GB) | NVIDIA RTX 3050 (6GB) |
|---|---|---|
| **Execution Provider** | `CPUExecutionProvider` (AVX2) | `CUDAExecutionProvider` |
| **FinBERT Latency** | ~10–15ms | **~1.5–2.5ms** |
| **Chart Vision (DePlot/Florence)**| ~150–250ms | **~25–40ms** |
| **Total Memory Footprint** | < 350 MB RAM | ~1.8 GB VRAM |
| **Batch Throughput** | 65 samples/sec | **420+ samples/sec** |

---

## 7. The 75% Problem — Root Cause Analysis

After the first training run on 1,500 unbalanced samples, validation accuracy was **75%** but **Negative Recall was only 9.5%**.

**Root cause: Severe class imbalance.** Neutral made up ~64% of rows, causing the model to default to Neutral and miss 90% of bankruptcy and default risks.

**The Fix:**
1. **Class-Weighted Loss** (`CrossEntropyLoss(weight=...)`)
2. **Balanced 6,300-row Master Dataset** (2,100 per class)
3. **Complex Concessive Sentence Injection** (*"Although revenue grew 14%, cash flow turned deeply negative"*)

**Result:** Negative recall jumped from **9.5% $\to$ 88.6%** and Macro-F1 reached **0.9022**.

---

## 8. Architecture Battle: Senior Architect vs. Ponytail

To find the optimal improvement strategy, we held a structured debate between two engineering philosophies:

### 🏛️ Senior Architect argued for:
- Full multimodal layout awareness (handling charts that hide bad news)
- C++ bare-metal inference with hardware abstraction layer (HAL)
- Asymmetric risk gates ($T=1.25$ temperature scaling) & strict typed contracts

### 💇 Ponytail countered:
- "No 70B LLMs, no 3-second LangChain agent chains"
- "DePlot token generation for math is a waste: use OCR + 3 lines of stdlib Python math (`delta = (b-a)/a`)"
- "Train once in Python, export to ONNX INT8, run in C++"

### 🏆 The Unified Synthesis:
- **Python Training Pipeline** (Soup CLI / PyTorch) for offline dataset balancing and fine-tuning.
- **Vision Delta Pipeline** for chart-to-concessive-text injection without visual hallucination.
- **Laya System-1 C++ Engine** for sub-10ms native execution and instant ERP routing.

---

## 9. Laya System-1 Primitives & Financial Advisor Engine

**Laya** (ConvAI Innovations, Apache 2.0) formalizes decision AI into three non-autoregressive mathematical primitives:

1. **`Choice` (Discrete Sentiment):** Selects one label (`[NEGATIVE, NEUTRAL, POSITIVE]`) with calibrated confidence.
2. **`Score` (Continuous Distress Index):** Maps latent representations to a continuous scalar $[0.0, 100.0]$ (0 = Peak Health, 100 = Imminent Insolvency).
3. **`Noul` (Propositional Risk Hypotheses):** Computes calibrated probabilities $P(\text{True})$ for targeted binary risk queries:
   * `liquidity_distress` $\rightarrow P(\text{True}) = 0.98$
   * `debt_covenant_breach_risk` $\rightarrow P(\text{True}) = 0.99$
   * `growth_expansion_momentum` $\rightarrow P(\text{True}) = 0.01$
   * `capital_return_sustainable` $\rightarrow P(\text{True}) = 0.05$

### Live Engine Output (`infer.py --test`):

```text
====================================================================================================
               VIRDIXT: LAYA SYSTEM-1 DECISION ENGINE & FINANCIAL ADVISOR
====================================================================================================

[TEST CASE 1: Multi-Clause Distress with Revenue Growth Mask]
  Text    : "The company experienced a severe decline in liquidity and breached its debt covenant,
             although revenue showed a slight 2% growth."
  Layer 1 : Anchor Token Pruning: 0.12 ms
  Layer 2 : FinBERT + Laya Primitives: ~10ms (ONNX)
  Output  :
    -> RISK GRADE     : CRITICAL (TIER_4_BLOCKED)
    -> POLICY ACTION  : FREEZE_PURCHASE_ORDERS
    -> ADVISOR ACTION : • IMMEDIATE: Freeze uncommitted purchase orders and discretionary capex.
                        • CREDIT: Require 100% upfront cash or irrevocable letters of credit.
                        • AUDIT: Request immediate debt covenant compliance certificate.

[TEST CASE 2: High Growth & Margin Expansion]
  Text    : "Organic ARR grew by 45% and EBITDA margin expanded significantly over the fiscal year."
  Output  :
    -> RISK GRADE     : MINIMAL (TIER_1_SAFE)
    -> POLICY ACTION  : PROCEED_NORMAL
    -> ADVISOR ACTION : • Counterparty healthy, proceed with standard commercial credit terms.
====================================================================================================
```

---

## 10. Multimodal Vision Pipeline

Corporate filings often bury bad news inside graphs while writing cheerful commentary. Virdixt bridges this gap deterministically:

```
[ Financial Document / PDF ] 
         ├── Text Stream  ───────────────→ [ Layer 1 Regex Pruner ]
         └── Image Stream (Charts/Graphs)  
                   │
                   ▼
         [ 1. Chart Detector (Florence-2) ]
                   │
                   ▼
         [ 2. Chart-to-Table Parser (DePlot) ]
         "Metric | Q1 | Q2\nRevenue | $45M | $48M\nMargin | 18% | 11%"
                   │
                   ▼
         [ 3. Deterministic Delta Calculator (Python/C++) ]
         Calculates: Revenue +6.7%, Margin -38.9%
                   │
                   ▼
         [ 4. Concessive Text Generator ]
         "Although Revenue grew 6.7%, Gross Margin declined 38.9%."
                   │
                   ▼
         [ Injected directly into FinBERT & Laya System-1 ]
```

---

## 11. High-Throughput C++ Runtime Engine

Located in [`cpp/`](file:///d:/Soup/cpp/), the native C++ engine provides zero-copy inference:

- **`include/laya_primitives.hpp`**: Header-only SIMD implementation of Temperature Softmax ($T=1.25$), continuous Distress Score $[0, 100]$, and Noul sigmoid propositions.
- **`include/inference_engine.hpp`**: ONNX Runtime C++ API wrapping Intel AVX2 and NVIDIA CUDA execution providers.
- **`include/text_preprocessor.hpp`**: Fast regex-based anchor token pruner.
- **`include/erp_advisor.hpp`**: Deterministic enterprise policy engine mapping risk grades to operational action flags.

---

## 12. Dataset Pipeline Design

The 6,300-row master dataset is assembled from 3 distinct sources:

```
┌────────────────────────────────────────────────────────┐
│              Master Dataset Builder                    │
├────────────────────────────────────────────────────────┤
│  Real Financial News         57.2%  (3,064 rows)       │
│  (Twitter Financial News Sentiment)                    │
│                                                        │
│  Synthetic Corporate Data    28.4%  (1,519 rows)       │
│  (synthetic_builder.py)                               │
│  • Balance-sheet statements, Impairments, Covenants    │
│                                                        │
│  Complex Multi-Clause        14.4%  (772 rows)         │
│  (complex_sentence_injector.py)                       │
│  • Concessive conjunctions ("Although X, Y")          │
│                                                        │
│  TOTAL: 6,300 rows │ 33.3% Neg / 33.4% Neu / 33.3% Pos│
└────────────────────────────────────────────────────────┘
```

---

## 13. Complex Sentence Injection

Financial disclosures often use concessive conjunctions (*"Although"*, *"Despite"*, *"Notwithstanding"*). Complex sentence injection teaches the model **financial hierarchy**:
* **Operating Profit > Revenue**
* **Forward Guidance > Past Quarter**
* **Cash Flow > Headline Earnings**

---

## 14. Final Results & Benchmarks

### Training Metrics (2 Epochs, FinBERT Backbone)

| Metric | Value |
|---|---|
| Validation Accuracy | **90.26%** |
| **Macro F1** | **0.9022** |
| Macro Precision | 0.9027 |
| Macro Recall | 0.9023 |
| **Negative Recall** | **88.64%** (up from 9.5%) |

### Confusion Matrix (945 Validation Samples)

```
                 Pred Negative   Pred Neutral   Pred Positive
Actual negative:     281 ✅           19             17
Actual neutral :      26            269 ✅           15
Actual positive:       7              8            303 ✅
```

---

## 15. Repository Structure

```text
Virdixt/
│
├── 📄 requirements.txt               # Dependencies
├── 📄 soup.yaml                      # Soup CLI training config
├── 📄 README.md                      # Complete system documentation
│
├── ── PYTHON DATA & TRAINING ───────────
├── 📄 prepare_data.py                # Real news downloader (Twitter Financial News)
├── 📄 synthetic_builder.py           # Accounting synthetic sentence generator
├── 📄 complex_sentence_injector.py   # Multi-clause adversarial sentence generator
├── 📄 build_rich_dataset.py          # Master dataset assembler (balanced 6,300 rows)
├── 📄 train.py                       # Standalone CPU/GPU FinBERT fine-tuning
├── 📄 eval.py                        # Validation & confusion matrix report
├── 📄 export_onnx.py                 # ONNX + INT8 quantization exporter
│
├── ── INFERENCE & VISION ───────────────
├── 📄 infer.py                       # Laya System-1 Engine (Choice, Score, Noul) + Decision Advisor
├── 📂 vision/
│   ├── 📄 chart_detector.py          # Florence-2 chart classifier
│   ├── 📄 chart_extractor.py         # DePlot chart-to-table parser
│   ├── 📄 delta_calculator.py        # Deterministic % math + concessive sentence generator
│   └── 📄 pipeline.py                # Full visual document orchestrator
│
├── ── C++ HIGH-THROUGHPUT RUNTIME ──────
├── 📂 cpp/
│   ├── 📄 CMakeLists.txt             # Build config (ONNX Runtime + CUDA)
│   ├── 📂 include/
│   │   ├── 📄 laya_primitives.hpp    # Laya System-1 math & probability calibration
│   │   ├── 📄 inference_engine.hpp   # ONNX Runtime C++ wrapper
│   │   ├── 📄 erp_advisor.hpp        # Deterministic ERP Policy Engine
│   │   └── 📄 text_preprocessor.hpp  # Fast regex signal sentence pruner
│   └── 📂 src/
│       └── 📄 main.cpp               # C++ test runner
│
└── ── DATASETS ─────────────────────────
    └── 📂 data/                      # train.jsonl (5,355 rows) & val.jsonl (945 rows)
```

---

## 16. How to Run

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Build the Balanced Dataset
```bash
python prepare_data.py
python synthetic_builder.py
python complex_sentence_injector.py
python build_rich_dataset.py
```

### Step 3: Fine-Tune the Model
```bash
# Standalone PyTorch:
python train.py

# OR via Soup CLI:
soup train --config soup.yaml
```

### Step 4: Run the Laya Decision Engine
```bash
python infer.py --test
```

### Step 5: Export to ONNX (for C++ runtime)
```bash
python export_onnx.py              # FP16 ONNX export
python export_onnx.py --quantize   # INT8 Quantized ONNX export
```

### Step 6: Build and Run C++ Engine
```bash
cd cpp
cmake -B build -DONNXRUNTIME_DIR=/path/to/onnxruntime
cmake --build build --config Release
./build/virdixt_engine
```

---

## License

MIT — Free for academic, open-source, and commercial use.
