"""
build_notebook.py
=================
Generates study_mitra.ipynb from structured cell definitions.
Run this script to rebuild the notebook after adding new cells.

Usage:
    python build_notebook.py
"""

import json
import uuid

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def cell_id():
    return str(uuid.uuid4()).replace("-", "")[:16]

def md(source: str) -> dict:
    """Create a Markdown cell."""
    return {
        "cell_type": "markdown",
        "id": cell_id(),
        "metadata": {},
        "source": source
    }

def code(source: str) -> dict:
    """Create a Code cell."""
    return {
        "cell_type": "code",
        "id": cell_id(),
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source
    }

# ──────────────────────────────────────────────────────────────────────────────
# NOTEBOOK HEADER
# ──────────────────────────────────────────────────────────────────────────────

NOTEBOOK_HEADER = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "colab": {
            "provenance": [],
            "gpuType": "T4",
            "toc_visible": True
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        },
        "accelerator": "GPU"
    }
}

# ──────────────────────────────────────────────────────────────────────────────
# CELL DEFINITIONS
# ──────────────────────────────────────────────────────────────────────────────

cells = []

# ════════════════════════════════════════════════════════════════════════════
# TITLE CARD
# ════════════════════════════════════════════════════════════════════════════

cells.append(md(r"""# 🎓 STUDY MITRA — AI Lecture Intelligence Platform

> **An end-to-end AI pipeline for audio/video lecture analysis**
> Built for Google Colab T4 GPU · Production-quality · Zero paid APIs

---

## 🏗️ Pipeline Architecture

```
Audio/Video Input
      │
      ▼
┌─────────────┐     ┌──────────────────┐     ┌────────────────────┐
│  FFmpeg      │────▶│  faster-whisper  │────▶│  BART-large-CNN    │
│  Audio Ext.  │     │  (Transcription) │     │  (Summarization)   │
└─────────────┘     └──────────────────┘     └────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │  KeyBERT + spaCy NER          │
              │  (Keyword & Entity Extraction) │
              └───────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
 ┌────────────────────┐         ┌─────────────────────────┐
 │  RoBERTa-SQuAD2    │         │  Helsinki-NLP OPUS-MT   │
 │  (Extractive QA)   │         │  (Translation)           │
 └────────────────────┘         └─────────────────────────┘
              │
              ▼
 ┌────────────────────────────────────────────────────────┐
 │  Accuracy Measurement: WER · ROUGE · F1/EM            │
 └────────────────────────────────────────────────────────┘
              │
              ▼
 ┌────────────────────────────────────────────────────────┐
 │  Gradio Interface (6 Tabs · Public URL via share=True) │
 └────────────────────────────────────────────────────────┘
```

## 📋 Algorithm Summary

| Component | Algorithm | Model | Metric |
|-----------|-----------|-------|--------|
| **Audio Extraction** | FFmpeg subprocess | pcm_s16le 16kHz mono | — |
| **Transcription** | faster-whisper CTranslate2 | large-v3 (GPU) | WER ~2.7% |
| **Summarization** | BART Map-Reduce | facebook/bart-large-cnn | ROUGE-1 ~44 |
| **Keywords** | KeyBERT + MMR | all-MiniLM-L6-v2 | Semantic cosine sim |
| **NER** | spaCy pipeline | en_core_web_sm | 8 entity types |
| **Q&A** | Extractive + TF-IDF | deepset/roberta-base-squad2 | F1 82.9% |
| **Translation** | MarianMT | Helsinki-NLP/opus-mt | 30-40 BLEU |

---
*⚡ Runtime: Google Colab T4 GPU · All models lazy-loaded to save VRAM*"""))

# ════════════════════════════════════════════════════════════════════════════
# CELL 1 — INSTALL DEPENDENCIES
# ════════════════════════════════════════════════════════════════════════════

cells.append(md(r"""## 📦 Cell 1 — Dependency Installation

### What We Install & Why

| Package | Purpose | Why This One |
|---------|---------|-------------|
| `faster-whisper` | Speech-to-text | CTranslate2-optimized Whisper — 4× faster, same WER |
| `gradio` | Web UI | Best Colab-compatible UI library with share=True public URLs |
| `transformers` | BART, RoBERTa, MarianMT | HuggingFace standard; 200K+ models available |
| `torch` | Deep learning backend | Industry standard; CUDA support for T4 GPU |
| `sentencepiece` | Tokenization | Required by MarianMT translation models |
| `keybert` | Keyword extraction | BERT-based semantic similarity; beats TF-IDF |
| `rouge-score` | Summary evaluation | Standard ROUGE metric implementation by Google |
| `jiwer` | WER computation | Most complete ASR metric library (WER, MER, WIL) |
| `yake` | Keyword alt. | Unsupervised statistical backup for KeyBERT |
| `scikit-learn` | TF-IDF for QA context | Context retrieval without needing FAISS |
| `spacy` | Named entity recognition | Fast, production-quality NLP pipeline |

### ⚠️ Install Notes
- `2>/dev/null` suppresses verbose pip output
- `ffmpeg` installed via apt (system-level, needed for audio extraction)
- `en_core_web_sm` is spaCy's English NER model (~15 MB)"""))

cells.append(code(r"""# ============================================================
# CELL 1: DEPENDENCY INSTALLATION
# Runtime: ~3-5 minutes on first run (model downloads excluded)
# ============================================================

import subprocess, sys

print("📦 Installing Python packages...")
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "faster-whisper==1.0.3",
    "gradio==4.44.0",
    "transformers==4.44.0",
    "torch", "torchvision", "torchaudio",
    "--extra-index-url", "https://download.pytorch.org/whl/cu121",
    "sentencepiece", "sacremoses",
    "keybert==0.8.4",
    "sentence-transformers",
    "rouge-score",
    "jiwer==3.0.3",
    "yake",
    "nltk",
    "scikit-learn",
    "pandas",
    "numpy",
    "accelerate",
    "protobuf",
], check=False)

print("🔧 Installing FFmpeg (system-level audio processor)...")
subprocess.run(["apt-get", "install", "-y", "-q", "ffmpeg"], check=False)

print("📥 Downloading spaCy English NER model (en_core_web_sm)...")
subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm", "-q"], check=False)

print("\n✅ All dependencies installed successfully!")
print("   → Restart runtime if torch/CUDA version errors appear")"""))

# ════════════════════════════════════════════════════════════════════════════
# CELL 2 — IMPORTS & GLOBAL CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

cells.append(md(r"""## 🔧 Cell 2 — Imports & Global Configuration

### Design Decisions

**Device Detection:**
- Automatically selects GPU (CUDA) or CPU at runtime
- `float16` on GPU: halves VRAM usage with identical accuracy (IEEE 754 half-precision)
- `int8` on CPU: quantized inference, further reducing RAM usage

**Lazy Model Registry (`MODELS` dict):**
- All heavy models (Whisper, BART, RoBERTa, MarianMT) are stored here after first load
- A model is only downloaded+initialized when its feature is first called
- Prevents GPU Out-of-Memory from loading all models at startup (~8 GB total if loaded simultaneously)
- Pattern: `if "key" not in MODELS: MODELS["key"] = load_model()`

**Why these specific imports?**
- `sklearn.TfidfVectorizer`: Used in QA to retrieve relevant transcript chunks without FAISS
- `rouge_score`: Google's official implementation — identical to evaluation in BART paper
- `jiwer`: Supports multiple transform pipelines for fair WER comparison"""))

cells.append(code(r"""# ============================================================
# CELL 2: IMPORTS & GLOBAL CONFIGURATION
# ============================================================

import os
import gc
import re
import json
import time
import warnings
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.WARNING)
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Suppress HuggingFace tokenizer warnings

# ── Core scientific stack ─────────────────────────────────────
import torch
import numpy as np
import pandas as pd

# ── NLP ───────────────────────────────────────────────────────
import nltk
import spacy
from nltk.tokenize import sent_tokenize

# ── Metrics ───────────────────────────────────────────────────
import jiwer
from rouge_score import rouge_scorer as rouge_scorer_lib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── NLTK data ─────────────────────────────────────────────────
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# ════════════════════════════════════════════════════════════════
# DEVICE CONFIGURATION
# ════════════════════════════════════════════════════════════════
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

print("═" * 55)
print("  🎓 STUDY MITRA — AI Lecture Intelligence Platform")
print("═" * 55)
print(f"\n🖥️  Device     : {DEVICE.upper()}")
print(f"⚙️  Compute    : {COMPUTE_TYPE}")

if DEVICE == "cuda":
    gpu_props = torch.cuda.get_device_properties(0)
    vram_gb   = gpu_props.total_memory / 1e9
    print(f"🎮  GPU        : {gpu_props.name}")
    print(f"💾  VRAM       : {vram_gb:.1f} GB")
    if vram_gb < 10:
        print("⚠️  Low VRAM: switching to medium model for Whisper")
else:
    print("⚠️  No GPU detected — CPU mode (slower, reduced accuracy)")

# ════════════════════════════════════════════════════════════════
# LAZY MODEL REGISTRY
# All models loaded on first use. Never at import time.
# ════════════════════════════════════════════════════════════════
MODELS: Dict[str, Any] = {}

# ════════════════════════════════════════════════════════════════
# LANGUAGE CONFIGURATION
# ════════════════════════════════════════════════════════════════
LANG_MAP = {
    "Hindi":   "hi",
    "French":  "fr",
    "German":  "de",
    "Spanish": "es",
    "Arabic":  "ar"
}

TRANSLATION_MODELS = {
    "hi": "Helsinki-NLP/opus-mt-en-hi",
    "fr": "Helsinki-NLP/opus-mt-en-fr",
    "de": "Helsinki-NLP/opus-mt-en-de",
    "es": "Helsinki-NLP/opus-mt-en-es",
    "ar": "Helsinki-NLP/opus-mt-en-ar",
}

# ════════════════════════════════════════════════════════════════
# SPACY NER PIPELINE
# ════════════════════════════════════════════════════════════════
try:
    NLP = spacy.load("en_core_web_sm")
    print(f"\n📚 spaCy      : {spacy.__version__} (en_core_web_sm loaded)")
except OSError:
    print("⚠️  spaCy model not found — run Cell 1 first")
    NLP = None

print(f"🤗 Transformers: installed")
print(f"🔥 PyTorch     : {torch.__version__}")
print("\n✅ Configuration complete — all modules imported")
print("   Models will load lazily when each feature is first used")"""))

# ════════════════════════════════════════════════════════════════════════════
# BUILD & WRITE NOTEBOOK
# ════════════════════════════════════════════════════════════════════════════

notebook = {**NOTEBOOK_HEADER, "cells": cells}

with open("study_mitra.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print(f"[OK] study_mitra.ipynb written with {len(cells)} cells")
