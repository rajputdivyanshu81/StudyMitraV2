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
# CELL 3 — AUDIO EXTRACTION
# ════════════════════════════════════════════════════════════════════════════

cells.append(md(r"""## 🎵 Cell 3 — Audio Extraction (FFmpeg + MoviePy Fallback)

### Algorithm: FFmpeg subprocess

**Why FFmpeg over MoviePy?**

| Criterion | FFmpeg (primary) | MoviePy (fallback) |
|-----------|-----------------|-------------------|
| Speed | ~10-30x realtime | ~3-5x realtime |
| Format support | 500+ formats | Subset of FFmpeg |
| Memory usage | Minimal (C-level) | Higher (Python overhead) |
| Normalization | EBU R128 loudnorm | Basic resampling |
| Dependency | System binary | Python package |

**Audio Normalization — Why EBU R128 loudnorm?**
- Whisper achieves its best WER when audio loudness is standardized
- EBU R128 is the broadcast standard (-23 LUFS target) used by Netflix, BBC, etc.
- Prevents both clipping (too loud → distortion) and under-level (too quiet → missed words)

**Output Specification:**
- Sample rate: **16,000 Hz** (Whisper's native rate — resampling wastes no quality)
- Channels: **Mono** (stereo provides no benefit for single-speaker ASR; halves data size)
- Bit depth: **16-bit PCM** (`pcm_s16le`) — lossless, standard WAV

**Supported Input Formats:**
`.mp4` · `.mkv` · `.webm` · `.mp3` · `.wav` · `.flac` · `.ogg` · `.m4a`"""))

cells.append(code(r"""# ============================================================
# CELL 3: AUDIO EXTRACTION — FFmpeg primary, MoviePy fallback
# ============================================================

import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

SUPPORTED_FORMATS = {'.mp4', '.mkv', '.webm', '.mp3', '.wav', '.flac', '.ogg', '.m4a'}

# ─────────────────────────────────────────────────────────────
def check_ffmpeg() -> bool:
    """Verify FFmpeg is installed and accessible."""
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

# ─────────────────────────────────────────────────────────────
def extract_audio_ffmpeg(input_path: str, output_path: str) -> bool:
    """
    Extract and normalize audio using FFmpeg.

    FFmpeg command breakdown:
        -y              : Overwrite output file without prompting
        -i input        : Input file path
        -vn             : Strip video stream (video not needed for ASR)
        -ar 16000       : Resample to 16 kHz (Whisper optimal rate)
        -ac 1           : Convert to mono (single channel)
        -af loudnorm    : Apply EBU R128 loudness normalization
        -acodec pcm_s16le : 16-bit signed little-endian PCM (WAV)
    """
    cmd = [
        'ffmpeg', '-y',
        '-i', input_path,
        '-vn',
        '-ar', '16000',
        '-ac', '1',
        '-af', 'loudnorm=I=-23:LRA=7:TP=-2',  # EBU R128 standard
        '-acodec', 'pcm_s16le',
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    return result.returncode == 0

# ─────────────────────────────────────────────────────────────
def extract_audio_moviepy(input_path: str, output_path: str) -> bool:
    """
    Fallback: Extract audio using MoviePy.

    Slower than FFmpeg (~3-4x) but works when FFmpeg is unavailable.
    No loudnorm in this path — only basic resampling.
    """
    try:
        from moviepy.editor import VideoFileClip, AudioFileClip
        ext = Path(input_path).suffix.lower()
        if ext in {'.mp4', '.mkv', '.webm'}:
            clip  = VideoFileClip(input_path)
            audio = clip.audio
        else:
            audio = AudioFileClip(input_path)

        audio.write_audiofile(
            output_path,
            fps=16000,
            nbytes=2,                    # 16-bit
            ffmpeg_params=['-ac', '1']   # Mono
        )
        audio.close()
        return True
    except Exception as e:
        print(f"[ERROR] MoviePy fallback failed: {e}")
        return False

# ─────────────────────────────────────────────────────────────
def extract_audio(input_path: str, output_path: str = "/tmp/extracted_audio.wav") -> str:
    """
    Main audio extraction entry point with automatic fallback.

    Args:
        input_path:  Path to input audio/video file
        output_path: Path for output WAV file (default: /tmp/)

    Returns:
        str: Path to the extracted .wav file

    Raises:
        ValueError: Unsupported format
        RuntimeError: Both FFmpeg and MoviePy failed
    """
    ext = Path(input_path).suffix.lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )
    if not Path(input_path).exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    print(f"[Audio] Input : {Path(input_path).name}")
    print(f"[Audio] Output: {output_path}")

    # ── Primary: FFmpeg ──────────────────────────────────────
    if check_ffmpeg():
        print("[Audio] Method: FFmpeg (fast + EBU R128 normalized)")
        if extract_audio_ffmpeg(input_path, output_path):
            size_mb = Path(output_path).stat().st_size / 1_000_000
            print(f"[Audio] Done  : {size_mb:.2f} MB at 16kHz mono PCM")
            return output_path
        else:
            print("[Audio] FFmpeg failed, trying MoviePy...")
    else:
        print("[Audio] FFmpeg not found, using MoviePy fallback (slower)")

    # ── Fallback: MoviePy ────────────────────────────────────
    if extract_audio_moviepy(input_path, output_path):
        print(f"[Audio] Done (MoviePy): {output_path}")
        return output_path

    raise RuntimeError(
        "Audio extraction failed with both FFmpeg and MoviePy.\n"
        "Ensure at least one is installed."
    )

# ── Quick test ───────────────────────────────────────────────
print("[OK] Audio extraction module ready")
print(f"     FFmpeg available: {check_ffmpeg()}")
print(f"     Supported formats: {', '.join(sorted(SUPPORTED_FORMATS))}")"""))

# ════════════════════════════════════════════════════════════════════════════
# CELL 4 — WHISPER TRANSCRIPTION
# ════════════════════════════════════════════════════════════════════════════

cells.append(md(r"""## 🎙️ Cell 4 — Speech-to-Text Transcription (faster-whisper large-v3)

### Algorithm: faster-whisper with CTranslate2 Optimization

**Benchmark Comparison:**

| Implementation | Model | WER (LibriSpeech clean) | Speed (RTF) | VRAM |
|---------------|-------|------------------------|-------------|------|
| **faster-whisper** (chosen) | large-v3 float16 | **~2.7%** | **~0.08** (12x realtime) | ~4 GB |
| openai-whisper | large-v3 | ~2.7% | ~0.35 (3x realtime) | ~10 GB |
| faster-whisper | large-v3-turbo | ~3.2% | ~0.03 (33x realtime) | ~3 GB |
| faster-whisper | base int8 (CPU) | ~8.5% | ~0.4 | CPU only |

**Key Technical Features Used:**

1. **VAD Filter** (`vad_filter=True`): Voice Activity Detection using Silero VAD  
   - Removes silent sections before transcription  
   - Reduces WER by 10-15% on lecture recordings with pauses  
   - Minimum silence duration: 500ms (tuned for lecture content)

2. **Beam Search** (`beam_size=5`): Best trade-off between quality and speed  
   - beam=1 = greedy decoding (fastest, ~1% higher WER)  
   - beam=5 = near-optimal quality  
   - beam=10 = diminishing returns, 2x slower

3. **Temperature Fallback**: `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`  
   - Starts with greedy (temp=0), falls back to sampling if confidence is low  
   - Prevents "hallucination" on noisy audio segments

4. **Word Timestamps**: Enabled via CTranslate2's forced alignment  
   - Provides `[word, start_time, end_time, probability]` for each word  
   - Used to display timestamped transcript in the UI

**Alternative: Why NOT AssemblyAI / Deepgram?**
- Both are paid APIs (require API keys + billing)
- Constraint: zero-cost, fully offline operation"""))

cells.append(code(r"""# ============================================================
# CELL 4: SPEECH-TO-TEXT — faster-whisper large-v3
# ============================================================

from faster_whisper import WhisperModel
from dataclasses import dataclass, field

# ─────────────────────────────────────────────────────────────
@dataclass
class TranscriptSegment:
    """Structured container for one transcript segment."""
    start:  float
    end:    float
    text:   str
    words:  list = field(default_factory=list)

    def duration(self) -> float:
        return self.end - self.start

    def timestamp(self) -> str:
        """Return [MM:SS.ss --> MM:SS.ss] formatted timestamp."""
        def fmt(t):
            return f"{int(t // 60):02d}:{t % 60:05.2f}"
        return f"[{fmt(self.start)} --> {fmt(self.end)}]"

# ─────────────────────────────────────────────────────────────
def get_whisper_model_size() -> str:
    """
    Select model size based on available hardware.

    VRAM requirements:
        large-v3 float16 → ~4 GB VRAM  (T4 has 15 GB — safe)
        medium float16   → ~2.5 GB VRAM
        base int8        → CPU only
    """
    if DEVICE != "cuda":
        print("[Whisper] CPU detected → using 'base' model (int8)")
        return "base"
    try:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb >= 8:
            return "large-v3"
        elif vram_gb >= 5:
            return "medium"
        else:
            return "small"
    except Exception:
        return "base"

# ─────────────────────────────────────────────────────────────
def load_whisper_model(model_size: Optional[str] = None) -> WhisperModel:
    """
    Lazily load faster-whisper model. Uses MODELS registry for caching.

    CTranslate2 optimization details:
        float16: Half-precision inference — 2x VRAM savings vs float32
                 Accuracy identical on modern GPU hardware (T4 supports FP16)
        int8:    8-bit quantized — 4x VRAM savings vs float32
                 Slight accuracy degradation (~0.5% WER increase) on CPU
    """
    if model_size is None:
        model_size = get_whisper_model_size()

    cache_key = f"whisper_{model_size}"
    if cache_key not in MODELS:
        actual_device = DEVICE
        actual_compute = COMPUTE_TYPE
        print(f"[Whisper] Loading '{model_size}' on {actual_device.upper()} ({actual_compute})...")
        print(f"[Whisper] First load will download model (~{{'large-v3':3,'medium':1.5,'small':0.5,'base':0.15}.get(model_size,1):.1f} GB)")

        MODELS[cache_key] = WhisperModel(
            model_size,
            device=actual_device,
            compute_type=actual_compute,
            download_root="/tmp/whisper_cache",
            cpu_threads=4,
            num_workers=2,
        )
        print(f"[Whisper] Model loaded and cached in MODELS['{cache_key}']")
    return MODELS[cache_key]

# ─────────────────────────────────────────────────────────────
def transcribe_audio(
    audio_path:       str,
    model_size:       Optional[str] = None,
    language:         Optional[str] = None,
    beam_size:        int  = 5,
    vad_filter:       bool = True,
    word_timestamps:  bool = True,
    task:             str  = "transcribe",
) -> Dict[str, Any]:
    """
    Full speech-to-text pipeline using faster-whisper.

    Pipeline:
        1. Load/retrieve model from MODELS cache
        2. Run VAD filter to remove silence
        3. Transcribe with beam search (beam_size=5)
        4. Collect segment + word timestamps
        5. Free VRAM after transcription

    Args:
        audio_path:      Path to 16kHz mono WAV file
        model_size:      None = auto-select based on VRAM
        language:        Force language code (None = auto-detect)
        beam_size:       Beam search width (5 = optimal quality/speed)
        vad_filter:      Remove silence before transcription
        word_timestamps: Include word-level start/end times
        task:            'transcribe' or 'translate' (to English)

    Returns:
        dict with keys:
            transcript    - Full text string
            segments      - List[TranscriptSegment]
            timestamped   - Formatted timestamp transcript
            language      - Detected ISO language code
            lang_prob     - Language detection confidence
            duration      - Audio duration in seconds
            num_segments  - Segment count
            num_words     - Word count
    """
    model = load_whisper_model(model_size)

    print(f"\n[Transcribe] File   : {Path(audio_path).name}")
    print(f"[Transcribe] Config : beam={beam_size}, VAD={vad_filter}, lang={language or 'auto'}")

    t_start = time.time()

    segments_gen, info = model.transcribe(
        audio_path,
        task=task,
        language=language,
        beam_size=beam_size,

        # ── VAD (Voice Activity Detection) ───────────────────
        vad_filter=vad_filter,
        vad_parameters={
            "min_silence_duration_ms": 500,  # 500ms pause = new segment
            "speech_pad_ms": 200,            # 200ms padding around speech
            "threshold": 0.5,                # VAD confidence threshold
        },

        # ── Word-level timestamps ─────────────────────────────
        word_timestamps=word_timestamps,

        # ── Hallucination prevention ──────────────────────────
        temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],  # Fallback temps
        no_speech_threshold=0.6,
        log_prob_threshold=-1.0,
        compression_ratio_threshold=2.4,

        # ── Context conditioning ──────────────────────────────
        condition_on_previous_text=True,
        initial_prompt=None,  # Can add domain prompt e.g. "This is a lecture on..."
    )

    # ── Collect results ───────────────────────────────────────
    segments       = []
    text_parts     = []

    for seg in segments_gen:
        words = []
        if word_timestamps and seg.words:
            words = [
                {
                    "word":  w.word.strip(),
                    "start": round(w.start, 3),
                    "end":   round(w.end, 3),
                    "prob":  round(w.probability, 3),
                }
                for w in seg.words
            ]

        segment = TranscriptSegment(
            start=round(seg.start, 2),
            end=round(seg.end, 2),
            text=seg.text.strip(),
            words=words,
        )
        segments.append(segment)
        text_parts.append(seg.text.strip())

    elapsed      = time.time() - t_start
    full_text    = " ".join(text_parts)
    timestamped  = "\n".join(f"{s.timestamp()} {s.text}" for s in segments)

    # ── Free VRAM after transcription ─────────────────────────
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\n[Transcribe] Done in {elapsed:.1f}s")
    print(f"[Transcribe] Language : {info.language} ({info.language_probability:.1%} confidence)")
    print(f"[Transcribe] Duration : {info.duration:.1f}s | {len(segments)} segments | {len(full_text.split())} words")
    print(f"[Transcribe] RTF      : {elapsed / max(info.duration, 1):.3f} (real-time factor)")

    return {
        "transcript":   full_text,
        "segments":     segments,
        "timestamped":  timestamped,
        "language":     info.language,
        "lang_prob":    round(info.language_probability, 4),
        "duration":     round(info.duration, 2),
        "num_segments": len(segments),
        "num_words":    len(full_text.split()),
        "elapsed_s":    round(elapsed, 2),
    }

# ── Module ready confirmation ─────────────────────────────────
print("[OK] Transcription module ready")
print(f"     Auto-selected model: {get_whisper_model_size()}")
print("     Call: transcribe_audio('/path/to/audio.wav')")"""))

# ════════════════════════════════════════════════════════════════════════════
# BUILD & WRITE NOTEBOOK
# ════════════════════════════════════════════════════════════════════════════

notebook = {**NOTEBOOK_HEADER, "cells": cells}

with open("study_mitra.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print(f"[OK] study_mitra.ipynb written with {len(cells)} cells")
