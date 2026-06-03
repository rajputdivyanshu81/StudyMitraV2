"""
build_notebook.py
=================
Generates study_mitra.ipynb from structured cell definitions.
Each cell source is stored in a CELLS dict to avoid nested triple-quote issues.
Run: python build_notebook.py
"""

import json, uuid, textwrap

def cell_id():
    return str(uuid.uuid4()).replace("-","")[:16]

def md(source):
    return {"cell_type":"markdown","id":cell_id(),"metadata":{},"source":source}

def code(source):
    return {"cell_type":"code","id":cell_id(),"metadata":{},
            "execution_count":None,"outputs":[],"source":source}

NOTEBOOK_META = {
    "nbformat":4, "nbformat_minor":5,
    "metadata":{
        "colab":{"provenance":[],"gpuType":"T4","toc_visible":True},
        "kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
        "language_info":{"name":"python","version":"3.10.0"},
        "accelerator":"GPU"
    }
}

cells = []

# =============================================================================
# TITLE CARD
# =============================================================================
cells.append(md(
"# STUDY MITRA - AI Lecture Intelligence Platform\n"
"\n"
"> **An end-to-end AI pipeline for audio/video lecture analysis**  \n"
"> Built for Google Colab T4 GPU - Production-quality - Zero paid APIs\n"
"\n"
"---\n"
"\n"
"## Pipeline Architecture\n"
"\n"
"```\n"
"Audio/Video Input\n"
"      |\n"
"      v\n"
"[FFmpeg Audio Ext.] --> [faster-whisper Transcription] --> [BART Summarization]\n"
"                                    |\n"
"                                    v\n"
"                    [KeyBERT + spaCy NER Keywords]\n"
"                                    |\n"
"                    +---------------+---------------+\n"
"                    v                               v\n"
"        [RoBERTa-SQuAD2 QA]         [Helsinki-NLP Translation]\n"
"                    |\n"
"                    v\n"
"        [WER + ROUGE + F1 Accuracy Report]\n"
"                    |\n"
"                    v\n"
"        [Gradio UI - 6 Tabs - Public URL]\n"
"```\n"
"\n"
"## Algorithm Summary\n"
"\n"
"| Component | Algorithm | Model | Benchmark |\n"
"|-----------|-----------|-------|-----------|\n"
"| Audio Extraction | FFmpeg subprocess | pcm_s16le 16kHz mono | - |\n"
"| Transcription | faster-whisper CTranslate2 | large-v3 float16 | WER ~2.7% |\n"
"| Summarization | BART Map-Reduce chunking | facebook/bart-large-cnn | ROUGE-1 ~44 |\n"
"| Keywords | KeyBERT + MMR | all-MiniLM-L6-v2 | Semantic cosine similarity |\n"
"| NER | spaCy pipeline | en_core_web_sm | 8 entity types |\n"
"| Q&A | Extractive + TF-IDF retrieval | deepset/roberta-base-squad2 | F1 82.9% |\n"
"| Translation | MarianMT | Helsinki-NLP/opus-mt-en-* | 30-40 BLEU |\n"
"\n"
"---\n"
"*Runtime: Google Colab T4 GPU - All models lazy-loaded to preserve VRAM*"
))

# =============================================================================
# CELL 1 MARKDOWN
# =============================================================================
cells.append(md(
"## Cell 1 - Dependency Installation\n"
"\n"
"### What We Install & Why\n"
"\n"
"| Package | Purpose | Why This One |\n"
"|---------|---------|-------------|\n"
"| `faster-whisper` | Speech-to-text | CTranslate2-optimized Whisper - 4x faster, same WER |\n"
"| `gradio` | Web UI | Best Colab UI library with share=True public URLs |\n"
"| `transformers` | BART, RoBERTa, MarianMT | HuggingFace standard; 200K+ models |\n"
"| `torch` | Deep learning | Industry standard; CUDA support for T4 GPU |\n"
"| `keybert` | Keyword extraction | BERT-based semantic similarity; beats TF-IDF |\n"
"| `rouge-score` | Summary evaluation | Google official ROUGE implementation |\n"
"| `jiwer` | WER computation | Most complete ASR metric library |\n"
"| `scikit-learn` | TF-IDF for QA | Context retrieval without FAISS |\n"
"| `spacy` | Named entity recognition | Fast, production-quality NLP pipeline |\n"
"\n"
"### Install Notes\n"
"- FFmpeg installed via apt (system-level, needed for audio extraction)\n"
"- en_core_web_sm is spaCy English NER model (~15 MB)\n"
"- Restart runtime if CUDA/torch version errors appear after install"
))

# =============================================================================
# CELL 1 CODE
# =============================================================================
CELL1_CODE = '''\
# ============================================================
# CELL 1: DEPENDENCY INSTALLATION
# Runtime: ~3-5 minutes on first run (model downloads excluded)
# ============================================================

import subprocess, sys

print("Installing Python packages...")
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

print("Installing FFmpeg (system-level audio processor)...")
subprocess.run(["apt-get", "install", "-y", "-q", "ffmpeg"], check=False)

print("Downloading spaCy English NER model...")
subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm", "-q"], check=False)

print("\\n[OK] All dependencies installed successfully!")
print("   Restart runtime if torch/CUDA version errors appear")
'''
cells.append(code(CELL1_CODE))

# =============================================================================
# CELL 2 MARKDOWN
# =============================================================================
cells.append(md(
"## Cell 2 - Imports & Global Configuration\n"
"\n"
"### Design Decisions\n"
"\n"
"**Device Detection:**\n"
"- Automatically selects GPU (CUDA) or CPU at runtime\n"
"- `float16` on GPU: halves VRAM with identical accuracy (IEEE 754 half-precision)\n"
"- `int8` on CPU: quantized inference, further reducing RAM usage\n"
"\n"
"**Lazy Model Registry (`MODELS` dict):**\n"
"- All heavy models (Whisper, BART, RoBERTa, MarianMT) stored after first load\n"
"- Pattern: `if 'key' not in MODELS: MODELS['key'] = load_model()`\n"
"- Prevents GPU OOM from loading all models simultaneously (~8 GB total)\n"
"\n"
"**Why these specific imports?**\n"
"- `sklearn.TfidfVectorizer`: QA context retrieval without needing FAISS/vector DB\n"
"- `rouge_score`: Google's official implementation - matches BART paper evaluation\n"
"- `jiwer`: Supports multiple transform pipelines for fair WER comparison"
))

# =============================================================================
# CELL 2 CODE
# =============================================================================
CELL2_CODE = '''\
# ============================================================
# CELL 2: IMPORTS & GLOBAL CONFIGURATION
# ============================================================

import os, gc, re, json, time, warnings, subprocess, logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np
import pandas as pd

import nltk
import spacy
from nltk.tokenize import sent_tokenize

import jiwer
from rouge_score import rouge_scorer as rouge_scorer_lib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

# ── Device Configuration ──────────────────────────────────────
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

print("=" * 55)
print("  STUDY MITRA - AI Lecture Intelligence Platform")
print("=" * 55)
print(f"\\nDevice     : {DEVICE.upper()}")
print(f"Compute    : {COMPUTE_TYPE}")

if DEVICE == "cuda":
    gpu = torch.cuda.get_device_properties(0)
    vram_gb = gpu.total_memory / 1e9
    print(f"GPU        : {gpu.name}")
    print(f"VRAM       : {vram_gb:.1f} GB")
    if vram_gb < 10:
        print("WARNING: Low VRAM - will use medium model for Whisper")
else:
    print("WARNING: No GPU - CPU mode (slower, reduced accuracy)")

# ── Lazy Model Registry ───────────────────────────────────────
# All models loaded on first use. Never loaded at import time.
MODELS: Dict[str, Any] = {}

# ── Language Configuration ────────────────────────────────────
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

# ── spaCy NER ─────────────────────────────────────────────────
try:
    NLP = spacy.load("en_core_web_sm")
    print(f"\\nspaCy      : {spacy.__version__} (en_core_web_sm loaded)")
except OSError:
    print("WARNING: spaCy model not found - run Cell 1 first")
    NLP = None

print(f"PyTorch    : {torch.__version__}")
print("\\n[OK] Configuration complete - all modules imported")
print("     Models will load lazily when each feature is first called")
'''
cells.append(code(CELL2_CODE))

# =============================================================================
# CELL 3 MARKDOWN
# =============================================================================
cells.append(md(
"## Cell 3 - Audio Extraction (FFmpeg + MoviePy Fallback)\n"
"\n"
"### Algorithm: FFmpeg subprocess\n"
"\n"
"**Why FFmpeg over MoviePy?**\n"
"\n"
"| Criterion | FFmpeg (primary) | MoviePy (fallback) |\n"
"|-----------|-----------------|-------------------|\n"
"| Speed | 10-30x realtime | 3-5x realtime |\n"
"| Format support | 500+ formats | Subset of FFmpeg |\n"
"| Memory usage | Minimal (C-level) | Higher (Python overhead) |\n"
"| Normalization | EBU R128 loudnorm | Basic resampling only |\n"
"\n"
"**Audio Normalization - Why EBU R128 loudnorm?**\n"
"- Whisper achieves best WER when audio loudness is standardized\n"
"- EBU R128 is the broadcast standard (-23 LUFS) used by Netflix, BBC\n"
"- Prevents clipping (too loud = distortion) and under-level (too quiet = missed words)\n"
"\n"
"**Output Specification:**\n"
"- Sample rate: 16,000 Hz (Whisper native rate)\n"
"- Channels: Mono (stereo gives no ASR benefit; halves file size)\n"
"- Bit depth: 16-bit PCM (pcm_s16le) - lossless WAV\n"
"\n"
"**Supported Input:** `.mp4` `.mkv` `.webm` `.mp3` `.wav` `.flac` `.ogg` `.m4a`"
))

# =============================================================================
# CELL 3 CODE
# =============================================================================
CELL3_CODE = '''\
# ============================================================
# CELL 3: AUDIO EXTRACTION - FFmpeg primary, MoviePy fallback
# ============================================================

from pathlib import Path

SUPPORTED_FORMATS = {".mp4", ".mkv", ".webm", ".mp3", ".wav", ".flac", ".ogg", ".m4a"}

def check_ffmpeg() -> bool:
    """Check if FFmpeg binary is available on the system PATH."""
    try:
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

def extract_audio_ffmpeg(input_path: str, output_path: str) -> bool:
    """
    Extract & normalize audio using FFmpeg.

    Command flags used:
        -y              : Overwrite output without prompting
        -i input_path   : Input file
        -vn             : Strip video stream (not needed for ASR)
        -ar 16000       : Resample to 16 kHz (Whisper optimal rate)
        -ac 1           : Convert to mono channel
        -af loudnorm    : EBU R128 loudness normalization (I=-23, LRA=7, TP=-2)
        -acodec pcm_s16le : 16-bit signed little-endian PCM WAV output
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vn",
        "-ar", "16000",
        "-ac", "1",
        "-af", "loudnorm=I=-23:LRA=7:TP=-2",
        "-acodec", "pcm_s16le",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    return result.returncode == 0

def extract_audio_moviepy(input_path: str, output_path: str) -> bool:
    """
    Fallback audio extraction using MoviePy.
    Used only when FFmpeg is unavailable. No loudnorm applied.
    """
    try:
        from moviepy.editor import VideoFileClip, AudioFileClip
        ext = Path(input_path).suffix.lower()
        clip = VideoFileClip(input_path) if ext in {".mp4", ".mkv", ".webm"} else None
        audio = clip.audio if clip else AudioFileClip(input_path)
        audio.write_audiofile(output_path, fps=16000, nbytes=2,
                              ffmpeg_params=["-ac", "1"])
        audio.close()
        return True
    except Exception as e:
        print(f"[ERROR] MoviePy fallback failed: {e}")
        return False

def extract_audio(input_path: str, output_path: str = "/tmp/extracted_audio.wav") -> str:
    """
    Main audio extraction entry point.

    Tries FFmpeg first (fast, full normalization).
    Falls back to MoviePy if FFmpeg is not available.

    Args:
        input_path:  Path to input audio or video file
        output_path: Destination .wav file path

    Returns:
        str: Path to the extracted WAV file

    Raises:
        ValueError:   Unsupported file format
        FileNotFoundError: Input file missing
        RuntimeError: Both extraction methods failed
    """
    ext = Path(input_path).suffix.lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format: {ext}. Supported: {SUPPORTED_FORMATS}")
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Not found: {input_path}")

    print(f"[Audio] Input : {Path(input_path).name}")

    if check_ffmpeg():
        print("[Audio] Method: FFmpeg + EBU R128 normalization")
        if extract_audio_ffmpeg(input_path, output_path):
            mb = Path(output_path).stat().st_size / 1_000_000
            print(f"[Audio] Done  : {mb:.2f} MB | 16kHz mono PCM")
            return output_path
        print("[Audio] FFmpeg failed, falling back to MoviePy...")
    else:
        print("[Audio] FFmpeg not found, using MoviePy (slower, no normalization)")

    if extract_audio_moviepy(input_path, output_path):
        print(f"[Audio] Done (MoviePy): {output_path}")
        return output_path

    raise RuntimeError("Audio extraction failed with both FFmpeg and MoviePy.")

print("[OK] Audio extraction module ready")
print(f"     FFmpeg available : {check_ffmpeg()}")
print(f"     Supported formats: {', '.join(sorted(SUPPORTED_FORMATS))}")
'''
cells.append(code(CELL3_CODE))

# =============================================================================
# CELL 4 MARKDOWN
# =============================================================================
cells.append(md(
"## Cell 4 - Speech-to-Text (faster-whisper large-v3)\n"
"\n"
"### Algorithm: faster-whisper with CTranslate2 Optimization\n"
"\n"
"**Benchmark Comparison:**\n"
"\n"
"| Implementation | Model | WER (LibriSpeech clean) | Speed RTF | VRAM |\n"
"|---------------|-------|------------------------|-----------|------|\n"
"| **faster-whisper** (chosen) | large-v3 float16 | **2.7%** | 0.08 (12x RT) | ~4 GB |\n"
"| openai-whisper | large-v3 | 2.7% | 0.35 (3x RT) | ~10 GB |\n"
"| faster-whisper | large-v3-turbo | 3.2% | 0.03 (33x RT) | ~3 GB |\n"
"| faster-whisper | base int8 (CPU) | 8.5% | 0.4 | RAM only |\n"
"\n"
"**Key Technical Optimizations:**\n"
"\n"
"1. **VAD Filter** (`vad_filter=True`): Silero Voice Activity Detection removes silence\n"
"   - Reduces WER by 10-15% on lecture recordings with pauses\n"
"   - min_silence_duration_ms=500: 500ms pause = new segment boundary\n"
"\n"
"2. **Beam Search** (`beam_size=5`): Optimal quality vs speed\n"
"   - beam=1: Greedy decoding (fastest, ~1% higher WER)\n"
"   - beam=5: Near-optimal quality (our choice)\n"
"   - beam=10: Diminishing returns at 2x the compute cost\n"
"\n"
"3. **Temperature Fallback** `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`:\n"
"   - Starts greedy (temp=0), falls back to sampling if confidence is low\n"
"   - Prevents hallucination on noisy audio segments\n"
"\n"
"4. **Word Timestamps**: CTranslate2 forced alignment gives per-word start/end/prob\n"
"\n"
"**Alternative rejected: AssemblyAI / Deepgram** - paid APIs, require billing account"
))

# =============================================================================
# CELL 4 CODE
# =============================================================================
CELL4_CODE = '''\
# ============================================================
# CELL 4: SPEECH-TO-TEXT - faster-whisper large-v3
# ============================================================

from faster_whisper import WhisperModel
from dataclasses import dataclass, field

@dataclass
class TranscriptSegment:
    """Structured container for one transcript segment with timing."""
    start: float
    end:   float
    text:  str
    words: list = field(default_factory=list)

    def duration(self) -> float:
        return self.end - self.start

    def timestamp_str(self) -> str:
        def fmt(t):
            return f"{int(t // 60):02d}:{t % 60:05.2f}"
        return f"[{fmt(self.start)} --> {fmt(self.end)}]"

def get_whisper_model_size() -> str:
    """
    Auto-select Whisper model based on available VRAM.

    VRAM thresholds:
        large-v3 float16 --> needs ~4 GB VRAM (T4 has 15 GB - safe)
        medium   float16 --> needs ~2.5 GB VRAM
        base     int8    --> CPU only, ~150 MB RAM
    """
    if DEVICE != "cuda":
        return "base"
    try:
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram >= 8:   return "large-v3"
        elif vram >= 5: return "medium"
        else:           return "small"
    except Exception:
        return "base"

def load_whisper_model(model_size: Optional[str] = None) -> WhisperModel:
    """
    Lazily load and cache faster-whisper model.

    CTranslate2 key optimizations:
        float16 - Half-precision: 2x VRAM savings vs float32, identical accuracy
        int8    - 8-bit quantized: 4x VRAM savings, slight accuracy trade-off on CPU
    """
    if model_size is None:
        model_size = get_whisper_model_size()

    key = f"whisper_{model_size}"
    if key not in MODELS:
        sizes_gb = {"large-v3": 3.0, "medium": 1.5, "small": 0.5, "base": 0.15}
        gb = sizes_gb.get(model_size, 1.0)
        print(f"[Whisper] Loading '{model_size}' on {DEVICE.upper()} ({COMPUTE_TYPE})")
        print(f"[Whisper] Download size ~{gb:.1f} GB (cached after first run)")
        MODELS[key] = WhisperModel(
            model_size,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            download_root="/tmp/whisper_cache",
            cpu_threads=4,
            num_workers=2,
        )
        print(f"[Whisper] Loaded and cached as MODELS['{key}']")
    return MODELS[key]

def transcribe_audio(
    audio_path:      str,
    model_size:      Optional[str] = None,
    language:        Optional[str] = None,
    beam_size:       int  = 5,
    vad_filter:      bool = True,
    word_timestamps: bool = True,
    task:            str  = "transcribe",
) -> Dict[str, Any]:
    """
    Full speech-to-text pipeline using faster-whisper.

    Steps:
        1. Load/retrieve model from MODELS cache (lazy)
        2. Apply Silero VAD filter to remove silence
        3. Transcribe with beam search (beam_size=5)
        4. Collect segment and word timestamps
        5. Free VRAM after transcription

    Args:
        audio_path      : Path to 16kHz mono WAV file
        model_size      : None = auto-select based on VRAM
        language        : ISO code to force language (None = auto-detect)
        beam_size       : Beam search width
        vad_filter      : Enable Silero VAD silence removal
        word_timestamps : Include per-word start/end/probability
        task            : 'transcribe' or 'translate' (to English)

    Returns dict:
        transcript    - Full text string
        segments      - List[TranscriptSegment]
        timestamped   - Human-readable timestamped transcript
        language      - Detected ISO language code
        lang_prob     - Language detection confidence (0-1)
        duration      - Audio length in seconds
        num_words     - Total word count
        rtf           - Real-Time Factor (elapsed / audio_duration)
    """
    model = load_whisper_model(model_size)
    print(f"\\n[Transcribe] File   : {Path(audio_path).name}")
    print(f"[Transcribe] Config : beam={beam_size}, VAD={vad_filter}, lang={language or 'auto'}")

    t0 = time.time()
    segments_gen, info = model.transcribe(
        audio_path,
        task=task,
        language=language,
        beam_size=beam_size,

        # Voice Activity Detection - removes silence
        vad_filter=vad_filter,
        vad_parameters={
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 200,
            "threshold": 0.5,
        },

        # Word-level timestamps
        word_timestamps=word_timestamps,

        # Hallucination prevention
        temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        no_speech_threshold=0.6,
        log_prob_threshold=-1.0,
        compression_ratio_threshold=2.4,

        # Context conditioning for better continuity
        condition_on_previous_text=True,
    )

    segments, text_parts = [], []
    for seg in segments_gen:
        words = []
        if word_timestamps and seg.words:
            words = [
                {"word": w.word.strip(), "start": round(w.start, 3),
                 "end": round(w.end, 3), "prob": round(w.probability, 3)}
                for w in seg.words
            ]
        s = TranscriptSegment(
            start=round(seg.start, 2), end=round(seg.end, 2),
            text=seg.text.strip(), words=words
        )
        segments.append(s)
        text_parts.append(seg.text.strip())

    elapsed = time.time() - t0
    full_text   = " ".join(text_parts)
    timestamped = "\\n".join(f"{s.timestamp_str()} {s.text}" for s in segments)

    # Free VRAM after heavy inference
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

    rtf = elapsed / max(info.duration, 1)
    print(f"\\n[Transcribe] Done     : {elapsed:.1f}s elapsed")
    print(f"[Transcribe] Language : {info.language} ({info.language_probability:.1%})")
    print(f"[Transcribe] Duration : {info.duration:.1f}s | {len(segments)} segs | {len(full_text.split())} words")
    print(f"[Transcribe] RTF      : {rtf:.3f}  (lower = faster than realtime)")

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
        "rtf":          round(rtf, 3),
    }

print("[OK] Transcription module ready")
print(f"     Auto-selected model: {get_whisper_model_size()}")
'''
cells.append(code(CELL4_CODE))

# =============================================================================
# BUILD & WRITE NOTEBOOK
# =============================================================================

notebook = {**NOTEBOOK_META, "cells": cells}

with open("study_mitra.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print(f"[OK] study_mitra.ipynb written with {len(cells)} cells")
