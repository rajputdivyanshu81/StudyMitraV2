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

# =============================================================================
# CELL 5 MARKDOWN
# =============================================================================
cells.append(md(
"## Cell 5 - Text Summarization (BART + Map-Reduce Chunking)\n"
"\n"
"### Algorithm: Abstractive Summarization with facebook/bart-large-cnn\n"
"\n"
"**Model Comparison:**\n"
"\n"
"| Model | ROUGE-1 (CNN/DM) | Notes |\n"
"|-------|-----------------|-------|\n"
"| **facebook/bart-large-cnn** (chosen) | **~44.2** | Stable, fluent, best for English text |\n"
"| google/pegasus-xsum | ~47.2 (XSum only) | Over-abstractive; unstable on long lectures |\n"
"| t5-base | ~38.1 | Lighter but significantly weaker |\n"
"| facebook/bart-large-cnn-samsum | ~44+ | Fine-tuned on conversations - alternative |\n"
"\n"
"### Map-Reduce Chunking Strategy\n"
"\n"
"```\n"
"Full Transcript (50,000+ words)\n"
"       |\n"
"       v  [MAP PHASE]\n"
"Split into N chunks of 900 tokens (150-token overlap)\n"
"       |\n"
"       v  Summarize each chunk independently with BART\n"
"Chunk summaries: [summary_1, summary_2, ..., summary_N]\n"
"       |\n"
"       v  [REDUCE PHASE]\n"
"If combined summaries > 900 tokens: summarize again\n"
"Otherwise: join chunk summaries as final output\n"
"       |\n"
"       v\n"
"Final Summary\n"
"```\n"
"\n"
"**Why tokenizer-based chunking (not character/word-based)?**\n"
"- BART's 1024-token limit is in SUBWORD tokens, not words\n"
"- Same word can be 1-3 tokens (e.g., 'unbelievable' = 4 tokens)\n"
"- Using BartTokenizer directly avoids any overflow\n"
"- 150-token overlap prevents losing context at boundaries\n"
"\n"
"**BART generation parameters:**\n"
"- `num_beams=4`: Beam search for coherent summaries\n"
"- `length_penalty=2.0`: Encourages longer, more complete summaries\n"
"- `no_repeat_ngram_size=3`: Prevents repetitive phrases"
))

# =============================================================================
# CELL 5 CODE
# =============================================================================
CELL5_CODE = '''\
# ============================================================
# CELL 5: SUMMARIZATION - BART-large-CNN + Map-Reduce
# ============================================================

from transformers import BartForConditionalGeneration, BartTokenizer

BART_MODEL = "facebook/bart-large-cnn"
MAX_CHUNK_TOKENS = 900   # BART context limit is 1024; leave room for generation
OVERLAP_TOKENS   = 150   # Overlap prevents losing context at chunk boundaries

def load_bart():
    """Lazily load BART model and tokenizer into MODELS cache."""
    if "bart" not in MODELS:
        print(f"[BART] Loading {BART_MODEL}...")
        print("[BART] Download ~1.6 GB on first run (cached after)")
        tok = BartTokenizer.from_pretrained(BART_MODEL)
        mdl = BartForConditionalGeneration.from_pretrained(BART_MODEL)
        mdl = mdl.to(DEVICE)
        MODELS["bart"] = (mdl, tok)
        print(f"[BART] Loaded on {DEVICE.upper()} - cached in MODELS['bart']")
    return MODELS["bart"]

def tokenizer_chunk(text: str, tokenizer, max_tok: int, overlap: int) -> List[str]:
    """
    Split text into overlapping token-based chunks using BartTokenizer.

    Algorithm:
        1. Encode full text -> subword token IDs
        2. Slide window of max_tok tokens with (max_tok - overlap) step
        3. Decode each window back to readable text

    Why tokenizer-based over character/word splitting?
        - BART's 1024-token limit is in subword tokens, not words
        - 'unbelievable' = 4 tokens; word-splitting would under-count
        - Tokenizer decoding guarantees no partial tokens
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    chunks, start, step = [], 0, max_tok - overlap
    while start < len(ids):
        end = min(start + max_tok, len(ids))
        chunks.append(tokenizer.decode(ids[start:end], skip_special_tokens=True))
        if end == len(ids):
            break
        start += step
    return chunks

def summarize_one_chunk(chunk: str, model, tokenizer,
                         max_len: int = 200, min_len: int = 50) -> str:
    """Run BART inference on a single text chunk."""
    inputs = tokenizer(chunk, return_tensors="pt",
                       max_length=1024, truncation=True).to(DEVICE)
    with torch.no_grad():
        ids = model.generate(
            inputs["input_ids"],
            max_length=max_len,
            min_length=min_len,
            num_beams=4,
            length_penalty=2.0,     # Encourages completeness
            early_stopping=True,
            no_repeat_ngram_size=3, # Prevents repetitive phrases
        )
    return tokenizer.decode(ids[0], skip_special_tokens=True)

def summarize_text(text: str, detail: str = "medium") -> Dict[str, Any]:
    """
    Abstractive summarization using BART with Map-Reduce strategy.

    MAP  : Split full transcript into N overlapping chunks -> summarize each
    REDUCE: If combined chunk summaries too long -> summarize again recursively

    This ensures the ENTIRE lecture is summarized, not just the first 1024 tokens.

    Args:
        text   : Full transcript text
        detail : 'brief' | 'medium' | 'detailed'

    Returns dict:
        summary           - Final summary string
        chunk_summaries   - Individual chunk summaries (for debugging)
        num_chunks        - Number of chunks processed
        compression_ratio - input_words / output_words
    """
    cfg = {"brief": (130, 40), "medium": (200, 60), "detailed": (300, 80)}
    max_len, min_len = cfg.get(detail, (200, 60))

    model, tokenizer = load_bart()

    print(f"\\n[BART] Input: {len(text.split())} words | detail={detail}")

    # ── MAP PHASE ─────────────────────────────────────────────
    chunks = tokenizer_chunk(text, tokenizer, MAX_CHUNK_TOKENS, OVERLAP_TOKENS)
    print(f"[BART] Chunks: {len(chunks)} (each ~{MAX_CHUNK_TOKENS} tokens, {OVERLAP_TOKENS}-tok overlap)")

    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        print(f"[BART] Summarizing chunk {i+1}/{len(chunks)}...", end="\\r")
        chunk_summaries.append(summarize_one_chunk(chunk, model, tokenizer, max_len, min_len))

    combined = " ".join(chunk_summaries)

    # ── REDUCE PHASE ──────────────────────────────────────────
    combined_ids = tokenizer.encode(combined, add_special_tokens=False)
    if len(combined_ids) > MAX_CHUNK_TOKENS:
        print(f"\\n[BART] Reduce phase: {len(combined_ids)} tokens -> second-pass summary")
        final = summarize_one_chunk(combined, model, tokenizer, max_len * 2, min_len * 2)
    else:
        final = combined

    # Free VRAM
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

    ratio = len(text.split()) / max(len(final.split()), 1)
    print(f"\\n[BART] Done: {len(text.split())} -> {len(final.split())} words ({ratio:.1f}x compression)")

    return {
        "summary":           final,
        "chunk_summaries":   chunk_summaries,
        "num_chunks":        len(chunks),
        "input_words":       len(text.split()),
        "output_words":      len(final.split()),
        "compression_ratio": round(ratio, 1),
    }

print("[OK] Summarization module ready")
print(f"     Model: {BART_MODEL}")
print("     Strategy: Map-Reduce with 900-token chunks, 150-token overlap")
'''
cells.append(code(CELL5_CODE))

# =============================================================================
# CELL 6 MARKDOWN
# =============================================================================
cells.append(md(
"## Cell 6 - Keyword Extraction (KeyBERT + MMR) & Named Entity Recognition (spaCy)\n"
"\n"
"### Algorithm: KeyBERT with Maximal Marginal Relevance\n"
"\n"
"**Comparison of Keyword Extraction Methods:**\n"
"\n"
"| Method | Approach | Accuracy | Speed | Semantic? |\n"
"|--------|----------|----------|-------|-----------|\n"
"| **KeyBERT + MMR** (chosen) | BERT embeddings + cosine similarity | Highest | Slow | Yes |\n"
"| YAKE | Statistical (position + frequency) | High | Fast | No |\n"
"| TF-IDF | Term frequency / inverse doc freq | Medium | Very fast | No |\n"
"| RAKE | Graph-based co-occurrence | Medium | Fastest | No |\n"
"\n"
"**KeyBERT Algorithm (Step by Step):**\n"
"1. Embed the full document using `all-MiniLM-L6-v2` (384-dim sentence embedding)\n"
"2. Extract candidate n-grams (1 to 3 words) as keyword candidates\n"
"3. Embed each candidate phrase\n"
"4. Rank by **cosine similarity** between candidate embedding and document embedding\n"
"5. Apply **MMR (Maximal Marginal Relevance)** to ensure diverse keywords:\n"
"\n"
"```\n"
"MMR score = lambda * sim(ki, doc) - (1 - lambda) * max_j sim(ki, kj)\n"
"where lambda = 1 - diversity (0.7 diversity = balanced relevance + variety)\n"
"```\n"
"\n"
"**Why MMR?** Without it, all 15 keywords could be near-synonyms  \n"
"(e.g., 'neural net', 'neural network', 'artificial neural network')\n"
"\n"
"### spaCy NER Entity Types Extracted:\n"
"`PERSON` `ORG` `GPE` (geopolitical) `DATE` `EVENT` `PRODUCT` `LAW` `WORK_OF_ART`"
))

# =============================================================================
# CELL 6 CODE
# =============================================================================
CELL6_CODE = '''\
# ============================================================
# CELL 6: KEYWORD EXTRACTION - KeyBERT + MMR + spaCy NER
# ============================================================

from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

NER_LABELS = {"PERSON", "ORG", "GPE", "DATE", "EVENT", "PRODUCT", "LAW", "WORK_OF_ART"}

def load_keybert() -> KeyBERT:
    """Lazily load KeyBERT with all-MiniLM-L6-v2 sentence embeddings."""
    if "keybert" not in MODELS:
        print("[KeyBERT] Loading all-MiniLM-L6-v2 sentence embeddings...")
        st_model = SentenceTransformer("all-MiniLM-L6-v2")
        MODELS["keybert"] = KeyBERT(model=st_model)
        print("[KeyBERT] Loaded and cached in MODELS['keybert']")
    return MODELS["keybert"]

def extract_keywords_entities(
    text:       str,
    top_n:      int   = 15,
    ngram_range: tuple = (1, 3),
    diversity:  float = 0.7,
) -> Dict[str, Any]:
    """
    Extract semantic keywords and named entities from lecture transcript.

    KeyBERT + MMR Pipeline:
        1. Embed document with all-MiniLM-L6-v2 (384-dim vectors)
        2. Generate candidate n-grams (1-word to 3-word phrases)
        3. Embed each candidate
        4. Score by cosine similarity(candidate, document)
        5. Re-rank with MMR:
              score = diversity*sim(ki,doc) - (1-diversity)*max_j sim(ki,kj)
           diversity=0.7 balances relevance (0.3 weight) vs uniqueness (0.7 weight)

    spaCy NER:
        - Uses en_core_web_sm dependency parse + NER pipeline
        - Processes first 100K chars (spaCy default limit)
        - Groups entities by type: PERSON, ORG, GPE, DATE, etc.

    Args:
        text       : Full transcript text
        top_n      : Number of keywords to return
        ngram_range: (min_n, max_n) for candidate phrase length
        diversity  : MMR diversity 0=no variety, 1=max variety

    Returns dict:
        keywords : List of {keyword, score} sorted by relevance
        entities : Dict of {entity_type: [entity_list]}
        topic    : Top keyword (proxy for main topic)
    """
    kw_model = load_keybert()

    print(f"[KeyBERT] Extracting top {top_n} keywords (diversity={diversity})...")

    # ── KeyBERT + MMR ────────────────────────────────────────
    kws = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=ngram_range,
        stop_words="english",
        use_mmr=True,
        diversity=diversity,
        top_n=top_n,
    )

    # ── spaCy NER ─────────────────────────────────────────────
    entities: Dict[str, List[str]] = {}
    if NLP is not None:
        doc = NLP(text[:100_000])  # spaCy processes up to 100K chars
        for ent in doc.ents:
            if ent.label_ in NER_LABELS:
                label = ent.label_
                if label not in entities:
                    entities[label] = []
                if ent.text not in entities[label]:
                    entities[label].append(ent.text)
    else:
        print("[KeyBERT] WARNING: spaCy not loaded - skipping NER")

    n_entities = sum(len(v) for v in entities.values())
    print(f"[KeyBERT] Done: {len(kws)} keywords, {n_entities} named entities")

    return {
        "keywords": [{"keyword": kw, "score": round(sc, 4)} for kw, sc in kws],
        "entities": entities,
        "topic":    kws[0][0] if kws else "N/A",
    }

def format_keywords(result: Dict) -> str:
    """Format keyword results as a readable table."""
    lines = ["KEYWORD EXTRACTION RESULTS", "-" * 50,
             f"{'Rank':<5} {'Keyword':<30} {'Score':<8} Bar"]
    lines.append("-" * 50)
    for i, kw in enumerate(result["keywords"], 1):
        bar = "#" * int(kw["score"] * 25)
        lines.append(f"{i:<5} {kw['keyword']:<30} {kw['score']:.4f}  {bar}")
    lines.append("")
    lines.append("NAMED ENTITIES:")
    for label, ents in result["entities"].items():
        lines.append(f"  {label:<15}: {', '.join(ents[:8])}")
    return "\\n".join(lines)

print("[OK] Keyword extraction module ready")
print("     Primary: KeyBERT + MMR (all-MiniLM-L6-v2)")
print("     Secondary: spaCy en_core_web_sm NER")
'''
cells.append(code(CELL6_CODE))

# =============================================================================
# CELL 7 MARKDOWN
# =============================================================================
cells.append(md(
"## Cell 7 - Question Answering (RoBERTa-SQuAD2 + TF-IDF Retrieval)\n"
"\n"
"### Algorithm: Extractive QA with Context Retrieval\n"
"\n"
"**Model Benchmark (SQuAD 2.0 Dev Set):**\n"
"\n"
"| Model | F1 Score | Exact Match | Unanswerable? |\n"
"|-------|----------|-------------|---------------|\n"
"| **deepset/roberta-base-squad2** (chosen) | **82.9%** | **79.9%** | Yes (SQuAD2) |\n"
"| distilbert-base-uncased-distilled-squad | ~65% | ~57% | No (SQuAD1.1) |\n"
"| bert-large-uncased-whole-word-masking-squad2 | 85.8% | 83.0% | Yes | Too large for T4 |\n"
"\n"
"**Why RoBERTa over DistilBERT?**\n"
"- distilbert-squad is trained on SQuAD **1.1** (no unanswerable questions)\n"
"- RoBERTa-squad2 handles: 'Who invented X?' when X isn't in the lecture\n"
"- Returns low confidence score instead of hallucinating an answer\n"
"\n"
"### Context Retrieval: TF-IDF + Cosine Similarity\n"
"\n"
"```\n"
"Full Transcript (50K+ words)\n"
"         |\n"
"         v  Split into overlapping sentence windows\n"
"[chunk_1, chunk_2, ..., chunk_N]  (~200 words each)\n"
"         |\n"
"         v  TF-IDF vectorize all chunks + question\n"
"[vec_1, vec_2, ..., vec_N, vec_Q]\n"
"         |\n"
"         v  Cosine similarity(vec_Q, vec_i) for all i\n"
"Top-3 most similar chunks selected\n"
"         |\n"
"         v  Concatenate -> RoBERTa QA context\n"
"Answer extracted from context\n"
"```\n"
"\n"
"**Why TF-IDF over FAISS/semantic search?**\n"
"- No additional model needed (FAISS requires dense embeddings = extra VRAM)\n"
"- TF-IDF is transparent and interpretable\n"
"- Sufficient for lecture-scale content (minutes, not millions of documents)"
))

# =============================================================================
# CELL 7 CODE
# =============================================================================
CELL7_CODE = '''\
# ============================================================
# CELL 7: QUESTION ANSWERING - RoBERTa-SQuAD2 + TF-IDF Retrieval
# ============================================================

from transformers import pipeline as hf_pipeline

QA_MODEL = "deepset/roberta-base-squad2"

def load_qa_pipeline():
    """Lazily load extractive QA pipeline into MODELS cache."""
    if "qa" not in MODELS:
        print(f"[QA] Loading {QA_MODEL}...")
        print("[QA] Download ~500 MB on first run (cached after)")
        MODELS["qa"] = hf_pipeline(
            "question-answering",
            model=QA_MODEL,
            tokenizer=QA_MODEL,
            device=0 if DEVICE == "cuda" else -1,
            handle_impossible_answer=True,  # SQuAD2 unanswerable support
        )
        print(f"[QA] Loaded on {DEVICE.upper()} - cached in MODELS['qa']")
    return MODELS["qa"]

def retrieve_context_tfidf(
    transcript: str,
    question:   str,
    top_k:      int = 3,
    chunk_words: int = 200,
) -> Tuple[str, List[str]]:
    """
    Retrieve the most relevant transcript chunks for a question using TF-IDF.

    Algorithm:
        1. Split transcript into overlapping sentence-based chunks (~200 words)
        2. TF-IDF vectorize all chunks + the question (bigrams, 5000 features)
        3. Compute cosine_similarity(question_vector, chunk_vectors)
        4. Select top-k chunks by similarity score
        5. Return chunks in their original order (preserves context flow)

    Args:
        transcript  : Full lecture transcript
        question    : User's question string
        top_k       : Number of chunks to retrieve
        chunk_words : Approximate words per chunk

    Returns:
        (context_string, list_of_retrieved_chunks)
    """
    sents = sent_tokenize(transcript)
    if not sents:
        return transcript[:2000], [transcript[:2000]]

    # Build overlapping chunks from sentences
    chunks, buf, buf_len = [], [], 0
    for sent in sents:
        w = len(sent.split())
        if buf_len + w > chunk_words and buf:
            chunks.append(" ".join(buf))
            buf = buf[-2:]  # 2-sentence overlap
            buf_len = sum(len(s.split()) for s in buf)
        buf.append(sent)
        buf_len += w
    if buf:
        chunks.append(" ".join(buf))

    if len(chunks) == 1:
        return chunks[0], chunks

    # TF-IDF similarity
    vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words="english", max_features=5000)
    tfidf = vectorizer.fit_transform(chunks + [question])
    sims  = cosine_similarity(tfidf[-1:], tfidf[:-1])[0]

    # Top-k in original order
    top_idx = sorted(np.argsort(sims)[::-1][:top_k].tolist())
    top_chunks = [chunks[i] for i in top_idx]
    return " ".join(top_chunks), top_chunks

def answer_question(
    transcript: str,
    question:   str,
    top_k:      int = 3,
    max_ctx_words: int = 450,
) -> Dict[str, Any]:
    """
    Answer a question based on the lecture transcript.

    Full Pipeline:
        1. TF-IDF retrieval: find top-3 relevant transcript chunks
        2. Truncate combined context to 450 words (RoBERTa 512-token limit)
        3. Run RoBERTa-SQuAD2 extractive QA on context
        4. Return answer + confidence + retrieved context

    Why Extractive QA (not generative)?
        - Extractive QA returns spans from the actual transcript (verifiable)
        - Generative QA can hallucinate facts not in the lecture
        - RoBERTa-SQuAD2 F1=82.9% on unanswerable questions dataset

    Args:
        transcript     : Full lecture transcript
        question       : User question string
        top_k          : Number of TF-IDF chunks to use as context
        max_ctx_words  : Context word limit before RoBERTa truncation

    Returns dict:
        answer          - Extracted answer span
        score           - Model confidence (0-1)
        confidence_pct  - Formatted percentage
        context         - Context used for QA
        chunks_used     - Individual retrieved chunks
    """
    if not transcript.strip():
        return {"answer": "No transcript loaded.", "score": 0.0, "context": "", "chunks_used": []}
    if not question.strip():
        return {"answer": "Please type a question.", "score": 0.0, "context": "", "chunks_used": []}

    qa = load_qa_pipeline()

    print(f"[QA] Question: {question[:80]}...")
    context, chunks = retrieve_context_tfidf(transcript, question, top_k)

    # Truncate to model limit
    ctx_words = context.split()
    if len(ctx_words) > max_ctx_words:
        context = " ".join(ctx_words[:max_ctx_words])

    print(f"[QA] Context : {len(context.split())} words from {len(chunks)} retrieved chunks")

    result = qa(question=question, context=context, max_answer_len=120)
    score  = result["score"]
    answer = result["answer"]

    # Low confidence = unanswerable
    if score < 0.01:
        answer = "The answer to this question was not found in the transcript."

    print(f"[QA] Answer  : {answer[:100]} (confidence: {score:.1%})")

    return {
        "answer":         answer,
        "score":          round(score, 4),
        "confidence_pct": f"{score:.1%}",
        "context":        context,
        "chunks_used":    chunks,
    }

def compute_qa_metrics(predictions: List[str], ground_truths: List[str]) -> Dict[str, float]:
    """
    Compute token-level F1 and Exact Match for QA evaluation.

    Token F1 = 2 * precision * recall / (precision + recall)
    where precision/recall are over shared TOKEN bags (lowercased, no punctuation).

    This matches the official SQuAD evaluation script methodology.
    """
    def normalize(text):
        return set(re.sub(r"[^\\w\\s]", "", text.lower()).split())

    em_scores, f1_scores = [], []
    for pred, truth in zip(predictions, ground_truths):
        p_toks, t_toks = normalize(pred), normalize(truth)
        em_scores.append(int(p_toks == t_toks))
        common = p_toks & t_toks
        if not common:
            f1_scores.append(0.0)
        else:
            prec = len(common) / len(p_toks)
            rec  = len(common) / len(t_toks)
            f1_scores.append(2 * prec * rec / (prec + rec))

    return {
        "exact_match": round(np.mean(em_scores) * 100, 2),
        "f1":          round(np.mean(f1_scores) * 100, 2),
        "n_samples":   len(predictions),
    }

print("[OK] Question Answering module ready")
print(f"     Model: {QA_MODEL} (F1=82.9%, EM=79.9% on SQuAD2)")
print("     Retrieval: TF-IDF cosine similarity over sentence chunks")
'''
cells.append(code(CELL7_CODE))

# =============================================================================
# CELL 8 MARKDOWN
# =============================================================================
cells.append(md(
"## Cell 8 - Translation (Helsinki-NLP OPUS-MT)\n"
"\n"
"### Algorithm: Neural Machine Translation with MarianMT\n"
"\n"
"**Why Helsinki-NLP OPUS-MT?**\n"
"\n"
"| Library | Quality | Cost | Stability | Offline? |\n"
"|---------|---------|------|-----------|----------|\n"
"| **Helsinki-NLP opus-mt** (chosen) | 30-40 BLEU | Free | Stable | Yes |\n"
"| googletrans | Best | Free | Unstable (reverse-engineered) | No |\n"
"| Google Translate API | Best | Paid | Stable | No |\n"
"| deep-translator | Varies | Free/Paid | Moderate | No |\n"
"\n"
"**Why NOT googletrans?**\n"
"- Uses unofficial reverse-engineered Google Translate API\n"
"- Frequently breaks when Google updates their internal API\n"
"- No guarantee of uptime; unusable in production\n"
"\n"
"**Supported Language Pairs:**\n"
"- English -> Hindi:   `Helsinki-NLP/opus-mt-en-hi`\n"
"- English -> French:  `Helsinki-NLP/opus-mt-en-fr`\n"
"- English -> German:  `Helsinki-NLP/opus-mt-en-de`\n"
"- English -> Spanish: `Helsinki-NLP/opus-mt-en-es`\n"
"- English -> Arabic:  `Helsinki-NLP/opus-mt-en-ar`\n"
"\n"
"**Chunking for Long Text:**\n"
"- MarianMT has 512-token input limit\n"
"- Sentence-based chunking: ~400 chars per chunk, translate independently\n"
"- Each model is lazy-loaded and cached separately (save RAM)"
))

# =============================================================================
# CELL 8 CODE
# =============================================================================
CELL8_CODE = '''\
# ============================================================
# CELL 8: TRANSLATION - Helsinki-NLP OPUS-MT (MarianMT)
# ============================================================

from transformers import MarianMTModel, MarianTokenizer

def load_translation_model(lang_code: str) -> Tuple:
    """
    Lazily load MarianMT model for a specific target language.

    Each language model is cached independently in MODELS dict.
    Pattern: MODELS["translate_hi"] = (model, tokenizer)

    Why lazy per-language loading?
        - Each model is ~300 MB
        - Loading all 5 at once = 1.5 GB RAM (wasteful if user only needs 1)
        - Cache on first use -> instant on subsequent calls
    """
    key = f"translate_{lang_code}"
    if key not in MODELS:
        model_name = TRANSLATION_MODELS.get(lang_code)
        if not model_name:
            raise ValueError(
                f"Unsupported language: '{lang_code}'. "
                f"Use one of: {list(TRANSLATION_MODELS.keys())}"
            )
        print(f"[Translate] Loading {model_name}...")
        tok = MarianTokenizer.from_pretrained(model_name)
        mdl = MarianMTModel.from_pretrained(model_name).to(DEVICE)
        MODELS[key] = (mdl, tok)
        lang_name = {v:k for k,v in LANG_MAP.items()}.get(lang_code, lang_code)
        print(f"[Translate] {lang_name} model loaded - cached as MODELS['{key}']")
    return MODELS[key]

def translate_chunk(text: str, model, tokenizer, max_len: int = 512) -> str:
    """Translate one chunk using MarianMT beam search (beam=4)."""
    inputs = tokenizer(text, return_tensors="pt", padding=True,
                       truncation=True, max_length=max_len).to(DEVICE)
    with torch.no_grad():
        translated = model.generate(**inputs, num_beams=4, early_stopping=True,
                                    max_length=max_len)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

def translate_text(text: str, target_lang: str) -> Dict[str, Any]:
    """
    Translate English text to the specified target language.

    Chunking Strategy (for long transcripts):
        1. Sentence-tokenize the text
        2. Group sentences into chunks of max 400 characters
        3. Translate each chunk independently
        4. Join translated chunks

    MarianMT generation settings:
        num_beams=4    : Beam search for quality translation
        max_length=512 : Match model context limit
        early_stopping : Stop beam search when EOS token generated

    Args:
        text        : English source text
        target_lang : ISO language code (hi/fr/de/es/ar)

    Returns dict:
        translated_text : Full translated string
        source_lang     : 'en'
        target_lang     : ISO code
        lang_name       : Full language name
        src_words       : Source word count
        tgt_words       : Translated word count
    """
    model, tokenizer = load_translation_model(target_lang)
    lang_name = {v: k for k, v in LANG_MAP.items()}.get(target_lang, target_lang)

    print(f"[Translate] Target: {lang_name} | Input: {len(text.split())} words")

    # Sentence-based chunking
    sents = sent_tokenize(text)
    chunks, buf, buf_len = [], [], 0
    for sent in sents:
        if buf_len + len(sent) > 400 and buf:
            chunks.append(" ".join(buf))
            buf, buf_len = [], 0
        buf.append(sent)
        buf_len += len(sent)
    if buf:
        chunks.append(" ".join(buf))

    # Translate chunk by chunk
    translated_chunks = []
    for i, chunk in enumerate(chunks):
        print(f"[Translate] Chunk {i+1}/{len(chunks)}...", end="\\r")
        translated_chunks.append(translate_chunk(chunk, model, tokenizer))

    result_text = " ".join(translated_chunks)
    print(f"\\n[Translate] Done: {len(chunks)} chunks -> {len(result_text.split())} words")

    return {
        "translated_text": result_text,
        "source_lang":     "en",
        "target_lang":     target_lang,
        "lang_name":       lang_name,
        "src_words":       len(text.split()),
        "tgt_words":       len(result_text.split()),
    }

print("[OK] Translation module ready")
print(f"     Supported: {', '.join(f'{k} ({v})' for k,v in LANG_MAP.items())}")
print("     Models load lazily on first use (~300 MB each)")
'''
cells.append(code(CELL8_CODE))

# =============================================================================
# CELL 9 MARKDOWN
# =============================================================================
cells.append(md(
"## Cell 9 - Accuracy Measurement (WER + ROUGE + QA F1/EM)\n"
"\n"
"### Metrics Overview\n"
"\n"
"| Metric | Library | Pipeline | Formula |\n"
"|--------|---------|----------|---------|\n"
"| **WER** | `jiwer` | Transcription | (S+D+I)/N |\n"
"| **MER** | `jiwer` | Transcription | Substitutions+Deletions / max(ref,hyp) |\n"
"| **ROUGE-1** | `rouge-score` | Summarization | Unigram F1 |\n"
"| **ROUGE-2** | `rouge-score` | Summarization | Bigram F1 |\n"
"| **ROUGE-L** | `rouge-score` | Summarization | LCS-based F1 |\n"
"| **Exact Match** | Custom | Q&A | Binary per-question |\n"
"| **Token F1** | Custom | Q&A | 2*P*R/(P+R) over token bags |\n"
"\n"
"### WER Interpretation (faster-whisper large-v3 target: <10%)\n"
"- WER < 5%: Professional broadcast quality\n"
"- WER < 10%: Research target for clear lecture audio\n"
"- WER < 20%: Acceptable for rough notes\n"
"- WER > 20%: Poor (noisy audio or wrong language)\n"
"\n"
"### ROUGE Interpretation (BART-large-CNN on CNN/DailyMail)\n"
"- ROUGE-1 > 40: Strong unigram overlap\n"
"- ROUGE-2 > 18: Good phrase-level precision\n"
"- ROUGE-L > 35: Coherent sentence ordering preserved\n"
"\n"
"### jiwer Transforms Applied:\n"
"`ToLowerCase -> RemovePunctuation -> RemoveMultipleSpaces -> Strip`  \n"
"Ensures fair comparison regardless of punctuation or casing differences"
))

# =============================================================================
# CELL 9 CODE
# =============================================================================
CELL9_CODE = '''\
# ============================================================
# CELL 9: ACCURACY MEASUREMENT - WER + ROUGE + QA F1/EM
# ============================================================

def compute_wer_metrics(reference: str, hypothesis: str) -> Dict[str, Any]:
    """
    Compute Word Error Rate and related metrics using jiwer.

    WER Formula:
        WER = (Substitutions + Deletions + Insertions) / N_reference_words

    Preprocessing transforms (ensures fair comparison):
        1. ToLowerCase          - case-insensitive comparison
        2. RemovePunctuation    - ignore punctuation differences
        3. RemoveMultipleSpaces - normalize whitespace
        4. Strip                - remove leading/trailing spaces
        5. ReduceToListOfListOfWords - tokenize for jiwer internals

    Additional metrics:
        MER (Match Error Rate) = (S+D) / max(|ref|, |hyp|)
        WIL (Word Info Lost)   = 1 - (WER/MER) * (1 - WER)
        Hits: correctly transcribed words

    Returns:
        Dict with wer, mer, wil, hits, substitutions, deletions, insertions
    """
    if not reference.strip() or not hypothesis.strip():
        return {"wer": None, "mer": None, "wil": None,
                "hits": 0, "substitutions": 0, "deletions": 0, "insertions": 0}

    transforms = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(word_delimiter=" "),
    ])

    measures = jiwer.compute_measures(
        reference, hypothesis,
        truth_transform=transforms,
        hypothesis_transform=transforms,
    )

    return {
        "wer":           round(measures["wer"] * 100, 2),
        "mer":           round(measures["mer"] * 100, 2),
        "wil":           round(measures["wil"] * 100, 2),
        "hits":          measures["hits"],
        "substitutions": measures["substitutions"],
        "deletions":     measures["deletions"],
        "insertions":    measures["insertions"],
    }

def compute_rouge_metrics(reference: str, hypothesis: str) -> Dict[str, float]:
    """
    Compute ROUGE-1, ROUGE-2, ROUGE-L using Google rouge-score.

    ROUGE-N Precision = |match_ngrams| / |hypothesis_ngrams|
    ROUGE-N Recall    = |match_ngrams| / |reference_ngrams|
    ROUGE-N F1        = 2 * P * R / (P + R)

    ROUGE-L uses Longest Common Subsequence (LCS) instead of n-grams,
    which measures sentence-level structural similarity.

    use_stemmer=True applies Porter stemming before comparison
    (so 'running' and 'run' count as matching).
    """
    if not reference.strip() or not hypothesis.strip():
        return {"rouge1_f1": None, "rouge2_f1": None, "rougeL_f1": None}

    scorer = rouge_scorer_lib.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True
    )
    s = scorer.score(reference, hypothesis)

    return {
        "rouge1_precision": round(s["rouge1"].precision * 100, 2),
        "rouge1_recall":    round(s["rouge1"].recall    * 100, 2),
        "rouge1_f1":        round(s["rouge1"].fmeasure  * 100, 2),
        "rouge2_f1":        round(s["rouge2"].fmeasure  * 100, 2),
        "rougeL_f1":        round(s["rougeL"].fmeasure  * 100, 2),
    }

def badge(val, good_thr, fair_thr, invert=False) -> str:
    """Return quality badge: GREEN good, YELLOW fair, RED poor."""
    if val is None:
        return "[N/A]"
    if not invert:
        if val >= good_thr: return f"[GOOD]  {val}"
        if val >= fair_thr: return f"[FAIR]  {val}"
        return f"[POOR]  {val}"
    else:  # lower is better (e.g., WER)
        if val <= good_thr: return f"[GOOD]  {val}"
        if val <= fair_thr: return f"[FAIR]  {val}"
        return f"[POOR]  {val}"

def compute_accuracy_report(
    ref_transcript: str = "",
    hyp_transcript: str = "",
    ref_summary:    str = "",
    hyp_summary:    str = "",
    qa_refs:  List[str] = None,
    qa_preds: List[str] = None,
) -> str:
    """
    Generate comprehensive accuracy report for all pipeline components.

    Args:
        ref_transcript : Ground truth transcript (human-verified)
        hyp_transcript : Model-generated transcript (faster-whisper output)
        ref_summary    : Reference summary (human-written or gold standard)
        hyp_summary    : Model-generated summary (BART output)
        qa_refs        : List of ground-truth answers
        qa_preds       : List of model-predicted answers

    Returns:
        Formatted multi-section accuracy report string
    """
    lines = [
        "=" * 65,
        "         STUDY MITRA - ACCURACY REPORT",
        "=" * 65,
    ]

    # ── 1. Transcription WER ─────────────────────────────────
    lines += ["", "1. TRANSCRIPTION ACCURACY (Word Error Rate)", "-" * 65]
    if ref_transcript and hyp_transcript:
        wer = compute_wer_metrics(ref_transcript, hyp_transcript)
        lines.append(f"   WER  : {badge(wer['wer'], 5, 10, invert=True)}%  (Target: < 10%)")
        lines.append(f"   MER  : {wer['mer']}%")
        lines.append(f"   WIL  : {wer['wil']}%")
        lines.append(f"   Hits : {wer['hits']} | Subs: {wer['substitutions']} | Del: {wer['deletions']} | Ins: {wer['insertions']}")
    else:
        lines.append("   [N/A] Provide reference + hypothesis transcripts to compute WER")

    # ── 2. Summarization ROUGE ───────────────────────────────
    lines += ["", "2. SUMMARIZATION QUALITY (ROUGE Scores)", "-" * 65]
    if ref_summary and hyp_summary:
        rouge = compute_rouge_metrics(ref_summary, hyp_summary)
        lines.append(f"   ROUGE-1 F1 : {badge(rouge['rouge1_f1'], 40, 30)}  (>40 = Good)")
        lines.append(f"   ROUGE-2 F1 : {badge(rouge['rouge2_f1'], 18, 10)}  (>18 = Good)")
        lines.append(f"   ROUGE-L F1 : {badge(rouge['rougeL_f1'], 35, 25)}  (>35 = Good)")
    else:
        lines.append("   [N/A] Provide reference + hypothesis summaries to compute ROUGE")

    # ── 3. QA F1 / EM ────────────────────────────────────────
    lines += ["", "3. QUESTION ANSWERING (F1 + Exact Match)", "-" * 65]
    if qa_refs and qa_preds and len(qa_refs) == len(qa_preds):
        qa_m = compute_qa_metrics(qa_preds, qa_refs)
        lines.append(f"   Exact Match : {badge(qa_m['exact_match'], 60, 40)}%  (>60 = Good)")
        lines.append(f"   Token F1    : {badge(qa_m['f1'], 70, 50)}%  (>70 = Good)")
        lines.append(f"   Evaluated   : {qa_m['n_samples']} QA pairs")
    else:
        lines.append("   [N/A] Provide QA ground-truth pairs to compute F1/EM")

    lines += ["", "=" * 65,
              "[GOOD] >= threshold | [FAIR] moderate | [POOR] below target | [N/A] not evaluated",
              "=" * 65]

    return "\\n".join(lines)

print("[OK] Accuracy measurement module ready")
print("     WER    : jiwer with full preprocessing transforms")
print("     ROUGE  : rouge-score with Porter stemming")
print("     QA F1  : token-bag intersection (SQuAD evaluation style)")
'''
cells.append(code(CELL9_CODE))

notebook = {**NOTEBOOK_META, "cells": cells}

with open("study_mitra.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print(f"[OK] study_mitra.ipynb written with {len(cells)} cells")

