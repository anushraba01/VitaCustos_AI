#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║      INFANT CRY DETECTION & CLASSIFICATION  —  RASPBERRY PI                  ║
║                                                                              ║
║  Model Files Required (from cry_rpi_models.zip):                             ║
║    config.json          — audio parameters & class names                     ║
║    label_classes.npy    — cry class labels                                   ║
║    scaler.pkl           — StandardScaler for feature normalisation           ║
║    pca.pkl              — PCA for dimensionality reduction                   ║
║    ml_ensemble.pkl      — RF + XGBoost + LightGBM + SVM voting ensemble      ║
║    ensemble_weights.npy — model blend weights                                ║
║                                                                              ║
║  Telegram Alerts: instant message + audio clip on every cry detection        ║
║                                                                              ║
║  INSTALL (one-time on Raspberry Pi):                                         ║
║    pip3 install librosa numpy joblib scikit-learn requests soundfile         ║
║    pip3 install xgboost lightgbm pyaudio                                     ║
║                                                                              ║
║  USAGE:                                                                      ║
║    python3 Cry_Detection_Classification.py              (live mic)           ║
║    python3 Cry_Detection_Classification.py --mode file --input cry.wav       ║
║    python3 Cry_Detection_Classification.py --mode batch --input ./dataset/   ║
║    python3 Cry_Detection_Classification.py --no-telegram                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import time
import queue
import logging
import argparse
import warnings
import threading
from datetime import datetime
from pathlib import Path
from collections import deque, Counter

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("CryDetector")

# ── Dependency check ───────────────────────────────────────────────────────────
try:
    import numpy as np
    import librosa
    import joblib
    import requests
except ImportError as e:
    sys.exit(
        f"\n[ERROR] Missing package: {e}\n"
        f"Run: pip3 install numpy librosa joblib requests\n"
    )

try:
    import pyaudio
    PYAUDIO_OK = True
except ImportError:
    PYAUDIO_OK = False
    log.warning("pyaudio not found — live mic mode disabled.  pip3 install pyaudio")

try:
    import soundfile as sf
    SF_OK = True
except ImportError:
    SF_OK = False
    log.warning("soundfile not found — audio clip saving disabled.  pip3 install soundfile")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ▶  EDIT THESE TWO LINES WITH YOUR TELEGRAM CREDENTIALS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TELEGRAM_TOKEN   = "YOUR_BOT_TOKEN_HERE"    # from @BotFather
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID_HERE"      # from @userinfobot
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── Runtime config (loaded / overridden from config.json at startup) ───────────
CONFIG = {
    # Audio processing
    "sample_rate"            : 22050,
    "duration"               : 4,
    "hop_length"             : 512,
    "n_fft"                  : 2048,
    "n_mfcc"                 : 40,
    "n_mels"                 : 128,
    "fmax"                   : 8000,
    "mel_width"              : 173,
    # Voice-Activity Detection
    "vad_energy_thresh"      : 0.02,
    "vad_zcr_high"           : 0.30,
    "vad_zcr_low"            : 0.01,
    "vad_frame_len"          : 1024,
    # Inference / smoothing
    "confidence_thresh"      : 0.50,
    "smoothing_window"       : 5,
    "inference_interval_sec" : 1.0,
    # Microphone
    "chunk_size"             : 1024,
    "mic_channels"           : 1,
    # Model directory
    "model_dir"              : "./models",
    # Telegram
    "telegram_enabled"       : True,
    "telegram_cooldown_sec"  : 30,
    "telegram_send_audio"    : True,
    "telegram_min_confidence": 0.55,
    "telegram_alert_classes" : None,   # None = all classes; or ["belly_pain","hungry"]
    # Logging / saving
    "log_csv"                : "./cry_log.csv",
    "save_audio_clips"       : True,
    "audio_clips_dir"        : "./detected_cries",
}

# ── Cry type metadata ──────────────────────────────────────────────────────────
CRY_META = {
    "hungry"     : {"emoji": "🍼", "label": "HUNGRY",
                    "urgency": "MEDIUM",
                    "advice":  "Baby is hungry — time to feed!"},
    "belly_pain" : {"emoji": "😣", "label": "BELLY PAIN",
                    "urgency": "HIGH",
                    "advice":  "Baby may have gas/colic — try gentle tummy massage."},
    "burping"    : {"emoji": "💨", "label": "BURPING",
                    "urgency": "LOW",
                    "advice":  "Baby needs to burp — hold upright, pat gently."},
    "discomfort" : {"emoji": "😖", "label": "DISCOMFORT",
                    "urgency": "MEDIUM",
                    "advice":  "Baby is uncomfortable — check diaper / temperature."},
    "tired"      : {"emoji": "😴", "label": "TIRED",
                    "urgency": "LOW",
                    "advice":  "Baby is sleepy — dim lights, reduce stimulation."},
    "unknown"    : {"emoji": "❓", "label": "UNKNOWN",
                    "urgency": "NONE",
                    "advice":  ""},
}
URGENCY_ICON = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢", "NONE": "⚪"}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TELEGRAM ALERT SYSTEM
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TelegramAlerter:
    """
    Non-blocking Telegram notifications via background thread.
    Supports: text alert, voice clip upload, cooldown per class,
              startup / shutdown messages, retry on failure.
    """
    _API = "https://api.telegram.org/bot{token}/{method}"

    def __init__(self, token, chat_id, enabled=True):
        self.token    = token
        self.chat_id  = str(chat_id)
        self.enabled  = enabled and token not in ("", "YOUR_BOT_TOKEN_HERE")
        self._last    = {}           # class → last alert timestamp
        self._lock    = threading.Lock()
        self._q       = queue.Queue()
        self._thread  = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        if self.enabled:
            ok, info = self._call("getMe")
            if ok:
                log.info(f"Telegram connected — @{info['result'].get('username','?')}")
            else:
                log.warning(f"Telegram connection failed: {info}")
                self.enabled = False
        else:
            log.info("Telegram disabled (token not set).")

    # ── Internal HTTP helper ───────────────────────────────────────────────────
    def _call(self, method, data=None, files=None, timeout=12):
        url = self._API.format(token=self.token, method=method)
        try:
            if files:
                r = requests.post(url, data=data, files=files, timeout=timeout)
            else:
                r = requests.post(url, json=data, timeout=timeout)
            r.raise_for_status()
            return True, r.json()
        except requests.exceptions.ConnectionError:
            return False, "No internet"
        except Exception as e:
            return False, str(e)

    # ── Background sender loop ─────────────────────────────────────────────────
    def _worker(self):
        while True:
            task = self._q.get()
            if task is None:
                break
            try:
                self._dispatch(task)
            except Exception as e:
                log.debug(f"Telegram worker error: {e}")
            self._q.task_done()

    def _dispatch(self, task):
        # Plain text message (startup / shutdown)
        if task.get("type") == "text":
            self._call("sendMessage", {
                "chat_id": self.chat_id,
                "text": task["text"],
                "parse_mode": "Markdown"
            })
            return

        # Cry alert
        result   = task["result"]
        audio_np = task.get("audio_np")
        cls      = result.get("class", "unknown")
        conf     = result.get("confidence", 0.0)
        info     = CRY_META.get(cls, CRY_META["unknown"])
        ts       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Build probability bar chart
        probs = result.get("probabilities", {})
        bar_text = ""
        for c, p in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            bar  = "█" * int(p * 10) + "░" * (10 - int(p * 10))
            em   = CRY_META.get(c, {}).get("emoji", "")
            mark = " ◄" if c == cls else ""
            bar_text += f"  {em}{c:<12s}[{bar}]{p*100:.0f}%{mark}\n"

        text = (
            f"{info['emoji']} *BABY CRY DETECTED!*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{URGENCY_ICON.get(info['urgency'],'⚪')} *Urgency:* {info['urgency']}\n"
            f"📋 *Type:* {info['label']}\n"
            f"📊 *Confidence:* {conf*100:.1f}%\n"
            f"🕐 *Time:* {ts}\n\n"
            f"💡 *Action:* {info['advice']}\n\n"
            f"📈 *Probabilities:*\n```\n{bar_text}```"
        )

        # Send text (up to 3 retries)
        for attempt in range(3):
            ok, _ = self._call("sendMessage", {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "Markdown"
            })
            if ok:
                log.info(f"Telegram alert sent: {cls} ({conf*100:.1f}%)")
                break
            time.sleep(2)

        # Send audio voice clip
        if CONFIG["telegram_send_audio"] and audio_np is not None and SF_OK:
            try:
                import io
                buf = io.BytesIO()
                sf.write(buf, audio_np, CONFIG["sample_rate"], format="WAV")
                buf.seek(0)
                fname = f"cry_{cls}_{datetime.now().strftime('%H%M%S')}.wav"
                ok2, _ = self._call(
                    "sendVoice",
                    data={
                        "chat_id": self.chat_id,
                        "caption": f"{info['emoji']} {cls} cry clip"
                    },
                    files={"voice": (fname, buf, "audio/wav")}
                )
                if ok2:
                    log.info("Telegram audio clip sent")
            except Exception as e:
                log.debug(f"Audio clip upload error: {e}")

    # ── Public interface ───────────────────────────────────────────────────────
    def send_alert(self, result, audio_np=None):
        """Queue a cry alert — returns immediately (non-blocking)."""
        if not self.enabled:
            return
        cls  = result.get("class", "unknown")
        conf = result.get("confidence", 0.0)

        # Class filter
        allowed = CONFIG["telegram_alert_classes"]
        if allowed and cls not in allowed:
            return

        # Confidence filter
        if conf < CONFIG["telegram_min_confidence"]:
            return

        # Per-class cooldown
        with self._lock:
            elapsed = time.time() - self._last.get(cls, 0)
            if elapsed < CONFIG["telegram_cooldown_sec"]:
                log.debug(f"Cooldown active for '{cls}': "
                          f"{CONFIG['telegram_cooldown_sec'] - elapsed:.0f}s remaining")
                return
            self._last[cls] = time.time()

        self._q.put({"result": result, "audio_np": audio_np})

    def send_text(self, text):
        if self.enabled:
            self._q.put({"type": "text", "text": text})

    def send_startup(self, classes):
        if not self.enabled:
            return
        ts       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cls_line = "  ".join(
            CRY_META.get(c, {}).get("emoji", "") + " " + c for c in classes
        )
        self.send_text(
            f"👶 *Infant Cry Monitor — STARTED*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🟢 System is now actively monitoring\n"
            f"🕐 Started: {ts}\n"
            f"🎯 Classes:\n`{cls_line}`\n"
            f"📱 You will receive an alert whenever a cry is detected."
        )

    def send_shutdown(self):
        if self.enabled:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.send_text(
                f"🔴 *Infant Cry Monitor — STOPPED*\n"
                f"🕐 Stopped: {ts}"
            )
            time.sleep(2)   # let the message queue flush

    def stop(self):
        self._q.put(None)
        self._thread.join(timeout=6)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  AUDIO UTILITIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def load_and_pad(source):
    """
    Load audio from a file path or raw numpy array (int16 / float32).
    Returns a float32 array of exactly sample_rate × duration samples.
    """
    sr      = CONFIG["sample_rate"]
    max_len = sr * CONFIG["duration"]

    if isinstance(source, (str, Path)):
        try:
            y, _ = librosa.load(str(source), sr=sr, mono=True)
        except Exception as e:
            log.error(f"Cannot load '{source}': {e}")
            return None
    else:
        y = source.astype(np.float32)
        if np.abs(y).max() > 1.0:
            y = y / 32768.0     # int16 → float32

    # Normalise amplitude
    peak = np.abs(y).max()
    if peak > 1e-6:
        y = y / peak * 0.95

    # Pad or trim to fixed length
    if len(y) < max_len:
        y = np.pad(y, (0, max_len - len(y)), mode="constant")
    return y[:max_len].astype(np.float32)


def voice_activity_detect(y):
    """
    Lightweight cry detector using RMS energy + zero-crossing rate.
    Returns (is_cry: bool, mean_energy: float, mean_zcr: float)
    """
    fl    = CONFIG["vad_frame_len"]
    rms_list, zcr_list = [], []

    for start in range(0, len(y) - fl, fl // 2):
        frame = y[start: start + fl]
        rms_list.append(float(np.sqrt(np.mean(frame ** 2))))
        zcr_list.append(float(np.mean(np.abs(np.diff(np.sign(frame)))) / 2))

    if not rms_list:
        return False, 0.0, 0.0

    mean_rms = float(np.mean(rms_list))
    max_rms  = float(np.max(rms_list))
    mean_zcr = float(np.mean(zcr_list))

    is_cry = (
        max_rms  >  CONFIG["vad_energy_thresh"] and
        mean_zcr >  CONFIG["vad_zcr_low"]       and
        mean_zcr <  CONFIG["vad_zcr_high"]
    )
    return is_cry, mean_rms, mean_zcr


def extract_features(y):
    """
    Extract the full 1-D feature vector used during training:
    MFCC (+ delta + delta2)  ·  Mel Spectrogram  ·  Chroma
    Spectral Centroid / BW / Rolloff / Contrast  ·  Tonnetz
    ZCR  ·  RMS  ·  F0 (pyin)
    Returns shape (1, N_FEATURES).
    """
    sr    = CONFIG["sample_rate"]
    feats = []

    # ── MFCC + first and second derivatives ───────────────────────────────────
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr,
        n_mfcc=CONFIG["n_mfcc"],
        n_fft=CONFIG["n_fft"],
        hop_length=CONFIG["hop_length"]
    )
    for arr in [mfcc,
                librosa.feature.delta(mfcc),
                librosa.feature.delta(mfcc, order=2)]:
        feats += [arr.mean(axis=1), arr.std(axis=1),
                  arr.max(axis=1),  arr.min(axis=1)]

    # ── Mel Spectrogram stats ─────────────────────────────────────────────────
    mel    = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_mels=CONFIG["n_mels"],
        n_fft=CONFIG["n_fft"],
        hop_length=CONFIG["hop_length"]
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    feats += [mel_db.mean(axis=1), mel_db.std(axis=1)]

    # ── Chroma ────────────────────────────────────────────────────────────────
    chroma = librosa.feature.chroma_stft(
        y=y, sr=sr,
        n_fft=CONFIG["n_fft"],
        hop_length=CONFIG["hop_length"]
    )
    feats += [chroma.mean(axis=1), chroma.std(axis=1)]

    # ── Spectral features ─────────────────────────────────────────────────────
    spectral_fns = [
        lambda: librosa.feature.spectral_centroid(y=y, sr=sr),
        lambda: librosa.feature.spectral_bandwidth(y=y, sr=sr),
        lambda: librosa.feature.spectral_rolloff(y=y, sr=sr),
        lambda: librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=CONFIG["n_fft"]),
        lambda: librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr),
        lambda: librosa.feature.zero_crossing_rate(y),
        lambda: librosa.feature.rms(y=y),
    ]
    for fn in spectral_fns:
        try:
            a = fn()
            feats += [a.mean(axis=1), a.std(axis=1)]
        except Exception:
            pass

    # ── Fundamental frequency (pyin) ─────────────────────────────────────────
    try:
        f0, _, _ = librosa.pyin(y, fmin=50, fmax=1000, sr=sr)
        f0_clean = f0[~np.isnan(f0)] if f0 is not None else np.array([0.0])
        if len(f0_clean) == 0:
            f0_clean = np.array([0.0])
        feats.append(np.array([f0_clean.mean(), f0_clean.std(), f0_clean.max()]))
    except Exception:
        feats.append(np.array([0.0, 0.0, 0.0]))

    vector = np.concatenate([f.flatten() for f in feats]).astype(np.float32)
    return vector.reshape(1, -1)   # shape (1, N)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN CLASSIFIER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class InfantCryClassifier:
    """
    Loads the 6 model files from ./models/:
        config.json  ·  label_classes.npy  ·  scaler.pkl
        pca.pkl      ·  ml_ensemble.pkl    ·  ensemble_weights.npy

    Inference pipeline:
        raw audio → VAD → feature extraction → StandardScaler
                       → PCA → ML Ensemble (RF+XGB+LGB+SVM) → class + confidence
    """

    def __init__(self, model_dir=None):
        self.model_dir  = Path(model_dir or CONFIG["model_dir"])
        self.classes    = []
        self.scaler     = None
        self.pca        = None
        self.ensemble   = None
        self._smooth_buf = deque(maxlen=CONFIG["smoothing_window"])
        self._load_models()

    def _load_models(self):
        md = self.model_dir
        if not md.exists():
            raise FileNotFoundError(
                f"Model directory not found: {md}\n"
                f"Create ./models/ and extract cry_rpi_models.zip into it."
            )

        # ── 1. config.json  ──────────────────────────────────────────────────
        cfg_path = md / "config.json"
        if cfg_path.exists():
            saved = json.load(open(cfg_path))
            audio_keys = [
                "sample_rate", "duration", "n_mfcc", "n_mels",
                "hop_length", "n_fft", "fmax", "mel_width"
            ]
            for k in audio_keys:
                if k in saved:
                    CONFIG[k] = saved[k]
            if "classes" in saved and saved["classes"]:
                self.classes = [str(c) for c in saved["classes"]]
            log.info(f"Config loaded from config.json | "
                     f"sr={CONFIG['sample_rate']} dur={CONFIG['duration']}s")
        else:
            log.warning("config.json not found — using default audio parameters")

        # ── 2. label_classes.npy  ────────────────────────────────────────────
        lbl_path = md / "label_classes.npy"
        if lbl_path.exists():
            # np.load returns np.str_ objects — cast to plain Python str
            self.classes = [
                str(c) for c in np.load(str(lbl_path), allow_pickle=True)
            ]
            log.info(f"Classes loaded: {self.classes}")
        elif not self.classes:
            # Graceful fallback if both files are missing
            self.classes = ["hungry", "belly_pain", "burping", "discomfort", "tired"]
            log.warning(f"label_classes.npy not found — using default: {self.classes}")

        # ── 3. scaler.pkl  ───────────────────────────────────────────────────
        scaler_path = md / "scaler.pkl"
        if not scaler_path.exists():
            raise FileNotFoundError(
                f"scaler.pkl not found in {md}\n"
                f"This file is required.  Re-export from Colab."
            )
        self.scaler = joblib.load(str(scaler_path))
        log.info(f"Scaler loaded  (n_features={self.scaler.n_features_in_})")

        # ── 4. pca.pkl  ──────────────────────────────────────────────────────
        pca_path = md / "pca.pkl"
        if not pca_path.exists():
            raise FileNotFoundError(
                f"pca.pkl not found in {md}\n"
                f"This file is required.  Re-export from Colab."
            )
        self.pca = joblib.load(str(pca_path))
        log.info(f"PCA loaded  "
                 f"(components={self.pca.n_components_}  "
                 f"from {self.pca.n_features_in_} raw features)")

        # ── 5. ml_ensemble.pkl  ──────────────────────────────────────────────
        ens_path = md / "ml_ensemble.pkl"
        if not ens_path.exists():
            raise FileNotFoundError(
                f"ml_ensemble.pkl not found in {md}\n"
                f"This file is required.  Re-export from Colab."
            )
        self.ensemble = joblib.load(str(ens_path))
        log.info("ML Ensemble loaded  (RF + XGBoost + LightGBM + SVM)")

        # ── 6. ensemble_weights.npy  (optional info) ─────────────────────────
        w_path = md / "ensemble_weights.npy"
        if w_path.exists():
            w = np.load(str(w_path))
            log.info(f"Ensemble weights: CNN={w[0]:.3f}  Hybrid={w[1]:.3f}  ML={w[2]:.3f}")
            log.info("  (CNN & Hybrid TFLite not present — running ML-only mode)")

        log.info("━━━ Classifier ready ━━━")
        log.info(f"  Classes  : {self.classes}")
        log.info(f"  Pipeline : feature extraction → scaler → PCA → ML ensemble")

    # ── Feature preprocessing ──────────────────────────────────────────────────
    def _preprocess(self, y):
        """Raw audio → scaled + PCA-reduced feature vector."""
        raw = extract_features(y)           # (1, N_RAW)
        scaled = self.scaler.transform(raw) # (1, N_RAW)  normalise
        reduced = self.pca.transform(scaled)# (1, N_PCA)  reduce dims
        return reduced.astype(np.float32)

    # ── Main prediction method ─────────────────────────────────────────────────
    def predict(self, y):
        """
        Classify a preprocessed audio array.
        Returns a dict with keys:
          class, confidence, probabilities, is_cry,
          energy, zcr, timestamp
        """
        # Voice-activity check
        is_cry, energy, zcr = voice_activity_detect(y)
        base = {
            "is_cry"    : is_cry,
            "energy"    : energy,
            "zcr"       : zcr,
            "timestamp" : datetime.now().isoformat(),
        }
        if not is_cry:
            return {
                **base,
                "class"          : "unknown",
                "confidence"     : 0.0,
                "probabilities"  : {c: 0.0 for c in self.classes},
            }

        # Feature pipeline
        try:
            features = self._preprocess(y)
        except Exception as e:
            log.error(f"Feature extraction failed: {e}")
            return {
                **base,
                "class"         : "unknown",
                "confidence"    : 0.0,
                "probabilities" : {},
            }

        # ML ensemble prediction
        try:
            proba = self.ensemble.predict_proba(features)[0]   # shape (n_classes,)
        except Exception as e:
            log.error(f"Ensemble prediction failed: {e}")
            return {
                **base,
                "class"         : "unknown",
                "confidence"    : 0.0,
                "probabilities" : {},
            }

        # Map probabilities to class labels
        prob_dict = {self.classes[i]: float(proba[i]) for i in range(len(self.classes))}

        # Top class
        top_idx  = int(np.argmax(proba))
        top_cls  = self.classes[top_idx]
        top_conf = float(proba[top_idx])

        # Temporal smoothing (majority vote over last N predictions)
        self._smooth_buf.append(top_cls)
        smoothed = Counter(self._smooth_buf).most_common(1)[0][0]

        return {
            **base,
            "class"         : smoothed,
            "raw_class"     : top_cls,
            "confidence"    : top_conf,
            "probabilities" : prob_dict,
        }

    def predict_file(self, path):
        """Convenience wrapper: load a WAV file and classify."""
        y = load_and_pad(path)
        return self.predict(y) if y is not None else None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DISPLAY HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def print_result(r):
    SEP = "─" * 64
    print(f"\n{SEP}")
    print(f"  🕐  {r.get('timestamp', '')}")
    if not r.get("is_cry", False):
        print(f"  🔇  No cry detected")
        print(f"      energy={r.get('energy', 0):.5f}  "
              f"zcr={r.get('zcr', 0):.4f}")
        print(SEP)
        return

    cls     = r["class"]
    conf    = r["confidence"]
    info    = CRY_META.get(cls, CRY_META["unknown"])
    urgency = info["urgency"]

    print(f"  {URGENCY_ICON.get(urgency, '⚪')}  URGENCY  : {urgency}")
    print(f"  {info['emoji']}  TYPE     : {info['label']}")
    print(f"  📊  Confidence : {conf * 100:.1f}%")
    print(f"  🔊  Energy     : {r.get('energy', 0):.5f}")
    print(f"  💡  Action     : {info['advice']}")

    probs = r.get("probabilities", {})
    if probs:
        print()
        for c, p in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            bar  = "█" * int(p * 24) + "░" * (24 - int(p * 24))
            em   = CRY_META.get(c, {}).get("emoji", " ")
            mark = "  ◄" if c == cls else ""
            print(f"    {em} {c:<13s} [{bar}] {p * 100:5.1f}%{mark}")
    print(SEP)


def log_to_csv(r, path=None):
    path = path or CONFIG["log_csv"]
    is_new = not Path(path).exists()
    try:
        with open(path, "a") as f:
            if is_new:
                f.write("timestamp,is_cry,class,confidence,energy,zcr\n")
            f.write(
                f"{r.get('timestamp', '')},"
                f"{r.get('is_cry', False)},"
                f"{r.get('class', 'unknown')},"
                f"{r.get('confidence', 0):.4f},"
                f"{r.get('energy', 0):.5f},"
                f"{r.get('zcr', 0):.4f}\n"
            )
    except Exception as e:
        log.debug(f"CSV log error: {e}")


def save_audio_clip(y, cry_class):
    if not SF_OK or not CONFIG.get("save_audio_clips"):
        return None
    out_dir = CONFIG["audio_clips_dir"]
    os.makedirs(out_dir, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"{cry_class}_{ts}.wav")
    try:
        sf.write(path, y, CONFIG["sample_rate"])
        return path
    except Exception as e:
        log.debug(f"Audio save error: {e}")
        return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MICROPHONE CAPTURE  (non-blocking thread)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class MicCapture:
    def __init__(self):
        self._q  = queue.Queue()
        self._pa = None
        self._st = None

    def start(self):
        if not PYAUDIO_OK:
            raise RuntimeError(
                "pyaudio is required for live mode.\n"
                "Install: pip3 install pyaudio"
            )
        self._pa = pyaudio.PyAudio()
        info = self._pa.get_default_input_device_info()
        log.info(f"Microphone: {info.get('name', 'default')} "
                 f"(ch={CONFIG['mic_channels']} sr={CONFIG['sample_rate']})")

        def _cb(in_data, frame_count, time_info, status):
            self._q.put(np.frombuffer(in_data, dtype=np.int16))
            return None, pyaudio.paContinue

        self._st = self._pa.open(
            format=pyaudio.paInt16,
            channels=CONFIG["mic_channels"],
            rate=CONFIG["sample_rate"],
            input=True,
            frames_per_buffer=CONFIG["chunk_size"],
            stream_callback=_cb,
        )
        self._st.start_stream()
        log.info("Microphone capture started")

    def read_seconds(self, seconds):
        """Collect `seconds` of audio and return as float32 array."""
        sr         = CONFIG["sample_rate"]
        needed     = int(sr * seconds)
        n_chunks   = (needed + CONFIG["chunk_size"] - 1) // CONFIG["chunk_size"]
        collected  = []
        deadline   = time.time() + seconds + 2.0

        while len(collected) < n_chunks:
            if time.time() > deadline:
                log.warning("Mic read timeout")
                break
            try:
                collected.append(self._q.get(timeout=0.5))
            except queue.Empty:
                pass

        if not collected:
            return None
        audio = np.concatenate(collected)[:needed]
        return audio.astype(np.float32) / 32768.0

    def stop(self):
        if self._st:
            self._st.stop_stream()
            self._st.close()
        if self._pa:
            self._pa.terminate()
        log.info("Microphone stopped")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LIVE MODE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_live(clf, tg, duration=None):
    if not PYAUDIO_OK:
        print("\n[ERROR] Live mode requires pyaudio.  pip3 install pyaudio\n")
        return

    window   = CONFIG["duration"]
    interval = CONFIG["inference_interval_sec"]
    thresh   = CONFIG["confidence_thresh"]

    print(f"\n{'='*64}")
    print(f"  INFANT CRY MONITOR — LIVE DETECTION")
    print(f"  Window   : {window}s per analysis")
    print(f"  Threshold: {thresh*100:.0f}% confidence")
    print(f"  Telegram : {'ON ✓' if tg.enabled else 'OFF'}")
    print(f"  Press Ctrl+C to stop")
    print(f"{'='*64}\n")

    tg.send_startup(clf.classes)
    mic          = MicCapture()
    prev_crying  = False
    total_sec    = 0
    alert_count  = 0

    try:
        mic.start()
        time.sleep(0.5)   # warm up mic buffer
        t0 = time.time()

        while True:
            if duration and (time.time() - t0) > duration:
                print(f"\n⏱  Duration limit reached ({duration}s). Stopping.")
                break

            audio = mic.read_seconds(window)
            if audio is None:
                continue

            result = clf.predict(load_and_pad(audio))
            if result is None:
                continue

            total_sec += window
            crying = result.get("is_cry", False)
            conf   = result.get("confidence", 0.0)

            # Print on state change OR every confirmed cry
            if crying != prev_crying or (crying and conf >= thresh):
                print_result(result)
                log_to_csv(result)

            if crying and conf >= thresh:
                save_audio_clip(audio, result["class"])
                tg.send_alert(result, audio_np=audio)
                alert_count += 1

            prev_crying = crying
            sleep_time  = max(0.0, interval - window * 0.5)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\n  Stopping — please wait...")

    finally:
        mic.stop()
        tg.send_shutdown()
        tg.stop()
        print(f"\n  Session summary")
        print(f"  ├─ Monitored : {total_sec:.0f}s")
        print(f"  ├─ Alerts    : {alert_count}")
        print(f"  └─ Log saved : {CONFIG['log_csv']}\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FILE MODE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_file(clf, tg, input_path):
    p = Path(input_path)
    if not p.exists():
        print(f"\n[ERROR] Path not found: {p}\n")
        return

    wav_files = sorted(p.glob("**/*.wav")) if p.is_dir() else [p]
    if not wav_files:
        print(f"[ERROR] No .wav files found in {p}")
        return

    print(f"\n  Processing {len(wav_files)} file(s)...")
    thresh = CONFIG["confidence_thresh"]

    for wav in wav_files:
        print(f"\n  ▶  {wav.name}")
        result = clf.predict_file(wav)
        if result is None:
            print("     [SKIP] Failed to load file")
            continue
        print_result(result)
        log_to_csv(result)

        if result.get("is_cry") and result.get("confidence", 0) >= thresh:
            y = load_and_pad(wav)
            tg.send_alert(result, audio_np=y)

    tg.stop()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  BATCH EVALUATION MODE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_batch(clf, dataset_dir):
    from collections import defaultdict

    dp = Path(dataset_dir)
    if not dp.exists():
        print(f"\n[ERROR] Dataset directory not found: {dp}\n")
        return

    classes = sorted([
        d for d in os.listdir(dp)
        if (dp / d).is_dir() and not d.startswith(".")
    ])
    if not classes:
        print("[ERROR] No class subdirectories found")
        return

    print(f"\n  Batch Evaluation — {dp}")
    print(f"  Classes: {classes}\n")

    correct = defaultdict(int)
    total   = defaultdict(int)
    cm      = defaultdict(lambda: defaultdict(int))

    for true_cls in classes:
        wav_files = list((dp / true_cls).glob("*.wav"))
        print(f"  [{true_cls}] {len(wav_files)} files ... ", end="", flush=True)

        for wf in wav_files:
            result = clf.predict_file(wf)
            if result is None:
                continue
            pred_cls = result["class"] if result.get("is_cry") else "unknown"
            total[true_cls]   += 1
            cm[true_cls][pred_cls] += 1
            if pred_cls == true_cls:
                correct[true_cls] += 1

        cls_acc = correct[true_cls] / max(total[true_cls], 1)
        print(f"Accuracy: {cls_acc * 100:.1f}%  ({correct[true_cls]}/{total[true_cls]})")

    # Summary
    all_c = sum(correct.values())
    all_t = sum(total.values())
    print(f"\n{'='*52}")
    print(f"  OVERALL ACCURACY : {all_c / max(all_t, 1) * 100:.2f}%  ({all_c}/{all_t})")
    print(f"{'='*52}")

    # Confusion matrix
    header_label = "True \\ Pred"
    print(f"\n  {header_label:<16s}" +
          "".join(f"{c[:8]:>10s}" for c in classes))
    print("  " + "─" * (16 + 10 * len(classes)))
    for tc in classes:
        row = f"  {tc[:16]:<16s}" + "".join(
            f"{cm[tc].get(pc, 0):>10d}" for pc in classes
        )
        print(row)
    print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ARGUMENT PARSER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def build_parser():
    p = argparse.ArgumentParser(
        prog="Cry_Detection_Classification.py",
        description="Infant Cry Detector — Raspberry Pi (ML Ensemble + Telegram)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 Cry_Detection_Classification.py
  python3 Cry_Detection_Classification.py --mode live --duration 120
  python3 Cry_Detection_Classification.py --mode file  --input baby.wav
  python3 Cry_Detection_Classification.py --mode file  --input ./recordings/
  python3 Cry_Detection_Classification.py --mode batch --input ./dataset/
  python3 Cry_Detection_Classification.py --no-telegram
  python3 Cry_Detection_Classification.py --models /home/pi/models/
        """
    )
    p.add_argument("--mode",
                   choices=["live", "file", "batch"],
                   default="live",
                   help="Detection mode (default: live)")
    p.add_argument("--input",
                   type=str, default=None,
                   help="WAV file or directory (file / batch mode)")
    p.add_argument("--models",
                   type=str, default=None,
                   help="Path to model directory (default: ./models)")
    p.add_argument("--duration",
                   type=int, default=None,
                   help="Live mode max duration in seconds (default: unlimited)")
    p.add_argument("--threshold",
                   type=float, default=None,
                   help=f"Confidence threshold 0-1 (default: {CONFIG['confidence_thresh']})")
    p.add_argument("--no-telegram",
                   action="store_true",
                   help="Disable Telegram alerts for this run")
    p.add_argument("--tg-token",
                   type=str, default=None,
                   help="Telegram bot token (overrides hard-coded value)")
    p.add_argument("--tg-chat",
                   type=str, default=None,
                   help="Telegram chat ID (overrides hard-coded value)")
    p.add_argument("--verbose",
                   action="store_true",
                   help="Show debug-level logs")
    return p


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ENTRY POINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    args = build_parser().parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Apply CLI overrides
    if args.threshold:
        CONFIG["confidence_thresh"] = args.threshold
    if args.no_telegram:
        CONFIG["telegram_enabled"] = False
    if args.models:
        CONFIG["model_dir"] = args.models

    # Telegram credentials (CLI > hard-coded)
    token   = args.tg_token   or TELEGRAM_TOKEN
    chat_id = args.tg_chat    or TELEGRAM_CHAT_ID
    tg_on   = CONFIG["telegram_enabled"]

    print("\n" + "═" * 64)
    print("  INFANT CRY DETECTION & CLASSIFICATION  —  Raspberry Pi")
    print(f"  Mode     : {args.mode.upper()}")
    print(f"  Models   : {CONFIG['model_dir']}")
    print(f"  Telegram : {'ENABLED' if tg_on else 'DISABLED'}")
    print("═" * 64 + "\n")

    # ── Load classifier ──────────────────────────────────────────────────────
    try:
        clf = InfantCryClassifier(model_dir=args.models)
    except (FileNotFoundError, Exception) as e:
        print(f"\n[ERROR] {e}\n")
        sys.exit(1)

    # ── Init Telegram ────────────────────────────────────────────────────────
    tg = TelegramAlerter(
        token   = token,
        chat_id = chat_id,
        enabled = tg_on,
    )

    # ── Route to selected mode ───────────────────────────────────────────────
    if args.mode == "live":
        run_live(clf, tg, duration=args.duration)

    elif args.mode == "file":
        if not args.input:
            print("[ERROR] --input <file_or_dir> is required for file mode\n")
            sys.exit(1)
        run_file(clf, tg, args.input)

    elif args.mode == "batch":
        if not args.input:
            print("[ERROR] --input <dataset_dir> is required for batch mode\n")
            sys.exit(1)
        run_batch(clf, args.input)


if __name__ == "__main__":
    main()