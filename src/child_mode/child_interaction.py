"""
interaction.py — Voice Interaction Module for Raspberry Pi Bot
Supports: USB Microphone / USB Camera Mic  →  STT  →  AI Response  →  TTS  →  USB Speaker
Optimized for: Raspberry Pi 4 (2GB+ RAM), lightweight & responsive
Compatible with: Child & Elderly interaction profiles
"""

import os
import sys
import time
import wave
import queue
import logging
import threading
import subprocess
import audioop
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

# ── Third-party (install via: pip install -r requirements.txt) ──────────────
import pyaudio
import numpy as np

# STT: Vosk (fully offline, lightweight)
from vosk import Model as VoskModel, KaldiRecognizer
import json

# TTS: pyttsx3 (offline) or gTTS (online, better quality)
try:
    import pyttsx3
    TTS_ENGINE = "pyttsx3"
except ImportError:
    TTS_ENGINE = "gtts"

# Optional: OpenAI / local LLM for smarter responses
try:
    import openai
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/tmp/interaction.log"),
    ],
)
log = logging.getLogger("interaction")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         CONFIGURATION                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class UserProfile(Enum):
    CHILD   = "child"
    ELDERLY = "elderly"
    DEFAULT = "default"


@dataclass
class AudioConfig:
    sample_rate:    int   = 16000   # Vosk works best at 16 kHz
    channels:       int   = 1       # Mono
    chunk_size:     int   = 4096    # Frames per buffer (larger = less CPU on Pi)
    format:         int   = pyaudio.paInt16
    silence_thresh: int   = 500     # RMS threshold to detect silence
    silence_sec:    float = 1.8     # Seconds of silence before end-of-speech
    max_record_sec: float = 15.0    # Safety limit per utterance
    # Device indices — None = auto-detect
    input_device:   Optional[int] = None   # USB mic / camera mic
    output_device:  Optional[int] = None   # USB speaker


@dataclass
class BotConfig:
    # Vosk model path (download from https://alphacephei.com/vosk/models)
    # Recommended lightweight model: vosk-model-small-en-us-0.15 (~40 MB)
    vosk_model_path: str = "/home/pi/vosk-model-small-en-us-0.15"

    # OpenAI (leave empty for rule-based offline mode)
    openai_api_key:  str = os.getenv("OPENAI_API_KEY", "")
    openai_model:    str = "gpt-3.5-turbo"       # lightweight & cheap

    # TTS settings
    tts_rate_child:   int = 140     # Slower, clearer for kids
    tts_rate_elderly: int = 120     # Even slower for elderly
    tts_volume:     float = 1.0
    tts_voice_index:  int = 0       # 0 = first available voice

    # Interaction
    wake_words:      list = field(default_factory=lambda: ["hey bot", "hello bot", "ok bot"])
    user_profile:    UserProfile = UserProfile.DEFAULT
    response_timeout: float = 8.0   # Max seconds to wait for AI response

    # Paths
    audio_cache_dir: str = "/tmp/bot_audio"
    conversation_log: str = "/tmp/bot_conversation.json"


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         AUDIO UTILITIES                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class AudioDeviceManager:
    """Auto-detect USB mic and USB speaker on Raspberry Pi."""

    def __init__(self):
        self.pa = pyaudio.PyAudio()

    def list_devices(self) -> list[dict]:
        devices = []
        for i in range(self.pa.get_device_count()):
            info = self.pa.get_device_info_by_index(i)
            devices.append(info)
            log.debug(f"  [{i}] {info['name']}  in={info['maxInputChannels']}  out={info['maxOutputChannels']}")
        return devices

    def find_usb_input(self) -> Optional[int]:
        """Return device index of first USB/UVC input device."""
        keywords = ["usb", "uvc", "webcam", "camera", "logitech", "creative", "blue"]
        for i, d in enumerate(self.list_devices()):
            if d["maxInputChannels"] > 0:
                name = d["name"].lower()
                if any(k in name for k in keywords):
                    log.info(f"✔ USB mic found: [{i}] {d['name']}")
                    return i
        # Fallback: first available input
        for i, d in enumerate(self.list_devices()):
            if d["maxInputChannels"] > 0:
                log.info(f"✔ Mic fallback: [{i}] {d['name']}")
                return i
        return None

    def find_usb_output(self) -> Optional[int]:
        """Return device index of first USB/HDMI output device."""
        keywords = ["usb", "speaker", "audio", "sound", "usb audio"]
        for i, d in enumerate(self.list_devices()):
            if d["maxOutputChannels"] > 0:
                name = d["name"].lower()
                if any(k in name for k in keywords):
                    log.info(f"✔ USB speaker found: [{i}] {d['name']}")
                    return i
        for i, d in enumerate(self.list_devices()):
            if d["maxOutputChannels"] > 0:
                log.info(f"✔ Speaker fallback: [{i}] {d['name']}")
                return i
        return None

    def terminate(self):
        self.pa.terminate()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                     SPEECH-TO-TEXT  (Vosk — Offline)                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class SpeechRecognizer:
    """
    Lightweight offline STT using Vosk.
    Streams audio in real-time, detects speech end via silence detection.
    """

    def __init__(self, cfg: BotConfig, audio_cfg: AudioConfig):
        self.cfg       = cfg
        self.audio_cfg = audio_cfg
        self._load_model()

    def _load_model(self):
        model_path = Path(self.cfg.vosk_model_path)
        if not model_path.exists():
            log.error(
                f"Vosk model not found at {model_path}.\n"
                "Download: https://alphacephei.com/vosk/models\n"
                "Suggested: vosk-model-small-en-us-0.15"
            )
            sys.exit(1)
        log.info(f"Loading Vosk model from {model_path} …")
        self.model      = VoskModel(str(model_path))
        self.recognizer = KaldiRecognizer(self.model, self.audio_cfg.sample_rate)
        self.recognizer.SetWords(True)
        log.info("Vosk model ready ✔")

    def listen(self, pa: pyaudio.PyAudio, input_device: Optional[int]) -> str:
        """
        Open mic stream, listen until silence, return transcribed text.
        Returns empty string if nothing was heard.
        """
        stream = pa.open(
            format=self.audio_cfg.format,
            channels=self.audio_cfg.channels,
            rate=self.audio_cfg.sample_rate,
            input=True,
            input_device_index=input_device,
            frames_per_buffer=self.audio_cfg.chunk_size,
        )

        log.info("🎙 Listening …")
        frames          = []
        silent_chunks   = 0
        speaking        = False
        max_chunks      = int(self.audio_cfg.max_record_sec *
                              self.audio_cfg.sample_rate / self.audio_cfg.chunk_size)
        silence_chunks  = int(self.audio_cfg.silence_sec *
                              self.audio_cfg.sample_rate / self.audio_cfg.chunk_size)

        self.recognizer = KaldiRecognizer(self.model, self.audio_cfg.sample_rate)

        try:
            for _ in range(max_chunks):
                data = stream.read(self.audio_cfg.chunk_size, exception_on_overflow=False)
                rms  = audioop.rms(data, 2)

                if rms > self.audio_cfg.silence_thresh:
                    speaking      = True
                    silent_chunks = 0
                    frames.append(data)
                elif speaking:
                    frames.append(data)
                    silent_chunks += 1
                    if silent_chunks >= silence_chunks:
                        break  # End of utterance
        finally:
            stream.stop_stream()
            stream.close()

        if not frames:
            return ""

        # Feed frames to Vosk
        audio_data = b"".join(frames)
        self.recognizer.AcceptWaveform(audio_data)
        result = json.loads(self.recognizer.FinalResult())
        text   = result.get("text", "").strip()
        if text:
            log.info(f"👂 Heard: \"{text}\"")
        return text


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                      TEXT-TO-SPEECH                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class Speaker:
    """
    Lightweight TTS — uses pyttsx3 (offline) or falls back to espeak.
    Plays through the USB speaker.
    """

    def __init__(self, cfg: BotConfig, audio_cfg: AudioConfig,
                 output_device: Optional[int] = None):
        self.cfg           = cfg
        self.audio_cfg     = audio_cfg
        self.output_device = output_device
        self._init_tts()
        os.makedirs(cfg.audio_cache_dir, exist_ok=True)

    def _init_tts(self):
        global TTS_ENGINE
        if TTS_ENGINE == "pyttsx3":
            try:
                self.engine = pyttsx3.init()
                voices = self.engine.getProperty("voices")
                if voices:
                    self.engine.setProperty("voice", voices[self.cfg.tts_voice_index].id)
                self.engine.setProperty("volume", self.cfg.tts_volume)
                log.info("TTS engine: pyttsx3 (offline) ✔")
            except Exception as e:
                log.warning(f"pyttsx3 init failed ({e}), falling back to espeak")
                TTS_ENGINE = "espeak"

    def set_profile(self, profile: UserProfile):
        rate = (self.cfg.tts_rate_child   if profile == UserProfile.CHILD   else
                self.cfg.tts_rate_elderly if profile == UserProfile.ELDERLY else 150)
        if TTS_ENGINE == "pyttsx3":
            self.engine.setProperty("rate", rate)

    def speak(self, text: str):
        if not text.strip():
            return
        log.info(f"🔊 Speaking: \"{text}\"")

        if TTS_ENGINE == "pyttsx3":
            self._speak_pyttsx3(text)
        else:
            self._speak_espeak(text)

    def _speak_pyttsx3(self, text: str):
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            log.error(f"pyttsx3 error: {e}")
            self._speak_espeak(text)

    def _speak_espeak(self, text: str):
        """Fallback: espeak (always available on Raspberry Pi OS)."""
        rate   = self.cfg.tts_rate_elderly
        cmd    = ["espeak", "-s", str(rate), "-v", "en+f3", text]
        try:
            subprocess.run(cmd, timeout=30, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            log.error(f"espeak failed: {e}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                        RESPONSE ENGINE                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class ResponseEngine:
    """
    Generates bot responses.
    Mode 1 (online):  OpenAI GPT with profile-aware system prompt
    Mode 2 (offline): Rule-based pattern matching (no internet needed)
    """

    # ── Offline rule-based responses ────────────────────────────────────────
    RULES = {
        # Greetings
        ("hello", "hi", "hey", "good morning", "good evening", "good afternoon"):
            {"child":   "Hello there! 😊 I'm so happy to see you! What would you like to talk about today?",
             "elderly": "Hello! How wonderful to hear from you. How are you feeling today?",
             "default": "Hello! How can I help you today?"},

        # How are you
        ("how are you", "are you okay", "how do you feel"):
            {"child":   "I'm doing great, thanks for asking! Are you having a fun day?",
             "elderly": "I'm doing very well, thank you so much for asking! And how are you?",
             "default": "I'm functioning well, thank you!"},

        # Time
        ("what time", "current time", "tell me the time"):
            None,   # Handled dynamically

        # Weather placeholder
        ("weather", "temperature outside", "is it raining"):
            {"child":   "I'm not sure about the weather right now. Maybe look outside the window!",
             "elderly": "I don't have weather info right now, but you can check by looking outside or asking someone nearby.",
             "default": "I don't have live weather data right now."},

        # Jokes
        ("joke", "tell me a joke", "make me laugh", "funny"):
            {"child":   "Why did the scarecrow win an award? Because he was outstanding in his field! 😄",
             "elderly": "Why don't scientists trust atoms? Because they make up everything! 😄",
             "default": "Why did the bicycle fall over? Because it was two-tired! 😄"},

        # Stories
        ("story", "tell me a story", "bedtime story"):
            {"child":   "Once upon a time, in a land full of colorful butterflies and talking trees, there lived a brave little rabbit named Pip who wanted to climb the tallest mountain. Would you like to hear more?",
             "elderly": "Let me tell you a little story. Once there was a kind gardener who grew the most beautiful roses in the village. Everyone who passed by would stop to smell them and smile. Isn't it wonderful how small things can bring so much joy?",
             "default": "Once upon a time there was a curious adventurer who set out to discover the world."},

        # Help / what can you do
        ("help", "what can you do", "commands", "options"):
            {"child":   "I can tell you jokes, stories, the time, or just chat with you! What sounds fun?",
             "elderly": "I can chat with you, tell jokes, stories, or remind you of things. Just talk to me!",
             "default": "I can answer questions, tell jokes and stories, and have a conversation with you."},

        # Goodbye
        ("bye", "goodbye", "see you", "see you later", "good night", "goodnight"):
            {"child":   "Goodbye! Have a super-duper day! 👋",
             "elderly": "Goodbye, dear! Take care of yourself. I'm always here if you need me. 👋",
             "default": "Goodbye! Have a great day!"},

        # Feeling sad
        ("sad", "not feeling well", "i am tired", "im tired", "feeling lonely", "i feel lonely"):
            {"child":   "Aww, I'm sorry you feel that way. Want me to tell you a funny joke to cheer you up?",
             "elderly": "I'm sorry to hear that. Remember, you're not alone — I'm here to keep you company. Would you like to talk about it?",
             "default": "I'm sorry to hear that. Is there anything I can do to help?"},

        # Reminders
        ("remind me", "set reminder", "alarm"):
            {"child":   "I can't set alarms yet, but ask a grown-up to help you with that!",
             "elderly": "I can't set reminders just yet, but that's a great idea for the future! For now, maybe write it on a note.",
             "default": "I don't support reminders yet, but that feature is coming!"},
    }

    CHILD_SYSTEM_PROMPT = (
        "You are a friendly, warm, and playful robot companion for children aged 4-12. "
        "Use very simple words. Keep answers short (2-3 sentences). Use encouragement and fun. "
        "Avoid anything scary, violent, or adult. Add enthusiasm with exclamation marks."
    )

    ELDERLY_SYSTEM_PROMPT = (
        "You are a caring, patient, and respectful companion for elderly people. "
        "Speak clearly and warmly. Keep answers concise but kind. "
        "Offer help, companionship, and gentle humor. Avoid technical jargon. "
        "Always be respectful and attentive to their needs."
    )

    DEFAULT_SYSTEM_PROMPT = (
        "You are a helpful, friendly voice assistant on a Raspberry Pi robot. "
        "Give clear, concise answers. Be conversational and natural."
    )

    def __init__(self, cfg: BotConfig):
        self.cfg     = cfg
        self.history = []   # Conversation history for context
        if AI_AVAILABLE and cfg.openai_api_key:
            openai.api_key = cfg.openai_api_key
            log.info("Response engine: OpenAI GPT ✔")
        else:
            log.info("Response engine: Offline rule-based ✔")

    def _system_prompt(self, profile: UserProfile) -> str:
        return {
            UserProfile.CHILD:   self.CHILD_SYSTEM_PROMPT,
            UserProfile.ELDERLY: self.ELDERLY_SYSTEM_PROMPT,
        }.get(profile, self.DEFAULT_SYSTEM_PROMPT)

    def _rule_based(self, text: str, profile: UserProfile) -> str:
        t   = text.lower().strip()
        key = profile.value if profile != UserProfile.DEFAULT else "default"

        # Dynamic responses
        if any(w in t for w in ("what time", "current time", "tell me the time")):
            now = datetime.now().strftime("%I:%M %p")
            if profile == UserProfile.CHILD:
                return f"It's {now}! Time flies when we're having fun!"
            elif profile == UserProfile.ELDERLY:
                return f"The current time is {now}. Hope you're enjoying the day!"
            return f"The time is {now}."

        # Match rules
        for keywords, responses in self.RULES.items():
            if any(k in t for k in keywords):
                if responses is None:
                    continue
                return responses.get(key, responses.get("default", "I'm here to help!"))

        # Fallback
        fallbacks = {
            "child":   "Hmm, I'm not sure about that! Can you ask me something else? 😊",
            "elderly": "That's a good question. I'm not quite sure about that one. Is there anything else I can help you with?",
            "default": "I'm not sure about that. Could you ask in a different way?",
        }
        return fallbacks.get(key, fallbacks["default"])

    def get_response(self, text: str, profile: UserProfile) -> str:
        if not text.strip():
            return ""

        # Try online AI first
        if AI_AVAILABLE and self.cfg.openai_api_key:
            try:
                return self._openai_response(text, profile)
            except Exception as e:
                log.warning(f"OpenAI error ({e}), falling back to rules")

        return self._rule_based(text, profile)

    def _openai_response(self, text: str, profile: UserProfile) -> str:
        self.history.append({"role": "user", "content": text})
        # Keep last 10 turns to save tokens
        trimmed = self.history[-10:]

        response = openai.chat.completions.create(
            model   = self.cfg.openai_model,
            messages= [{"role": "system", "content": self._system_prompt(profile)}] + trimmed,
            max_tokens    = 150,
            temperature   = 0.7,
            timeout       = self.cfg.response_timeout,
        )
        reply = response.choices[0].message.content.strip()
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def reset_history(self):
        self.history.clear()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                     CONVERSATION LOGGER                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class ConversationLogger:
    def __init__(self, path: str):
        self.path = path
        self.log  = []

    def record(self, role: str, text: str, profile: UserProfile):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "profile":   profile.value,
            "role":      role,
            "text":      text,
        }
        self.log.append(entry)
        self._save()

    def _save(self):
        try:
            with open(self.path, "w") as f:
                json.dump(self.log, f, indent=2)
        except Exception as e:
            log.warning(f"Could not save conversation log: {e}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                      INTERACTION CONTROLLER                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class InteractionController:
    """
    Main controller — wires together mic → STT → AI → TTS → speaker.
    Supports:
      - Wake-word activation (optional)
      - Always-on mode (no wake word needed)
      - Auto user-profile detection (child / elderly)
      - LED / GPIO feedback hooks (extend as needed)
    """

    def __init__(self,
                 bot_cfg:   BotConfig   = None,
                 audio_cfg: AudioConfig = None):
        self.bot_cfg   = bot_cfg   or BotConfig()
        self.audio_cfg = audio_cfg or AudioConfig()

        # Auto-detect devices
        device_mgr = AudioDeviceManager()
        if self.audio_cfg.input_device is None:
            self.audio_cfg.input_device  = device_mgr.find_usb_input()
        if self.audio_cfg.output_device is None:
            self.audio_cfg.output_device = device_mgr.find_usb_output()
        device_mgr.terminate()

        # PyAudio instance (shared)
        self.pa = pyaudio.PyAudio()

        # Sub-systems
        self.stt     = SpeechRecognizer(self.bot_cfg, self.audio_cfg)
        self.speaker = Speaker(self.bot_cfg, self.audio_cfg, self.audio_cfg.output_device)
        self.engine  = ResponseEngine(self.bot_cfg)
        self.clog    = ConversationLogger(self.bot_cfg.conversation_log)

        self.profile  = self.bot_cfg.user_profile
        self.running  = False
        self._lock    = threading.Lock()

        # Set initial TTS speed
        self.speaker.set_profile(self.profile)

    # ── Profile helpers ──────────────────────────────────────────────────────

    def set_profile(self, profile: UserProfile):
        """Switch between CHILD / ELDERLY / DEFAULT at runtime."""
        self.profile = profile
        self.speaker.set_profile(profile)
        self.engine.reset_history()
        log.info(f"Profile switched → {profile.value}")

    def _detect_profile_from_text(self, text: str):
        """Very simple keyword-based profile hint (override if you have face-age detection)."""
        t = text.lower()
        if any(w in t for w in ("grandma", "grandpa", "grandparent", "elderly", "senior", "old person")):
            self.set_profile(UserProfile.ELDERLY)
        elif any(w in t for w in ("child", "kid", "boy", "girl", "son", "daughter")):
            self.set_profile(UserProfile.CHILD)

    # ── Wake-word detection ──────────────────────────────────────────────────

    def _has_wake_word(self, text: str) -> bool:
        t = text.lower()
        return any(w in t for w in self.bot_cfg.wake_words)

    def _strip_wake_word(self, text: str) -> str:
        t = text.lower()
        for w in self.bot_cfg.wake_words:
            t = t.replace(w, "").strip()
        return t

    # ── LED / GPIO hook (extend for physical feedback) ───────────────────────

    def _led_thinking(self):
        """Override to flash an LED while the bot is thinking."""
        pass

    def _led_listening(self):
        """Override to light an LED while the mic is open."""
        pass

    def _led_off(self):
        pass

    # ── Single interaction turn ───────────────────────────────────────────────

    def _process_turn(self, user_text: str):
        """STT result → response → TTS."""
        with self._lock:
            self._detect_profile_from_text(user_text)
            self.clog.record("user", user_text, self.profile)

            # Thinking indicator
            self._led_thinking()
            response = self.engine.get_response(user_text, self.profile)

            if not response:
                return

            self._led_off()
            self.clog.record("bot", response, self.profile)
            self.speaker.speak(response)

    # ── Main loop (always-on, no wake word) ──────────────────────────────────

    def run_always_on(self):
        """
        Continuous listen-respond loop.
        Best for dedicated companion robots with no privacy concern.
        """
        log.info("▶ Always-on interaction mode started")
        self.speaker.speak(self._greeting())
        self.running = True

        try:
            while self.running:
                self._led_listening()
                text = self.stt.listen(self.pa, self.audio_cfg.input_device)
                self._led_off()

                if not text:
                    continue

                # Check for special commands
                if self._handle_commands(text):
                    continue

                self._process_turn(text)

        except KeyboardInterrupt:
            log.info("Interaction stopped by user.")
        finally:
            self._cleanup()

    # ── Main loop (wake-word activated) ──────────────────────────────────────

    def run_wake_word(self):
        """
        Listen for wake word first, then open full response window.
        Saves CPU / power — better for battery-powered builds.
        """
        wake_str = " / ".join(self.bot_cfg.wake_words)
        log.info(f"▶ Wake-word mode — say: \"{wake_str}\"")
        print(f"\n  💬 Say '{wake_str}' to activate …\n")
        self.running = True

        try:
            while self.running:
                # Passive listen (short window)
                self._led_listening()
                text = self.stt.listen(self.pa, self.audio_cfg.input_device)
                self._led_off()

                if not text:
                    continue

                if self._has_wake_word(text):
                    clean = self._strip_wake_word(text)
                    self.speaker.speak("Yes, I'm listening!")
                    log.info("Wake word detected ✔")

                    # If the wake word contained a command already, process it
                    if clean:
                        self._process_turn(clean)
                    else:
                        # Open a second listen window for the actual question
                        self._led_listening()
                        followup = self.stt.listen(self.pa, self.audio_cfg.input_device)
                        self._led_off()
                        if followup:
                            self._process_turn(followup)

        except KeyboardInterrupt:
            log.info("Interaction stopped by user.")
        finally:
            self._cleanup()

    # ── Special built-in commands ─────────────────────────────────────────────

    def _handle_commands(self, text: str) -> bool:
        """Returns True if text was a special command (skip normal response)."""
        t = text.lower().strip()

        if t in ("switch to child mode", "child mode"):
            self.set_profile(UserProfile.CHILD)
            self.speaker.speak("Switched to child mode. Hi there little one!")
            return True

        if t in ("switch to elderly mode", "elderly mode", "senior mode"):
            self.set_profile(UserProfile.ELDERLY)
            self.speaker.speak("Switched to elderly companion mode. Hello, how can I help?")
            return True

        if t in ("exit", "shutdown", "power off", "quit"):
            self.speaker.speak("Goodbye! Shutting down now.")
            self.running = False
            return True

        if t in ("reset", "clear history", "forget everything"):
            self.engine.reset_history()
            self.speaker.speak("Conversation history cleared.")
            return True

        return False

    # ── Greeting based on time of day ─────────────────────────────────────────

    def _greeting(self) -> str:
        hour = datetime.now().hour
        tod  = "morning" if hour < 12 else "afternoon" if hour < 17 else "evening"
        if self.profile == UserProfile.CHILD:
            return f"Good {tod}! I'm your robot buddy! I'm so excited to chat with you today!"
        elif self.profile == UserProfile.ELDERLY:
            return f"Good {tod}! I'm your voice companion. I'm here to chat, tell stories, or just keep you company."
        return f"Good {tod}! I'm your voice assistant. How can I help?"

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def _cleanup(self):
        log.info("Cleaning up audio resources …")
        try:
            self.pa.terminate()
        except Exception:
            pass
        log.info("Interaction module shut down.")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                        ENTRY POINT                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Raspberry Pi Voice Interaction Bot")
    parser.add_argument("--profile",    choices=["child", "elderly", "default"],
                        default="default", help="User profile")
    parser.add_argument("--wake-word",  action="store_true",
                        help="Require wake word to activate (saves CPU)")
    parser.add_argument("--model",      default="/home/pi/vosk-model-small-en-us-0.15",
                        help="Path to Vosk model directory")
    parser.add_argument("--silence",    type=float, default=1.8,
                        help="Seconds of silence before end-of-speech")
    parser.add_argument("--list-devices", action="store_true",
                        help="List audio devices and exit")
    args = parser.parse_args()

    # List devices mode
    if args.list_devices:
        mgr = AudioDeviceManager()
        print("\n── Audio Devices ──────────────────────────────────")
        for i, d in enumerate(mgr.list_devices()):
            role = []
            if d["maxInputChannels"]  > 0: role.append("IN")
            if d["maxOutputChannels"] > 0: role.append("OUT")
            print(f"  [{i:2}] {'/'.join(role):6}  {d['name']}")
        print()
        mgr.terminate()
        return

    # Build configs
    bot_cfg = BotConfig(
        vosk_model_path = args.model,
        user_profile    = UserProfile(args.profile),
    )
    audio_cfg = AudioConfig(silence_sec=args.silence)

    # Start controller
    ctrl = InteractionController(bot_cfg, audio_cfg)
    ctrl.set_profile(UserProfile(args.profile))

    if args.wake_word:
        ctrl.run_wake_word()
    else:
        ctrl.run_always_on()


if __name__ == "__main__":
    main()