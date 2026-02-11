#!/usr/bin/env python3
"""
Skippy - A JARVIS-style voice assistant for Raspberry Pi.

Listens for a wake word, transcribes speech with faster-whisper,
queries Google Gemini Flash for a response, and speaks it using Piper TTS.

Usage:
    python3 skippy.py

Requires config.json alongside this script. See config.example.json.
"""

import contextlib
import ctypes
import io
import json
import logging
import os
import signal
import sys
import wave
from pathlib import Path

# Suppress ALSA/JACK error spam from PyAudio device enumeration
os.environ["JACK_NO_START_SERVER"] = "1"
try:
    _alsa_err = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int,
                                 ctypes.c_char_p, ctypes.c_int,
                                 ctypes.c_char_p)
    def _alsa_error_handler(filename, line, function, err, fmt):
        pass
    _alsa_handle = _alsa_err(_alsa_error_handler)
    ctypes.cdll.LoadLibrary("libasound.so.2").snd_lib_error_set_handler(_alsa_handle)
except Exception:
    pass

import numpy as np
import pyaudio
import openwakeword
from openwakeword import Model as OwwModel
from faster_whisper import WhisperModel

logger = logging.getLogger("skippy")


def _open_pyaudio():
    """Create PyAudio instance with stderr suppressed (hides JACK noise)."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    os.dup2(devnull, 2)
    try:
        p = pyaudio.PyAudio()
    finally:
        os.dup2(old_stderr, 2)
        os.close(devnull)
        os.close(old_stderr)
    return p


class AudioFeedback:
    """Generates audio feedback beeps using numpy sine waves through pyaudio."""

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.sample_rate = 22050

    def _generate_tone(self, frequency, duration, volume=0.3):
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        tone = np.sin(2 * np.pi * frequency * t) * volume
        fade_samples = int(0.02 * self.sample_rate)
        if fade_samples > 0 and len(tone) > fade_samples * 2:
            tone[:fade_samples] *= np.linspace(0, 1, fade_samples)
            tone[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        return (tone * 32767).astype(np.int16)

    def _play(self, audio_data):
        if not self.enabled:
            return
        p = _open_pyaudio()
        try:
            stream = p.open(format=pyaudio.paInt16, channels=1,
                            rate=self.sample_rate, output=True)
            stream.write(audio_data.tobytes())
            stream.stop_stream()
            stream.close()
        finally:
            p.terminate()

    def beep_listening(self):
        """Short high beep: wake word detected, now listening."""
        self._play(self._generate_tone(880, 0.15))

    def beep_thinking(self):
        """Two quick low beeps: processing."""
        tone1 = self._generate_tone(440, 0.1)
        silence = np.zeros(int(self.sample_rate * 0.05), dtype=np.int16)
        tone2 = self._generate_tone(440, 0.1)
        self._play(np.concatenate([tone1, silence, tone2]))

    def beep_error(self):
        """Descending tone: error occurred."""
        tone1 = self._generate_tone(440, 0.15)
        tone2 = self._generate_tone(330, 0.2)
        self._play(np.concatenate([tone1, tone2]))

    def beep_ready(self):
        """Ascending two-tone: system ready."""
        tone1 = self._generate_tone(523, 0.12)
        tone2 = self._generate_tone(659, 0.15)
        self._play(np.concatenate([tone1, tone2]))


class PlatformDetector:
    """Detects Raspberry Pi model and selects optimal Whisper model size."""

    @staticmethod
    def get_total_ram_gb():
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return kb / (1024 * 1024)
        except (FileNotFoundError, ValueError):
            pass
        return 1.0

    @staticmethod
    def get_pi_model():
        try:
            with open("/proc/device-tree/model") as f:
                return f.read().strip("\x00").strip()
        except FileNotFoundError:
            return "unknown"

    @staticmethod
    def select_whisper_model(config_value):
        """Select Whisper model based on config and available RAM.

        auto selection: tiny (<=4GB), base (<=8GB), small (16GB+)
        """
        if config_value != "auto":
            return config_value

        ram = PlatformDetector.get_total_ram_gb()
        if ram >= 12:
            return "small"
        elif ram >= 6:
            return "base"
        else:
            return "tiny"


class WakeWordListener:
    """Listens for the wake word using openWakeWord."""

    RATE = 16000
    CHUNK = 1280  # 80ms at 16kHz, optimal for openWakeWord
    FORMAT = pyaudio.paInt16
    CHANNELS = 1

    def __init__(self, model_name_or_path, threshold=0.5):
        self.threshold = threshold

        if os.path.isfile(model_name_or_path):
            self.oww_model = OwwModel(wakeword_models=[model_name_or_path])
        else:
            openwakeword.utils.download_models()
            self.oww_model = OwwModel()

        self._pyaudio = None
        self._stream = None

    def start(self):
        self._pyaudio = _open_pyaudio()
        self._stream = self._pyaudio.open(
            format=self.FORMAT, channels=self.CHANNELS,
            rate=self.RATE, input=True,
            frames_per_buffer=self.CHUNK
        )

    def listen_once(self):
        """Read one chunk and check for wake word. Returns True if detected."""
        audio_data = self._stream.read(self.CHUNK, exception_on_overflow=False)
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        self.oww_model.predict(audio_array)

        for model_name in self.oww_model.prediction_buffer.keys():
            scores = list(self.oww_model.prediction_buffer[model_name])
            if scores and scores[-1] > self.threshold:
                self.oww_model.reset()
                return True
        return False

    def stop(self):
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pyaudio:
            self._pyaudio.terminate()
            self._pyaudio = None


class SpeechRecorder:
    """Records user speech after wake word, with silence detection."""

    RATE = 16000
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1

    def __init__(self, silence_threshold=500, silence_duration=1.5,
                 max_duration=10.0):
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.max_duration = max_duration

    def record(self):
        """Record audio until silence or timeout. Returns WAV path or None."""
        p = _open_pyaudio()
        stream = p.open(format=self.FORMAT, channels=self.CHANNELS,
                        rate=self.RATE, input=True,
                        frames_per_buffer=self.CHUNK)

        # Discard first 0.3s to flush beep echo from mic buffer
        warmup_chunks = int(0.3 * self.RATE / self.CHUNK)
        for _ in range(warmup_chunks):
            stream.read(self.CHUNK, exception_on_overflow=False)

        frames = []
        silent_chunks = 0
        speech_chunks = 0
        max_silent = int(self.silence_duration * self.RATE / self.CHUNK)
        max_chunks = int(self.max_duration * self.RATE / self.CHUNK)
        # Require at least 0.5s of speech before silence detection triggers
        min_speech_chunks = int(0.5 * self.RATE / self.CHUNK)

        for _ in range(max_chunks):
            data = stream.read(self.CHUNK, exception_on_overflow=False)
            frames.append(data)
            audio_array = np.frombuffer(data, dtype=np.int16)
            amplitude = np.abs(audio_array).mean()

            if amplitude < self.silence_threshold:
                silent_chunks += 1
            else:
                silent_chunks = 0
                speech_chunks += 1

            if speech_chunks >= min_speech_chunks and \
               silent_chunks >= max_silent:
                break

        stream.stop_stream()
        stream.close()
        p.terminate()

        if speech_chunks == 0:
            return None

        wav_path = "/tmp/skippy_input.wav"
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(self.RATE)
            wf.writeframes(b"".join(frames))

        return wav_path


class Transcriber:
    """Transcribes audio files using faster-whisper."""

    def __init__(self, model_size="tiny"):
        logger.info(f"Loading Whisper model: {model_size}")
        self.model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
            cpu_threads=4
        )
        logger.info("Whisper model loaded")

    def transcribe(self, audio_path):
        """Transcribe audio file to text. Returns empty string on failure."""
        try:
            segments, _ = self.model.transcribe(
                audio_path,
                beam_size=3,
                language="en",
                vad_filter=True,
                condition_on_previous_text=False
            )
            text = " ".join(segment.text for segment in segments).strip()
            return text
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return ""


class Brain:
    """Gemini Flash conversation engine with multi-turn history."""

    def __init__(self, api_key, model_name, system_prompt,
                 max_history_turns=20):
        from google import genai
        from google.genai import types

        self._genai = genai
        self._types = types
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.max_history_turns = max_history_turns
        self._chat = None
        self._turn_count = 0
        self._reset_chat()

    def _reset_chat(self):
        self._chat = self.client.chats.create(
            model=self.model_name,
            config=self._types.GenerateContentConfig(
                system_instruction=self.system_prompt,
            )
        )
        self._turn_count = 0

    def ask(self, user_text):
        """Send user message and get response. Handles errors with retry."""
        try:
            response = self._chat.send_message(user_text)
            self._turn_count += 1

            if self._turn_count >= self.max_history_turns:
                logger.info("Conversation history limit reached, resetting")
                self._reset_chat()

            return response.text
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            try:
                self._reset_chat()
                response = self._chat.send_message(user_text)
                self._turn_count = 1
                return response.text
            except Exception as retry_err:
                logger.error(f"Gemini retry failed: {retry_err}")
                return "I'm sorry, I'm having trouble connecting right now."


class Speaker:
    """Text-to-speech using Piper TTS."""

    def __init__(self, voice_model_path):
        from piper.voice import PiperVoice

        logger.info(f"Loading Piper voice: {voice_model_path}")
        self.voice = PiperVoice.load(voice_model_path)
        self.sample_rate = self.voice.config.sample_rate
        logger.info("Piper voice loaded")

    def speak(self, text):
        """Synthesize text and play through speakers."""
        if not text:
            return

        try:
            # Synthesize to in-memory WAV buffer
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                self.voice.synthesize(text, wf)

            # Play the WAV data
            buf.seek(0)
            with wave.open(buf, "rb") as wf:
                p = _open_pyaudio()
                stream = p.open(
                    format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True
                )
                data = wf.readframes(1024)
                while data:
                    stream.write(data)
                    data = wf.readframes(1024)
                stream.stop_stream()
                stream.close()
                p.terminate()
        except Exception as e:
            logger.error(f"TTS playback error: {e}")


class Skippy:
    """Main orchestrator for the Skippy voice assistant."""

    def __init__(self, config):
        self.config = config
        self.running = False

        self.feedback = AudioFeedback(
            enabled=config.get("audio_feedback", True)
        )

        whisper_size = PlatformDetector.select_whisper_model(
            config.get("whisper_model", "auto")
        )
        pi_model = PlatformDetector.get_pi_model()
        ram = PlatformDetector.get_total_ram_gb()
        logger.info(
            f"Platform: {pi_model}, RAM: {ram:.1f}GB, "
            f"Whisper model: {whisper_size}"
        )

        self.wake_word = WakeWordListener(
            model_name_or_path=config.get("wake_word_model", "hey_jarvis"),
            threshold=config.get("wake_word_threshold", 0.5)
        )
        self.recorder = SpeechRecorder(
            silence_threshold=config.get("silence_threshold", 500),
            silence_duration=config.get("silence_duration", 1.5),
            max_duration=config.get("recording_timeout", 10)
        )
        self.transcriber = Transcriber(model_size=whisper_size)
        self.brain = Brain(
            api_key=config["gemini_api_key"],
            model_name=config.get("gemini_model", "gemini-2.0-flash"),
            system_prompt=config.get(
                "system_prompt",
                "You are Skippy, a helpful voice assistant. "
                "Keep responses to 1-3 sentences."
            ),
            max_history_turns=config.get("max_history_turns", 20)
        )

        voice_name = config.get("piper_voice", "en_US-lessac-medium")
        voice_path = Path(__file__).parent / "voices" / f"{voice_name}.onnx"
        self.speaker = Speaker(str(voice_path))

    def _handle_signal(self, signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

    def run(self):
        """Main loop: wake word -> record -> transcribe -> think -> speak."""
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        self.running = True
        logger.info("Skippy is starting up...")

        self.wake_word.start()
        self.feedback.beep_ready()
        logger.info("Skippy is ready. Listening for wake word...")

        try:
            while self.running:
                if not self.wake_word.listen_once():
                    continue

                logger.info("Wake word detected!")
                self.wake_word.stop()

                self.feedback.beep_listening()

                logger.info("Recording...")
                audio_path = self.recorder.record()

                if audio_path is None:
                    logger.info("No speech detected, returning to listening")
                    self.wake_word.start()
                    continue

                self.feedback.beep_thinking()
                logger.info("Transcribing...")
                user_text = self.transcriber.transcribe(audio_path)
                logger.info(f"User said: {user_text}")

                if not user_text:
                    logger.info("Empty transcription, returning to listening")
                    self.wake_word.start()
                    continue

                logger.info("Thinking...")
                response_text = self.brain.ask(user_text)
                logger.info(f"Skippy says: {response_text}")

                logger.info("Speaking...")
                self.speaker.speak(response_text)

                self.wake_word.start()
                logger.info("Listening for wake word...")

        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
            self.feedback.beep_error()
        finally:
            self.wake_word.stop()
            logger.info("Skippy shut down.")


def load_config():
    """Load configuration from config.json alongside this script."""
    script_dir = Path(__file__).parent
    config_path = script_dir / "config.json"

    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        print("Please create config.json from config.example.json:")
        print(f"  cp {script_dir}/config.example.json {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)

    if not config.get("gemini_api_key") or \
       config["gemini_api_key"] == "YOUR_GEMINI_API_KEY":
        print("Error: Please set your Gemini API key in config.json")
        print("Get a free key at: https://aistudio.google.com")
        sys.exit(1)

    return config


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    logger.info("Loading configuration...")
    config = load_config()

    skippy = Skippy(config)
    skippy.run()


if __name__ == "__main__":
    main()
