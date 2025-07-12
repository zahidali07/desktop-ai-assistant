#!/usr/bin/env python3
"""
Voice Input/Output Module for Desktop AI Assistant

This module handles all voice-related functionality including:
- Speech recognition (speech-to-text)
- Text-to-speech synthesis
- Wake word detection
- Audio input/output management
- Continuous listening capabilities

Author: Desktop AI Assistant Project
Version: 1.0.0
"""

import asyncio
import logging
import threading
import time
import wave
from typing import Optional, Callable, Dict, Any
from pathlib import Path

try:
    import speech_recognition as sr
    import pyttsx3
    import sounddevice as sd
    import soundfile as sf
    import numpy as np
    from vosk import Model, KaldiRecognizer
except ImportError as e:
    logging.error(f"Required audio libraries not installed: {e}")
    logging.error("Please install: pip install speechrecognition pyttsx3 sounddevice soundfile vosk")

logger = logging.getLogger(__name__)


class VoiceIO:
    """
    Voice Input/Output handler for the Desktop AI Assistant
    """
    
    def __init__(self, config):
        self.config = config
        self.recognizer = None
        self.microphone = None
        self.tts_engine = None
        self.vosk_model = None
        self.vosk_recognizer = None
        self.is_listening = False
        self.wake_words = ["hey assistant", "hello assistant", "assistant"]
        
        # Audio settings
        self.sample_rate = 16000
        self.chunk_duration = 0.5  # seconds
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        
        # Initialize components
        self._initialize_speech_recognition()
        self._initialize_text_to_speech()
        self._initialize_vosk()
    
    def _initialize_speech_recognition(self):
        """Initialize speech recognition components"""
        try:
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            
            # Adjust for ambient noise
            with self.microphone as source:
                logger.info("Calibrating microphone for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                
            logger.info("Speech recognition initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize speech recognition: {e}")
            raise
    
    def _initialize_text_to_speech(self):
        """Initialize text-to-speech engine"""
        try:
            self.tts_engine = pyttsx3.init()
            
            # Configure TTS settings
            voices = self.tts_engine.getProperty('voices')
            if voices:
                # Prefer female voice if available
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
                else:
                    self.tts_engine.setProperty('voice', voices[0].id)
            
            # Set speech rate and volume
            self.tts_engine.setProperty('rate', self.config.TTS_RATE if hasattr(self.config, 'TTS_RATE') else 200)
            self.tts_engine.setProperty('volume', self.config.TTS_VOLUME if hasattr(self.config, 'TTS_VOLUME') else 0.8)
            
            logger.info("Text-to-speech initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize text-to-speech: {e}")
            raise
    
    def _initialize_vosk(self):
        """Initialize Vosk for offline speech recognition"""
        try:
            # Download and use a small Vosk model for wake word detection
            model_path = Path("models/vosk-model-small-en-us-0.15")
            
            if model_path.exists():
                self.vosk_model = Model(str(model_path))
                self.vosk_recognizer = KaldiRecognizer(self.vosk_model, self.sample_rate)
                logger.info("Vosk offline recognition initialized")
            else:
                logger.warning("Vosk model not found. Download from https://alphacephei.com/vosk/models")
                
        except Exception as e:
            logger.warning(f"Vosk initialization failed: {e}. Falling back to online recognition.")
    
    async def listen_for_wake_word(self, timeout: float = 1.0) -> Optional[bytes]:
        """Listen for wake word activation"""
        try:
            with self.microphone as source:
                logger.debug("Listening for wake word...")
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=3)
                
                # Convert to text and check for wake words
                try:
                    text = self.recognizer.recognize_google(audio).lower()
                    logger.debug(f"Heard: {text}")
                    
                    for wake_word in self.wake_words:
                        if wake_word in text:
                            logger.info(f"Wake word detected: {wake_word}")
                            return audio.get_raw_data()
                            
                except sr.UnknownValueError:
                    pass  # No speech detected
                except sr.RequestError as e:
                    logger.error(f"Speech recognition error: {e}")
                    
            return None
            
        except sr.WaitTimeoutError:
            return None
        except Exception as e:
            logger.error(f"Error in wake word detection: {e}")
            return None
    
    async def listen_continuous(self, duration: float = 5.0) -> Optional[bytes]:
        """Listen continuously for voice input"""
        try:
            with self.microphone as source:
                logger.info("Listening for voice input...")
                audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=duration)
                return audio.get_raw_data()
                
        except sr.WaitTimeoutError:
            return None
        except Exception as e:
            logger.error(f"Error in continuous listening: {e}")
            return None
    
    async def speech_to_text(self, audio_data: bytes) -> Optional[str]:
        """Convert speech audio to text"""
        try:
            # Create AudioData object from raw bytes
            audio = sr.AudioData(audio_data, self.sample_rate, 2)
            
            # Try multiple recognition services
            recognition_methods = [
                ("Google", self.recognizer.recognize_google),
                ("Sphinx", self.recognizer.recognize_sphinx),
            ]
            
            for method_name, method in recognition_methods:
                try:
                    text = method(audio)
                    logger.info(f"Speech recognized using {method_name}: {text}")
                    return text
                    
                except sr.UnknownValueError:
                    logger.debug(f"{method_name} could not understand audio")
                    continue
                except sr.RequestError as e:
                    logger.warning(f"{method_name} recognition error: {e}")
                    continue
            
            # Try Vosk if available
            if self.vosk_recognizer:
                try:
                    # Convert audio for Vosk
                    audio_np = np.frombuffer(audio_data, dtype=np.int16)
                    
                    if self.vosk_recognizer.AcceptWaveform(audio_np.tobytes()):
                        result = self.vosk_recognizer.Result()
                        import json
                        result_dict = json.loads(result)
                        text = result_dict.get('text', '')
                        if text:
                            logger.info(f"Speech recognized using Vosk: {text}")
                            return text
                            
                except Exception as e:
                    logger.warning(f"Vosk recognition error: {e}")
            
            logger.warning("No speech recognition method succeeded")
            return None
            
        except Exception as e:
            logger.error(f"Error in speech-to-text conversion: {e}")
            return None
    
    async def text_to_speech(self, text: str) -> bool:
        """Convert text to speech and play it"""
        try:
            logger.info(f"Speaking: {text}")
            
            # Run TTS in a separate thread to avoid blocking
            def speak():
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            
            # Execute in thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, speak)
            
            return True
            
        except Exception as e:
            logger.error(f"Error in text-to-speech: {e}")
            return False
    
    def start_background_listening(self, callback: Callable[[str], None]):
        """Start background listening thread"""
        def listen_loop():
            self.is_listening = True
            logger.info("Background listening started")
            
            while self.is_listening:
                try:
                    # Listen for wake word
                    audio_data = asyncio.run(self.listen_for_wake_word())
                    
                    if audio_data:
                        # Convert to text
                        text = asyncio.run(self.speech_to_text(audio_data))
                        
                        if text:
                            callback(text)
                    
                    time.sleep(0.1)  # Small delay to prevent excessive CPU usage
                    
                except Exception as e:
                    logger.error(f"Error in background listening: {e}")
                    time.sleep(1)
        
        # Start listening thread
        self.listen_thread = threading.Thread(target=listen_loop, daemon=True)
        self.listen_thread.start()
    
    def stop_background_listening(self):
        """Stop background listening"""
        self.is_listening = False
        logger.info("Background listening stopped")
    
    def set_wake_words(self, wake_words: list):
        """Set custom wake words"""
        self.wake_words = [word.lower() for word in wake_words]
        logger.info(f"Wake words updated: {self.wake_words}")
    
    def get_available_voices(self) -> list:
        """Get list of available TTS voices"""
        try:
            voices = self.tts_engine.getProperty('voices')
            return [(voice.id, voice.name) for voice in voices]
        except Exception as e:
            logger.error(f"Error getting voices: {e}")
            return []
    
    def set_voice(self, voice_id: str):
        """Set TTS voice by ID"""
        try:
            self.tts_engine.setProperty('voice', voice_id)
            logger.info(f"Voice changed to: {voice_id}")
        except Exception as e:
            logger.error(f"Error setting voice: {e}")
    
    def set_speech_rate(self, rate: int):
        """Set TTS speech rate"""
        try:
            self.tts_engine.setProperty('rate', rate)
            logger.info(f"Speech rate set to: {rate}")
        except Exception as e:
            logger.error(f"Error setting speech rate: {e}")
    
    def test_audio_devices(self):
        """Test and list available audio devices"""
        try:
            logger.info("Available audio devices:")
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                logger.info(f"  {i}: {device['name']} - {device['max_input_channels']} in, {device['max_output_channels']} out")
                
            # Test microphone
            logger.info("Testing microphone...")
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=2, phrase_time_limit=3)
                logger.info("Microphone test successful")
                
            # Test TTS
            logger.info("Testing text-to-speech...")
            asyncio.run(self.text_to_speech("Audio test successful"))
            
        except Exception as e:
            logger.error(f"Audio device test failed: {e}")
    
    def stop(self):
        """Stop all voice I/O operations"""
        self.stop_background_listening()
        
        if self.tts_engine:
            try:
                self.tts_engine.stop()
            except:
                pass
                
        logger.info("Voice I/O stopped")


# Example usage and testing
if __name__ == "__main__":
    import sys
    sys.path.append('..')
    from config.settings import Config
    
    # Test the VoiceIO module
    config = Config()
    voice_io = VoiceIO(config)
    
    # Test audio devices
    voice_io.test_audio_devices()
    
    # Test wake word detection
    print("Say 'Hey Assistant' to test wake word detection...")
    audio_data = asyncio.run(voice_io.listen_for_wake_word(timeout=10))
    
    if audio_data:
        text = asyncio.run(voice_io.speech_to_text(audio_data))
        print(f"You said: {text}")
        
        asyncio.run(voice_io.text_to_speech(f"I heard you say: {text}"))
    else:
        print("No wake word detected")
