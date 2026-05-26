import speech_recognition as sr
import edge_tts
import asyncio
import requests
import json
import tempfile
import pygame
import os
import time

# =========================
# NVIDIA NIM API (FREE)
# =========================
# Sign up at https://build.nvidia.com to get your API key
NVIDIA_API_KEY = ""  # Get free key from build.nvidia.com

def ask_nvidia_nim(question):
    """Query NVIDIA NIM with Llama 3.1 8B model"""
    
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "meta/llama-3.1-8b-instruct",  # Free tier model
        "messages": [
            {
                "role": "system",
                "content": "You are VitaCustos AI, an advanced intelligent caregiving assistant. Provide helpful, concise responses focused on elderly care, health monitoring, medication reminders, and general assistance."
            },
            {
                "role": "user",
                "content": question
            }
        ],
        "temperature": 0.7,
        "max_tokens": 300,
        "top_p": 0.95
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        return f"API Error: {str(e)}"
    except KeyError as e:
        return f"Response parsing error. Please try again."

# =========================
# EDGE TTS - IMPROVED VERSION
# =========================

class EdgeTTSWrapper:
    def __init__(self):
        """Initialize Edge TTS with better voice options"""
        self.voice_options = {
            # English US voices (most natural)
            "male": "en-US-ChristopherNeural",
            "female": "en-US-JennyNeural",
            # English UK voices
            "male_uk": "en-GB-RyanNeural",
            "female_uk": "en-GB-SoniaNeural",
            # English India voices (good for English learners)
            "male_india": "en-IN-PrabhatNeural",
            "female_india": "en-IN-NeerjaNeural",
            # Other natural voices
            "friendly": "en-US-AriaNeural",
            "news": "en-US-GuyNeural"
        }
        
        # Default voice (US female)
        self.current_voice = self.voice_options["female"]
        
        # Initialize pygame mixer for audio playback
        pygame.mixer.init()
        
        # Create temp directory for audio files
        self.temp_dir = tempfile.gettempdir()
        
        print(f"✅ Edge TTS initialized with voice: {self.current_voice}")
    
    def set_voice(self, voice_type="female"):
        """Change voice: male, female, male_uk, female_uk, male_india, female_india, friendly, news"""
        if voice_type in self.voice_options:
            self.current_voice = self.voice_options[voice_type]
            print(f"🎤 Voice changed to: {self.current_voice}")
        else:
            print(f"Voice {voice_type} not found. Using current voice.")
    
    async def speak_async(self, text):
        """Asynchronously convert text to speech and play it"""
        if not text or text.strip() == "":
            return
        
        try:
            # Create temporary file for audio
            temp_file = os.path.join(self.temp_dir, f"tts_{int(time.time())}.mp3")
            
            # Generate speech using Edge TTS
            communicate = edge_tts.Communicate(text, self.current_voice)
            await communicate.save(temp_file)
            
            # Play the audio using pygame
            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                await asyncio.sleep(0.1)
            
            # Clean up temp file
            pygame.mixer.music.unload()
            time.sleep(0.1)  # Small delay to ensure file release
            try:
                os.remove(temp_file)
            except:
                pass  # File might be in use, ignore
            
        except Exception as e:
            print(f"🔊 TTS Error: {e}")
            print(f"💬 Text: {text}")
    
    def speak(self, text):
        """Synchronous wrapper for speak_async"""
        if not text or text.strip() == "":
            return
        
        try:
            # Run async function synchronously
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.speak_async(text))
            loop.close()
        except Exception as e:
            print(f"🔊 TTS Error: {e}")
            print(f"💬 Text: {text}")

# =========================
# LISTEN FUNCTION
# =========================

def init_recognizer():
    """Initialize speech recognizer with better microphone handling"""
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.8
    return recognizer

def listen(recognizer):
    """Listen and recognize speech with better error handling"""
    try:
        with sr.Microphone() as source:
            print("\n🎤 Listening...")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            print("👂 Ready...")
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
        
        print("🔄 Processing...")
        text = recognizer.recognize_google(audio)
        print(f"\n👤 You: {text}\n")
        return text.lower()
    
    except sr.WaitTimeoutError:
        print("⏰ No speech detected")
        return ""
    
    except sr.UnknownValueError:
        print("❓ Could not understand")
        return ""
    
    except sr.RequestError as e:
        print(f"🌐 Internet error: {e}")
        return ""
    
    except Exception as e:
        print(f"🎙️ Microphone error: {e}")
        return ""

# =========================
# MAIN LOOP
# =========================

def main():
    # Initialize components
    print("🔧 Initializing VitaCustos AI Assistant with Edge TTS...")
    
    # Initialize TTS
    tts = EdgeTTSWrapper()
    
    # Optionally change voice (uncomment to use different voice)
    # tts.set_voice("male")  # Try male voice
    # tts.set_voice("female_uk")  # Try UK female voice
    # tts.set_voice("friendly")  # Try friendly US voice
    
    # Initialize speech recognizer
    recognizer = init_recognizer()
    print("✅ Speech recognition ready")
    
    # Test TTS
    print("🔊 Testing voice...")
    tts.speak("VitaCustos AI is now online. How can I help you today?")
    
    while True:
        user_input = listen(recognizer)
        
        if user_input == "":
            continue
        
        # EXIT COMMANDS
        exit_commands = ["stop", "shutdown", "exit", "quit", "goodbye", "bye bye"]
        if any(cmd in user_input for cmd in exit_commands):
            tts.speak("Shutting down VitaCustos AI. Take care! Goodbye.")
            print("\n👋 Shutting down...")
            break
        
        # VOICE COMMAND
        if "change voice" in user_input or "switch voice" in user_input:
            if "male" in user_input:
                tts.set_voice("male")
                tts.speak("Voice changed to male")
            elif "female" in user_input:
                tts.set_voice("female")
                tts.speak("Voice changed to female")
            elif "uk" in user_input or "british" in user_input:
                tts.set_voice("female_uk")
                tts.speak("Voice changed to British English")
            else:
                tts.speak("You can say change voice to male or female")
            continue
        
        # HEALTH MONITORING SHORTCUTS
        if "health" in user_input or "medication" in user_input:
            tts.speak("Checking health information. One moment please.")
        
        # Show thinking indicator
        print("🤔 Thinking...")
        
        # GET RESPONSE FROM NVIDIA NIM
        answer = ask_nvidia_nim(user_input)
        
        # SPEAK RESPONSE
        tts.speak(answer)
        
        # Small pause between interactions
        time.sleep(0.5)

# =========================
# RUN THE ASSISTANT
# =========================

if __name__ == "__main__":
    # Install required packages if not present
    print("📦 Checking dependencies...")
    try:
        import edge_tts
        import pygame
    except ImportError as e:
        print("❌ Missing dependencies. Please install:")
        print("   pip install edge-tts pygame SpeechRecognition requests")
        exit(1)
    
    # API Key check
    if NVIDIA_API_KEY == "":
        print("\n⚠️  WARNING: Please set your NVIDIA API key!")
        print("   1. Go to https://build.nvidia.com")
        print("   2. Sign up for a free account")
        print("   3. Get your API key")
        print("   4. Replace NVIDIA_API_KEY = \"\" with your key\n")
        response = input("Do you want to continue without API key? (y/n): ")
        if response.lower() != 'y':
            exit(0)
    
    # Run the assistant
    try:
        print("\n🎯 VitaCustos AI Assistant Starting...")
        print("   Features:")
        print("   🎤 Voice recognition with Google Speech")
        print("   🔊 Natural voice using Microsoft Edge TTS")
        print("   🧠 AI responses from NVIDIA NIM")
        print("   🎙️ Say 'stop', 'exit', or 'quit' to end")
        print("   🔊 Say 'change voice to male/female' to change voice\n")
        
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye from VitaCustos AI!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please restart the assistant")
        time.sleep(2)