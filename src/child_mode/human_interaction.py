import vosk
import json
import queue
import sounddevice as sd
import numpy as np
import requests
import tempfile
import os
import time
import subprocess
import sys
import threading


def list_audio_devices():
    
    print("\n Scanning for audio devices...")
    try:
        devices = sd.query_devices()
        input_devices = []
        
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                input_devices.append(i)
                print(f"   Device {i}: {device['name']} (Input channels: {device['max_input_channels']})")
        
        if not input_devices:
            print("No input devices found!")
            return None
        
        return input_devices
    except Exception as e:
        print(f"Error scanning devices: {e}")
        return None

def get_default_mic():
    """Get the default microphone device index"""
    try:
        default_device = sd.default.device
        if isinstance(default_device, tuple):
            return default_device[0]
        return default_device
    except:
        return None

NVIDIA_API_KEY = ""  # Replace with your actual NVIDIA API key

def ask_nvidia_nim(question):
    """Query NVIDIA NIM with Llama 3.1 8B model"""
    
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "meta/llama-3.1-8b-instruct",
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


class SimpleTTS:
    def __init__(self):
       
        self.use_espeak = False
        
        try:
            subprocess.run(['which', 'espeak'], capture_output=True, check=True)
            self.use_espeak = True
            print("eSpeak TTS initialized")
            return
        except:
            pass
        
        print("No TTS available - text output only")
    
    def speak(self, text):
        
        if not text or text.strip() == "":
            return
        
        print(f"\nVitaCustos: {text}\n")
        
        if self.use_espeak:
            try:
                subprocess.run(['espeak', '-s', '130', text], 
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
            except:
                pass


class VoskRecognizer:
    def __init__(self, model_path="vosk-model-small-en-us-0.15"):
       
        self.model_path = model_path
        self.model = None
        self.recognizer = None
        self.audio_queue = queue.Queue()
        self.sample_rate = 16000
        self.device_index = None
        
        self.find_microphone()
        
        self.load_model()
    
    def find_microphone(self):
        """Find a working microphone"""
        print("\nLooking for microphone...")
        
        try:
            devices = sd.query_devices()
            input_devices = []
            
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    input_devices.append(i)
                    print(f"   Found mic {i}: {device['name']}")
            
            if not input_devices:
                print("   No microphone found!")
                print("   Please check:")
                print("   1. USB mic is connected")
                print("   2. Run: arecord -l")
                print("   3. Try with: sudo python3 interaction.py")
                return False
            
            usb_mic = None
            for i in input_devices:
                device_name = devices[i]['name'].lower()
                if 'usb' in device_name or 'mic' in device_name:
                    usb_mic = i
                    break
            
            if usb_mic is not None:
                self.device_index = usb_mic
                print(f"Selected USB microphone: {devices[usb_mic]['name']}")
            else:
                self.device_index = input_devices[0]
                print(f"Selected default microphone: {devices[self.device_index]['name']}")
            
            print("Testing microphone (listening for 1 second)...")
            try:
                test_recording = sd.rec(int(1 * self.sample_rate), 
                                      samplerate=self.sample_rate, 
                                      channels=1, 
                                      device=self.device_index,
                                      dtype='int16')
                sd.wait()
                print("Microphone is working!")
            except Exception as e:
                print(f" Microphone test failed: {e}")
                print("   Try running with: sudo python3 interaction.py")
            
            return True
            
        except Exception as e:
            print(f" Error finding microphone: {e}")
            return False
    
    def load_model(self):
        
        if not os.path.exists(self.model_path):
            print(f"\nDownloading Vosk model (this will take a few minutes)...")
            self.download_model()
        
        try:
            self.model = vosk.Model(self.model_path)
            self.recognizer = vosk.KaldiRecognizer(self.model, self.sample_rate)
            print("Speech recognition model loaded")
            return True
        except Exception as e:
            print(f"Failed to load Vosk model: {e}")
            return False
    
    def download_model(self):
        
        import urllib.request
        import zipfile
        
        model_url = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
        zip_path = "vosk-model.zip"
        
        print(f"Downloading from {model_url}")
        urllib.request.urlretrieve(model_url, zip_path)
        
        print("Extracting model...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        
        os.remove(zip_path)
        print(" Model downloaded and extracted")
    
    def audio_callback(self, indata, frames, time, status):
      
        if status:
            print(f"Audio status: {status}")
        self.audio_queue.put(bytes(indata))
    
    def listen(self):
        
        if not self.model:
            print("Speech model not loaded")
            return ""
        
        if self.device_index is None:
            print("No microphone found")
            return ""
        
        print("\nListening... (Speak now)")
        
        try:
            with sd.RawInputStream(samplerate=self.sample_rate, 
                                  blocksize=8000,
                                  device=self.device_index, 
                                  dtype='int16',
                                  channels=1, 
                                  callback=self.audio_callback):
                
                print("I'm listening...")
                start_time = time.time()
                speech_started = False
                last_speech_time = None
                accumulated_text = []
                
                while time.time() - start_time < 10:
                    try:
                        data = self.audio_queue.get(timeout=0.3)
                        
                        if self.recognizer.AcceptWaveform(data):
                            result = json.loads(self.recognizer.Result())
                            text = result.get('text', '')
                            if text:
                                accumulated_text.append(text)
                                print(f"\r Recognized: {text}")
                                speech_started = False
                                last_speech_time = time.time()
                        else:
                            partial = json.loads(self.recognizer.PartialResult())
                            partial_text = partial.get('partial', '')
                            if partial_text:
                                if not speech_started:
                                    print(f"\r Hearing: {partial_text}", end='', flush=True)
                                    speech_started = True
                                else:
                                    print(f"\r Hearing: {partial_text}", end='', flush=True)
                                last_speech_time = time.time()
                            elif speech_started and last_speech_time and time.time() - last_speech_time > 1.5:
                                print("\n Speech ended")
                                break
                    
                    except queue.Empty:
                        if speech_started and last_speech_time and time.time() - last_speech_time > 2.0:
                            print("\n Pause detected")
                            break
                        continue
                
                result = json.loads(self.recognizer.FinalResult())
                text = result.get('text', '')
                if text:
                    accumulated_text.append(text)
                
                if accumulated_text:
                    full_text = ' '.join(accumulated_text)
                    print(f"\n You said: {full_text}")
                    return full_text.lower()
                
        except sd.PortAudioError as e:
            if "Invalid device" in str(e):
                print(f" Microphone device {self.device_index} is not available")
                print("   Please check your USB microphone connection")
                print("   Try running: arecord -l")
            else:
                print(f" Audio error: {e}")
            return ""
        except Exception as e:
            print(f" Microphone error: {e}")
            return ""
        
        print("\n No speech detected")
        return ""


class VitaCustosAssistant:
    def __init__(self):
        """Initialize the complete assistant"""
        self.is_running = True
        self.stt = None
        self.tts = None
        
        print("\n" + "="*60)
        print(" VitaCustos AI Assistant - Offline Mode")
        print("="*60)
        
        print("\n Initializing Text-to-Speech...")
        self.tts = SimpleTTS()
        time.sleep(1)
        
        print("\n Initializing Speech Recognition...")
        self.stt = VoskRecognizer()
        
        if not self.stt.model or self.stt.device_index is None:
            print("\n Failed to initialize speech recognition!")
            print("\nTroubleshooting steps:")
            print("1. Check USB microphone: arecord -l")
            print("2. Test microphone: arecord -d 5 test.wav")
            print("3. Run with sudo: sudo python3 interaction.py")
            print("4. Reboot Raspberry Pi: sudo reboot")
            sys.exit(1)
        
        time.sleep(1)
        
        print("\n System ready!")
        self.tts.speak("Hello! I am VitaCustos AI. How can I help you today?")
    
    def process_command(self, command):
        """Process voice commands"""
        exit_commands = ["stop", "shutdown", "exit", "quit", "goodbye", "bye bye"]
        if any(cmd in command for cmd in exit_commands):
            self.tts.speak("Goodbye! Take care.")
            self.is_running = False
            return
        
        if "health" in command or "medication" in command:
            self.tts.speak("Let me check health information for you.")
        
        print(" Thinking...")
        self.tts.speak("Let me think about that.")
        
        answer = ask_nvidia_nim(command)
        
        self.tts.speak(answer)
    
    def run(self):
        """Main loop with proper pacing"""
        print("\n" + "="*60)
        print(" VitaCustos AI is now running")
        print("="*60)
        print("   Commands you can say:")
        print("   • 'What is my medication schedule?'")
        print("   • 'How are you feeling today?'")
        print("   • 'Remind me to take medicine'")
        print("   • 'What's the weather like?'")
        print("   • 'Help me with...'")
        print("   • 'Stop', 'exit', or 'goodbye' to quit")
        print("="*60 + "\n")
        
        self.tts.speak("I'm ready. Please speak clearly when you see the listening prompt.")
        
        while self.is_running:
            try:
                command = self.stt.listen()
                
                if command:
                    self.process_command(command)
                    
                    print("\n🎧 Ready for next command...\n")
                    time.sleep(1)
                else:
                    print("\n Waiting for command...\n")
                    time.sleep(2)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f" Error: {e}")
                time.sleep(2)
        
        print("\n Goodbye!")


def install_dependencies():
    """Install required dependencies"""
    print("\n Installing dependencies...")
    
    packages = [
        'python3-pip',
        'python3-numpy',
        'portaudio19-dev',
        'python3-pyaudio',
        'espeak'
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        subprocess.run(['sudo', 'apt-get', 'install', '-y', package], 
                      capture_output=True)
    
    print("\nInstalling Python packages...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', 
                   'vosk', 'sounddevice', 'numpy', 'requests'],
                  capture_output=True)
    
    print("\nInstallation complete!")
    print("\nNow run: sudo python3 interaction.py\n")

if __name__ == "__main__":
    if os.geteuid() != 0:
        print("\n  For best results, please run with sudo:")
        print("   sudo python3 interaction.py\n")
        response = input("Continue without sudo? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    try:
        import vosk
        import sounddevice
    except ImportError as e:
        print(f"\n Missing dependency: {e}")
        response = input("Would you like to install dependencies now? (y/n): ")
        if response.lower() == 'y':
            install_dependencies()
            exit(0)
        else:
            print("Please install dependencies and try again.")
            exit(1)
    
    if NVIDIA_API_KEY == "":
        print("\n  WARNING: Please set your NVIDIA API key!")
        print("   1. Go to https://build.nvidia.com")
        print("   2. Sign up for a free account")
        print("   3. Get your API key")
        print("   4. Update NVIDIA_API_KEY in the script\n")
        response = input("Continue without API key? (y/n): ")
        if response.lower() != 'y':
            exit(0)
    
    try:
        assistant = VitaCustosAssistant()
        assistant.run()
    except KeyboardInterrupt:
        print("\n\n Goodbye!")
    except Exception as e:
        print(f"\n Fatal error: {e}")
        print("\nPlease run with: sudo python3 interaction.py")
        time.sleep(5)