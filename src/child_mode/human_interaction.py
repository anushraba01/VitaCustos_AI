import speech_recognition as sr
import pyttsx3
import requests
import json

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
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        return f"API Error: {str(e)}"
    except KeyError as e:
        return f"Response parsing error. Please try again."

# =========================
# TEXT TO SPEECH
# =========================

engine = pyttsx3.init()

engine.setProperty('rate', 170)
engine.setProperty('volume', 1.0)

voices = engine.getProperty('voices')
# Select voice (0 usually male, 1 female)
engine.setProperty('voice', voices[0].id)

# =========================
# SPEAK FUNCTION
# =========================

def speak(text):
    print(f"\nAI: {text}\n")
    engine.say(text)
    engine.runAndWait()

# =========================
# LISTEN FUNCTION
# =========================

recognizer = sr.Recognizer()

def listen():
    with sr.Microphone() as source:
        print("\nListening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    
    try:
        text = recognizer.recognize_google(audio)
        print(f"\nUser: {text}\n")
        return text.lower()
    
    except sr.UnknownValueError:
        print("Could not understand. Please try again.")
        return ""
    
    except sr.RequestError:
        speak("Internet connection error")
        return ""

# =========================
# MAIN LOOP
# =========================

def main():
    speak("VitaCustos AI is now online")
    
    while True:
        user_input = listen()
        
        if user_input == "":
            continue
        
        # EXIT COMMANDS
        if any(cmd in user_input for cmd in ["stop", "shutdown", "exit", "quit"]):
            speak("Shutting down VitaCustos AI")
            break
        
        # HEALTH MONITORING SHORTCUTS (optional)
        if "health" in user_input or "medication" in user_input:
            speak("I'm checking your health information. One moment please.")
        
        # GET RESPONSE FROM NVIDIA NIM
        answer = ask_nvidia_nim(user_input)
        
        # SPEAK RESPONSE
        speak(answer)

# =========================
# RUN THE ASSISTANT
# =========================

if __name__ == "__main__":
    # Test API key first
    if NVIDIA_API_KEY == "YOUR_NVIDIA_API_KEY":
        print("\n⚠️  WARNING: Please replace YOUR_NVIDIA_API_KEY with your actual NVIDIA API key")
        print("Get a free API key from: https://build.nvidia.com\n")
    
    try:
        main()
    except KeyboardInterrupt:
        speak("Goodbye!")
    except Exception as e:
        print(f"Error: {e}")
        speak("An error occurred. Restarting assistant.")