import os
import queue
import wave
import io
import base64
import time

import pyaudio
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

########################
# Load Environment Vars
########################
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

########################
# Audio Config
########################
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
SILENCE_THRESHOLD = 500
SILENCE_DURATION = 1.0
TEMP_WAV_FILE = "temp_input.wav"
RESPONSE_WAV_FILE = "response.wav"

# The phrase that triggers the bot to respond
WAKE_PHRASE = "hey bot"

# Store conversation context
conversation_history = []

def is_silence(audio_chunk):
    """Check if the audio chunk is silence"""
    try:
        data = np.frombuffer(audio_chunk, dtype=np.int16)
        amplitude = np.max(np.abs(data))
        print(f"Current amplitude: {amplitude}")  # Debug logging
        return amplitude < SILENCE_THRESHOLD
    except Exception as e:
        print(f"Error checking silence: {e}")
        return True

def record_audio(device_index=None):
    """Record audio until silence is detected"""
    p = pyaudio.PyAudio()
    
    try:
        # Get device info for debugging
        if device_index is not None:
            dev_info = p.get_device_info_by_index(device_index)
            print(f"\nUsing device: {dev_info['name']}")
            print(f"Device details: {dev_info}")
        
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
            input_device_index=device_index
        )
        
        # Test audio levels
        print("\nTesting audio levels - please speak...")
        for _ in range(10):
            data = stream.read(CHUNK, exception_on_overflow=False)
            amplitude = np.max(np.abs(np.frombuffer(data, dtype=np.int16)))
            print(f"Test amplitude: {amplitude}")
        
        print("\nListening... (speak your message)")
        frames = []
        silence_start = None
        is_recording = False
        min_frames = int(RATE * 0.5 / CHUNK)  # Minimum 0.5 seconds of audio
        
        while True:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                
                # Always collect the data
                frames.append(data)
                
                # Check if this is silence
                if is_silence(data):
                    if is_recording and len(frames) > min_frames:
                        if silence_start is None:
                            silence_start = time.time()
                        elif time.time() - silence_start > SILENCE_DURATION:
                            break  # Stop recording after silence duration
                else:
                    is_recording = True
                    silence_start = None
                    
            except OSError as e:
                print(f"Warning: {e}")
                continue
        
        print("Finished recording")
        return b''.join(frames) if frames else b''
        
    except Exception as e:
        print(f"Error in recording: {e}")
        return b''
        
    finally:
        try:
            stream.stop_stream()
            stream.close()
        except Exception:
            pass
        p.terminate()

def process_audio(audio_data):
    """Process audio data using GPT-4o"""
    if not audio_data:
        return ""
        
    try:
        # No need to convert since we're already in int16 format
        audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
        
        # Create WAV file with proper headers
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # 2 bytes for int16
            wf.setframerate(RATE)
            wf.writeframes(audio_int16.tobytes())
        
        # Convert to base64
        wav_buffer.seek(0)
        base64_audio = base64.b64encode(wav_buffer.read()).decode('utf-8')
        
        print("Sending audio to GPT-4o...")
        
        # Build conversation messages with proper structure
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Transcribe the audio input."},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": base64_audio,
                            "format": "wav"
                        }
                    }
                ]
            }
        ]
        
        response = client.chat.completions.create(
            model="gpt-4o-audio-preview",
            modalities=["text"],
            messages=messages
        )
        
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content
        return ""
        
    except Exception as e:
        print(f"Error processing audio: {e}")
        return ""

def generate_response(prompt_text: str) -> tuple:
    """Generate text and audio response using GPT-4o"""
    try:
        # Build conversation messages with history
        messages = [{"role": "system", "content": "You are a helpful meeting assistant."}]
        
        # Add conversation history with proper audio references
        for msg in conversation_history:
            if 'audio' in msg:
                messages.append({
                    "role": msg["role"],
                    "content": None,  # Required for audio responses
                    "audio": {"id": msg["audio"]["id"]}
                })
            else:
                messages.append(msg)
        
        # Add current prompt
        messages.append({"role": "user", "content": prompt_text})
        
        response = client.chat.completions.create(
            model="gpt-4o-audio-preview",
            modalities=["text", "audio"],
            audio={"voice": "alloy", "format": "wav"},
            messages=messages
        )
        
        if not response.choices:
            return "Sorry, I couldn't generate a response.", None
            
        choice = response.choices[0]
        text_response = choice.message.content
        
        # Save audio response if available
        if choice.message.audio:
            wav_bytes = base64.b64decode(choice.message.audio.data)
            with open(RESPONSE_WAV_FILE, "wb") as f:
                f.write(wav_bytes)
                
            # Update conversation history with both text and audio reference
            conversation_history.append({"role": "user", "content": prompt_text})
            conversation_history.append({
                "role": "assistant",
                "content": text_response,  # Store text for context
                "audio": {"id": choice.message.audio.id}
            })
            
            return text_response, RESPONSE_WAV_FILE
            
        return text_response, None
        
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Sorry, I had an error generating a response.", None

def play_audio(wav_file_path: str):
    """Play a WAV file"""
    try:
        wf = wave.open(wav_file_path, 'rb')
        p = pyaudio.PyAudio()
        
        stream = p.open(
            format=p.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True
        )
        
        data = wf.readframes(CHUNK)
        while data:
            stream.write(data)
            data = wf.readframes(CHUNK)
            
        stream.stop_stream()
        stream.close()
        p.terminate()
        
    except Exception as e:
        print(f"Error playing audio: {e}")

def main():
    p = pyaudio.PyAudio()
    
    # Show available input devices
    print("\nAvailable Input Devices:")
    default_input = None
    for i in range(p.get_device_count()):
        dev_info = p.get_device_info_by_index(i)
        if dev_info['maxInputChannels'] > 0:
            print(f"Device {i}: {dev_info['name']}")
            if dev_info.get('isDefaultInputDevice'):
                default_input = i
                print(f"  (Default Input Device)")
    
    # Let user select input device
    device_index = input("\nEnter the device number to use (press Enter for default): ").strip()
    device_index = int(device_index) if device_index else default_input
    
    print(f"\nUsing device {device_index}")
    print("Bot is ready! Say 'hey bot' followed by your question.")
    print("(Recording will stop after 1 second of silence)")
    
    try:
        while True:
            # Record audio until silence
            audio_data = record_audio(device_index)
            if len(audio_data) > 0:
                # Process the audio
                text = process_audio(audio_data)
                print(f"Transcribed: {text}")
                
                if text:
                    lower_text = text.lower()
                    if WAKE_PHRASE in lower_text:
                        # Extract the question
                        question = lower_text.split(WAKE_PHRASE, 1)[1].strip()
                        if question:
                            print(f"Question detected: {question}")
                            response_text, audio_file = generate_response(question)
                            print(f"Bot response: {response_text}")
                            
                            if audio_file:
                                play_audio(audio_file)
                        
    except KeyboardInterrupt:
        print("\nStopping the bot...")
    finally:
        p.terminate()

if __name__ == "__main__":
    main()