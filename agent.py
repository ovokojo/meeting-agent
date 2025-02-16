import os
import queue
import wave
import io
import base64
import time
import re

import pyaudio
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
from pydub import AudioSegment

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

# Store conversation context
conversation_history = []

def create_wake_phrase_regex(wake_phrase: str) -> re.Pattern:
    """Creates a regex pattern for the given wake phrase with optional punctuation/whitespace"""
    # Escape any special regex characters in the wake phrase
    escaped_phrase = re.escape(wake_phrase)
    # Split the escaped phrase into words and join with flexible whitespace/punctuation pattern
    words = escaped_phrase.split(r'\ ')
    pattern = r'\b' + r'\b[,\s]*\b'.join(words) + r'\b'
    return re.compile(pattern, re.IGNORECASE)

def extract_question(transcribed_text: str, wake_phrase_regex: re.Pattern) -> str:
    """Extracts the question part after the wake phrase if it exists."""
    match = wake_phrase_regex.search(transcribed_text)
    if match:
        # Get everything after the matched wake phrase and strip extra punctuation/spaces
        question = transcribed_text[match.end():].strip(" ,.:;!?")
        return question
    return ""

def is_silence(audio_chunk):
    """Check if the audio chunk is silence"""
    try:
        data = np.frombuffer(audio_chunk, dtype=np.int16)
        amplitude = np.max(np.abs(data))
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
        # Save the audio data to a temporary WAV file
        with wave.open(TEMP_WAV_FILE, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # 2 bytes for int16
            wf.setframerate(RATE)
            wf.writeframes(audio_data)
        
        print("Sending audio to Whisper...")
        
        # Open and send the WAV file
        with open(TEMP_WAV_FILE, 'rb') as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        
        # Clean up temporary file
        try:
            os.remove(TEMP_WAV_FILE)
        except Exception as e:
            print(f"Error cleaning up temporary file: {e}")
        
        return response
        
    except Exception as e:
        print(f"Error processing audio: {e}")
        return ""

def generate_response(prompt_text: str) -> tuple:
    """Generate text and audio response using GPT-4o"""
    try:
        # First get the text response
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful meeting assistant."},
                {"role": "user", "content": prompt_text}
            ]
        )
        
        text_response = completion.choices[0].message.content
        
        # Generate speech and save to a temporary MP3 file
        temp_mp3 = "temp_response.mp3"
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text_response
        )
        
        # Save the MP3 response
        with open(temp_mp3, "wb") as file:
            response_bytes = b''
            for chunk in response.iter_bytes():
                response_bytes += chunk
            file.write(response_bytes)
            file.flush()
            os.fsync(file.fileno())
        
        # Small delay to ensure file is written
        time.sleep(0.1)
        
        # Convert MP3 to WAV using pydub
        try:
            audio = AudioSegment.from_mp3(temp_mp3)
            audio.export(RESPONSE_WAV_FILE, format="wav")
            
            # Clean up temporary MP3 file
            os.remove(temp_mp3)
            
            return text_response, RESPONSE_WAV_FILE
            
        except Exception as e:
            print(f"Error converting audio format: {e}")
            if os.path.exists(temp_mp3):
                os.remove(temp_mp3)
            return text_response, None
        
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Sorry, I had an error generating a response.", None

def play_audio(wav_file_path: str):
    """Play a WAV file"""
    try:
        if not os.path.exists(wav_file_path):
            print(f"Audio file not found: {wav_file_path}")
            return
            
        # Ensure the file is completely written
        time.sleep(0.1)
        
        try:
            wf = wave.open(wav_file_path, 'rb')
        except Exception as e:
            print(f"Error opening WAV file: {e}")
            return
            
        p = pyaudio.PyAudio()
        
        try:
            # Open stream
            stream = p.open(
                format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True
            )
            
            # Read data in chunks
            data = wf.readframes(CHUNK)
            
            print("Playing audio response...")
            while len(data) > 0:
                stream.write(data)
                data = wf.readframes(CHUNK)
                
        finally:
            # Cleanup
            stream.stop_stream()
            stream.close()
            p.terminate()
            wf.close()
        
    except Exception as e:
        print(f"Error playing audio: {e}")
        # Print more detailed error information
        import traceback
        traceback.print_exc()

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
    
    # Let user set wake phrase
    wake_phrase = input("\nEnter the wake phrase you want to use (press Enter for default 'hey Cora'): ").strip()
    wake_phrase = wake_phrase if wake_phrase else "hey Cora"
    wake_phrase_regex = create_wake_phrase_regex(wake_phrase)
    
    print(f"\nUsing device {device_index}")
    print(f"Assistant is ready! Say '{wake_phrase}' followed by your question.")
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
                    question = extract_question(text, wake_phrase_regex)
                    if question:
                        print(f"Question detected: {question}")
                        response_text, audio_file = generate_response(question)
                        print(f"Assistant response: {response_text}")
                        
                        if audio_file:
                            print(f"Playing audio response from {audio_file}")
                            play_audio(audio_file)
                            try:
                                os.remove(audio_file)
                            except Exception as e:
                                print(f"Error cleaning up audio file: {e}")
                    
    except KeyboardInterrupt:
        print("\nStopping the assistant...")
    finally:
        p.terminate()

if __name__ == "__main__":
    main()