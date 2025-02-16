import os
import queue
import threading
import time
import wave
import io

import pyaudio
import pyttsx3
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

########################
# Load Environment Vars
########################
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

########################
# Audio / Whisper Config
########################
CHUNK = 8192
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
TEMP_WAV_FILE = "temp_input.wav"
RESPONSE_WAV_FILE = "temp_response.wav"
RECORD_SECONDS = 2.0  # Increased recording window
THRESHOLD = 1000  # Adjust this value based on your microphone sensitivity

# The phrase that triggers the bot to respond
WAKE_PHRASE = "hey bot"

########################
# Initialize TTS Engine
########################
tts_engine = pyttsx3.init()
# You can tweak voice properties if you like:
# voices = tts_engine.getProperty('voices')
# tts_engine.setProperty('voice', voices[0].id)  # pick a different index for a different voice
tts_engine.setProperty('rate', 180)  # speed
tts_engine.setProperty('volume', 1.0)  # max volume

class AudioStreamer:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.stop_thread = False
        self.is_speaking = False
        # Assuming RATE = 16000 and CHUNK = 8192
        # Each chunk is about 0.5 seconds (8192/16000)
        # So for 3 seconds we need about 6 chunks
        self.silence_threshold = 6  # Changed from 15 to get ~3 seconds
        self.silence_counter = 0

    def callback(self, in_data, frame_count, time_info, status):
        if is_speech(in_data):
            self.is_speaking = True
            self.silence_counter = 0
            print(f"Silence counter reset, is_speaking: {self.is_speaking}")
            self.audio_queue.put(in_data)
        elif self.is_speaking:
            self.silence_counter += 1
            print(f"Silence counter: {self.silence_counter}/{self.silence_threshold}")
            if self.silence_counter > self.silence_threshold:
                self.is_speaking = False
                self.silence_counter = 0
                print("Silence threshold reached, processing audio...")
                self.audio_queue.put(None)
        return (in_data, pyaudio.paContinue)

def is_speech(audio_chunk):
    """
    Check if the audio chunk contains speech based on amplitude
    """
    data = np.frombuffer(audio_chunk, dtype=np.int16)
    amplitude = np.max(np.abs(data))
    # Add debug logging
    if amplitude > THRESHOLD:
        print(f"Speech detected! Amplitude: {amplitude}")
    return amplitude > THRESHOLD

def process_audio_stream(audio_data):
    """Process accumulated audio data with Whisper API"""
    try:
        print(f"Processing audio chunk of size: {len(audio_data)} bytes")
        # Create WAV file with proper headers
        wav_buffer = io.BytesIO()
        wf = wave.open(wav_buffer, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(audio_data)
        wf.close()
        wav_buffer.seek(0)
        
        print("Sending audio to Whisper API...")
        try:
            # Create a named temporary file-like object that Whisper API can process
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=("audio.wav", wav_buffer, "audio/wav"),  # Properly specify the file format
                language="en"
            )
            print("Successfully received response from Whisper API")
            return response.text
        except Exception as api_error:
            print(f"Whisper API error: {str(api_error)}")
            return ""
            
    except Exception as e:
        print(f"Error in audio processing: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(traceback.format_exc())
        return ""

def generate_bot_response(prompt_text: str) -> str:
    """
    Send the text after the wake phrase to GPT for a response.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful meeting assistant."},
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.7
        )
        answer = response.choices[0].message.content.strip()
        return answer
    except Exception as e:
        print(f"Error in GPT response: {e}")
        return "Sorry, I had an error generating a response."

def synthesize_speech(text: str, output_wav_path: str):
    """
    Use pyttsx3 (local TTS) to synthesize `text` to a WAV file at `output_wav_path`.
    """
    tts_engine.save_to_file(text, output_wav_path)
    tts_engine.runAndWait()

def play_audio(wav_file_path: str):
    """
    Play a WAV file to the default output device. If you set your default
    device (or specifically this stream) to the loopback device, it will
    be fed into the Google Meet call.
    """
    wf = wave.open(wav_file_path, 'rb')
    p = pyaudio.PyAudio()

    # open an output stream
    output_stream = p.open(
        format=p.get_format_from_width(wf.getsampwidth()),
        channels=wf.getnchannels(),
        rate=wf.getframerate(),
        output=True
    )

    data = wf.readframes(CHUNK)
    while data:
        output_stream.write(data)
        data = wf.readframes(CHUNK)

    output_stream.stop_stream()
    output_stream.close()
    wf.close()
    p.terminate()

def speak_response(text: str):
    """Use text-to-speech to speak the response"""
    print(f"Speaking response: {text}")
    try:
        # Initialize a new engine for each response to avoid blocking
        engine = pyttsx3.init()
        engine.setProperty('rate', 180)
        engine.setProperty('volume', 1.0)
        
        # Get available voices and set to a good default
        voices = engine.getProperty('voices')
        if voices:  # If voices are available, use the first one
            engine.setProperty('voice', voices[0].id)
            
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        print(f"Error in speech synthesis: {e}")

def main():
    audio_streamer = AudioStreamer()
    p = pyaudio.PyAudio()
    
    # Test microphone input devices
    print("\nAvailable Input Devices:")
    for i in range(p.get_device_count()):
        dev_info = p.get_device_info_by_index(i)
        if dev_info['maxInputChannels'] > 0:  # Only show input devices
            print(f"Device {i}: {dev_info['name']}")
    
    # Open stream using callback
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
        stream_callback=audio_streamer.callback
    )

    print("\nAudio Configuration:")
    print(f"THRESHOLD: {THRESHOLD}")
    print(f"CHANNELS: {CHANNELS}")
    print(f"RATE: {RATE}")
    print(f"CHUNK: {CHUNK}")
    
    print("\nBot is now listening... Say 'hey bot' followed by your question!")
    stream.start_stream()

    try:
        current_audio_data = bytearray()
        
        while not audio_streamer.stop_thread:
            audio = audio_streamer.audio_queue.get()
            
            if audio is None and len(current_audio_data) > 0:
                # Only process if we have enough audio data (at least 0.5 seconds)
                min_audio_length = int(RATE * 0.5 * CHANNELS * 2)  # 0.5 seconds of audio
                if len(current_audio_data) < min_audio_length:
                    print(f"Audio segment too short ({len(current_audio_data)} bytes), ignoring...")
                    current_audio_data = bytearray()
                    continue
                    
                print(f"\nProcessing speech segment of {len(current_audio_data)} bytes")
                text_chunk = process_audio_stream(current_audio_data)
                print(f"Transcribed: {text_chunk}")
                
                if text_chunk:
                    lower_text = text_chunk.lower()
                    if WAKE_PHRASE in lower_text:
                        # Extract the question (text after wake phrase)
                        question = lower_text.split(WAKE_PHRASE, 1)[1].strip()
                        if question:
                            print(f"Question detected: {question}")
                            response = generate_bot_response(question)
                            print(f"Bot response: {response}")
                            speak_response(response)
                
                # Reset audio buffer
                current_audio_data = bytearray()
            elif audio is not None:
                # Accumulate audio data
                current_audio_data.extend(audio)

    except KeyboardInterrupt:
        print("\nStopping the bot...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main()