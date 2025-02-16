# Voice Interactive Agent

A simple voice-interactive agent that listens for a wake phrase ("hey bot"), transcribes speech using OpenAI's Whisper API, generates responses using GPT-3.5-turbo, and speaks back using text-to-speech.

## Features

- Wake word detection ("hey bot")
- Real-time audio streaming and speech detection
- Speech-to-text using OpenAI's Whisper API
- Response generation using GPT-3.5-turbo
- Text-to-speech response using pyttsx3
- Configurable audio parameters and thresholds

## Prerequisites

- Python 3.8 or higher
- An OpenAI API key
- A working microphone
- Speakers or headphones for audio output

## Installation

1. Create and activate a virtual environment:

```bash
# On Windows
python -m venv venv
.\venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

2. Create a `requirements.txt` file with the following dependencies:

```
openai
python-dotenv
pyaudio
pyttsx3
numpy
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root directory:

```
OPENAI_API_KEY=your_api_key_here
```

Replace `your_api_key_here` with your actual OpenAI API key.

## Audio Setup

The script will automatically list available input devices when started. You may need to configure your system's default audio input/output devices or modify the script to use specific device indices.

### PyAudio Installation Notes

- On Windows: PyAudio should install directly through pip
- On macOS: You may need to install PortAudio first:
  ```bash
  brew install portaudio
  ```
- On Linux: Install PortAudio development package first:
  ```bash
  sudo apt-get install python3-dev portaudio19-dev
  ```

## Usage

1. Ensure your virtual environment is activated
2. Run the script:

```bash
python main.py
```

3. When the bot is listening, say "hey bot" followed by your question
4. Wait for the bot to process your speech and respond

## Configuration

You can modify the following parameters in the script:

- `CHUNK`: Buffer size for audio processing
- `RATE`: Audio sampling rate (default: 16000 Hz)
- `THRESHOLD`: Amplitude threshold for speech detection
- `RECORD_SECONDS`: Recording window duration
- `WAKE_PHRASE`: The trigger phrase (default: "hey bot")

## Troubleshooting

1. **No audio input detected:**
   - Check if your microphone is properly connected
   - Verify the correct input device is selected
   - Adjust the `THRESHOLD` value based on your microphone sensitivity

2. **PyAudio installation issues:**
   - Make sure you have the appropriate audio development libraries installed for your OS
   - On Windows, try using a precompiled wheel if pip installation fails

3. **API errors:**
   - Verify your OpenAI API key is correctly set in the `.env` file
   - Check your internet connection
   - Ensure you have sufficient API credits