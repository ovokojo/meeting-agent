# Voice Interactive Agent

A simple voice-interactive agent that listens for a wake phrase ("hey bot"), transcribes speech using OpenAI's Whisper API, generates responses using GPT-3.5-turbo, and speaks back using text-to-speech.

## Features

- Wake word detection ("hey bot")
- Real-time audio streaming and speech detection
- **Audio-to-text transcription using GPT-4o**
- **Audio response generation using GPT-4o**
- Multi-turn voice conversations
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

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root directory:

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
4. The bot will respond with both text and audio
5. Continue the conversation naturally - context is maintained

## Configuration

Updated configuration options:
- `WAKE_PHRASE`: The trigger phrase (default: "hey bot")
- `VOICE_CHARACTER`: Choose from "alloy", "echo", "fable", "onyx", "nova", or "shimmer"
- `MAX_CONVERSATION_HISTORY`: Number of turns to keep in memory (default: 5)

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

4. **Audio playback issues:**
   - Ensure speakers are properly configured
   - Verify file permissions for writing WAV files
   - Check if generated WAV files contain audio (try playing manually)

5. **GPT-4o access issues:**
   - Verify your OpenAI account has access to GPT-4o models
   - Check your API quota limits
   - Ensure you're using OpenAI Python package v1.12+