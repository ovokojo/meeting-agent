# Meeting Agent

A simple voice-interactive agent that listens for a customizable wake phrase, transcribes speech using OpenAI's Whisper API, generates responses using GPT-4o, and speaks back using text-to-speech.

## Features

- Customizable wake phrase (set at launch)
- Real-time audio streaming and speech detection
- **Audio-to-text transcription using GPT-4o**
- **Audio response generation using GPT-4o**
- Multi-turn voice conversations
- Configurable audio parameters and thresholds

## Prerequisites

- Python 3.11
- An OpenAI API key
- A working microphone
- Speakers or headphones for audio output

## Express Setup (macOS/Linux)

For a quicker setup on macOS or Linux systems, you can use the provided shell script:

1. Make the script executable:
```bash
chmod +x run.sh
```

2. Run the script:
```bash
./run.sh
```

This script will:
- Create a virtual environment if it doesn't exist
- Activate the virtual environment
- Install all required packages
- Start the application

## Step-by-Step Installation

1. Install system dependencies:

```bash
# On macOS
brew install portaudio ffmpeg

# On Ubuntu/Linux
sudo apt-get install python3-pyaudio portaudio19-dev ffmpeg
```

2. Create and activate a virtual environment:

```bash
# On Windows
python3.11 -m venv venv
.\venv\Scripts\activate

# On macOS/Linux
python3.11 -m venv venv
source venv/bin/activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root directory:

```
OPENAI_API_KEY=your_api_key_here
SYSTEM_CONTEXT=your_custom_system_prompt_here
WAKE_PHRASE=your_custom_wake_phrase_here
```

Replace:
- `your_api_key_here` with your actual OpenAI API key
- `your_custom_system_prompt_here` with your desired system context (defaults to "You are a helpful assistant" if not set)
- `your_custom_wake_phrase_here` with your desired wake phrase (defaults to "meeting agent" if not set)

## Audio Setup

The script will automatically list available input devices when started. You may need to configure your system's default audio input/output devices or modify the script to use specific device indices.

### PyAudio Installation Notes

- On Windows: PyAudio should install directly through pip
- On macOS: You may need to install PortAudio and FFmpeg first:
  ```bash
  brew install portaudio ffmpeg
  ```
- On Linux: Install PortAudio development package first:
  ```bash
  sudo apt-get install python3-dev portaudio19-dev
  ```

## Usage

1. Ensure your virtual environment is activated
2. Run the script:

```bash
python agent.py
```

3. Select your audio input device (or press Enter for default)
4. Enter your preferred wake phrase (or press Enter to use the default "meeting agent")
5. When the assistant is listening, say your wake phrase followed by your question
6. The assistant will respond with both text and audio
7. Continue the conversation naturally - context is maintained

## Configuration

Updated configuration options:
- `WAKE_PHRASE`: Customizable at launch (default: "meeting agent")
- `SYSTEM_CONTEXT`: The AI's persona and behavior (default: "You are a helpful assistant")

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

## License

MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.