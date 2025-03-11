# Video Creator

A Python-based tool that creates dynamic videos by matching audio or video segments with relevant video clips from Pexels. The tool offers two main functionalities:
1. Audio-to-Video: Converts audio files into videos by transcribing segments and matching them with relevant video content.
2. Video-to-Video: Enhances existing videos by adding relevant visual cues based on speech content, with interactive customization options.

## Prerequisites

- Python 3.9 or higher
- Pexels API key (get one at [Pexels](https://www.pexels.com/api/))
- FFmpeg installed on your system

## Virtual Environment Setup

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:

On Windows:
```bash
venv\Scripts\activate
```

On macOS/Linux:
```bash
source venv/bin/activate
```

You'll know it's activated when you see `(venv)` at the beginning of your terminal prompt.

## Installation

1. Clone the repository
2. With your virtual environment activated, install dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies

- pydub (0.25.1): Audio processing
- moviepy (1.0.3): Video editing
- SpeechRecognition (3.10.0): Audio transcription
- g4f (0.4.7.7): GPT-4 integration for keyword extraction
- pexels-api (1.0.1): Pexels API integration
- pytube (15.0.0): YouTube integration
- requests (2.31.0): HTTP requests
- python-dotenv: Environment variable management
- whisper: OpenAI's speech recognition model

## Features

### Audio to Video (audio_to_video.py)
- Converts audio files into dynamic video presentations
- Automatically transcribes audio segments
- Downloads relevant video content from Pexels
- Supports both landscape (1280x720) and portrait (720x1280) video formats

### Video to Video (video_to_video.py)
- Enhances existing videos with relevant visual cues
- Interactive mode for customizing cue timing and keywords
- Uses Whisper for accurate speech transcription
- GPT-powered keyword extraction for better content matching
- Fixed interval processing (10-second segments)
- Visual cues inserted from 3.5s to 8.0s in each segment
- Supports both landscape and portrait video formats

## Usage

### Audio to Video
1. Prepare an MP3 file that you want to create a video for
2. Run:
```bash
python audio_to_video.py
```

### Video to Video
1. Prepare a video file you want to enhance
2. Run:
```bash
python video_to_video.py
```

### Interactive Mode (video_to_video.py)
When running in interactive mode, you can:
- Review and modify suggested keywords for each segment
- Choose from multiple video options for each visual cue
- Skip visual cues for specific segments
- Fine-tune the video selection process

### Video Aspects
Both scripts support two video aspects:
- Landscape: 1280x720 resolution
- Portrait: 720x1280 resolution

## Output
The final video will be saved as `output.mp4` in the project directory.

## Notes
- The video_to_video.py script processes videos in 10-second intervals
- Visual cues are consistently placed from 3.5s to 8.0s in each segment
- Downloaded videos are cached in the 'downloaded_videos' directory
- Interactive mode provides more control but requires manual input during processing