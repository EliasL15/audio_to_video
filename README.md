# Video Creator

A Python-based tool that creates dynamic videos by matching audio segments with relevant video clips from Pexels. The tool transcribes audio segments, extracts keywords, and automatically downloads matching video content to create a cohesive video presentation.

## Features

- Audio segmentation into 8-second intervals
- Speech-to-text transcription using Google's Speech Recognition
- Intelligent keyword extraction using GPT-4 (with fallback to text analysis)
- Automatic video content sourcing from Pexels
- Support for both landscape (1280x720) and portrait (720x1280) video formats
- Automatic video looping and trimming to match audio duration
- Fallback to neutral video content when specific matches aren't found

## Prerequisites

- Python 3.9 or higher
- Pexels API key (get one at [Pexels](https://www.pexels.com/api/))
- FFmpeg installed on your system

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd sotiriou
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Configuration

1. Replace the `PEXELS_API_KEY` in `video_creator.py` with your own API key:
```python
PEXELS_API_KEY = 'your-api-key-here'
```

## Usage

1. Prepare an MP3 file that you want to create a video for.

2. Run the script:
```bash
python video_creator.py
```

3. When prompted, enter the path to your MP3 file.

4. The script will:
   - Split the audio into segments
   - Transcribe each segment
   - Download relevant videos from Pexels
   - Create a final video with the matched clips

5. The output video will be saved as `output.mp4` in the project directory.

## Video Aspect Ratio

By default, the tool creates landscape videos (1280x720). To create portrait videos, modify the `VideoAspect` parameter when creating the `VideoCreator` instance:

```python
creator = VideoCreator(audio_path, PEXELS_API_KEY, video_dir, video_aspect=VideoAspect.portrait)
```

## Project Structure

- `video_creator.py`: Main script containing the VideoCreator class and logic
- `downloaded_videos/`: Directory where downloaded video clips are stored
- `requirements.txt`: List of Python package dependencies

## Dependencies

- pydub (0.25.1): Audio processing
- moviepy (1.0.3): Video editing
- SpeechRecognition (3.10.0): Audio transcription
- g4f (0.4.7.7): GPT-4 integration for keyword extraction
- pexels-api (1.0.1): Pexels API integration
- pytube (15.0.0): YouTube integration
- requests (2.31.0): HTTP requests

## Error Handling

The tool includes robust error handling for:
- Invalid API credentials
- Rate limiting
- Failed video downloads
- Transcription errors
- Video processing issues

## Notes

- Downloaded videos are cached in the `downloaded_videos` directory to prevent redundant downloads
- The tool uses a fallback keyword system when specific matches aren't found
- Temporary audio segments are automatically cleaned up after processing

## License

This project is licensed under the MIT License - see the LICENSE file for details.