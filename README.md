# Video Creator

A Python-based tool that creates dynamic videos by matching audio segments with relevant video clips from Pexels. The tool transcribes audio segments, extracts keywords, and automatically downloads matching video content to create a cohesive video presentation.

## Prerequisites

- Python 3.9 or higher
- Pexels API key (get one at [Pexels](https://www.pexels.com/api/))
- FFmpeg installed on your system

## Installation

1. Clone the repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

1. Create a `.env` file in the project root directory:
```bash
touch .env
```

2. Add your Pexels API key to the `.env` file:
```
PEXELS_API_KEY=your-api-key-here
```

Make sure to replace `your-api-key-here` with your actual Pexels API key.

**Note**: Never commit your `.env` file to version control. The repository includes a `.gitignore` file that already excludes it.

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

## Dependencies

- pydub (0.25.1): Audio processing
- moviepy (1.0.3): Video editing
- SpeechRecognition (3.10.0): Audio transcription
- g4f (0.4.7.7): GPT-4 integration for keyword extraction
- pexels-api (1.0.1): Pexels API integration
- pytube (15.0.0): YouTube integration
- requests (2.31.0): HTTP requests
- python-dotenv: Environment variable management

## Error Handling

If you encounter the error "PEXELS_API_KEY not found in environment variables", make sure:
1. You have created the `.env` file in the project root
2. The file contains the correct API key in the format specified above
3. The python-dotenv package is installed

## License

This project is licensed under the MIT License - see the LICENSE file for details.