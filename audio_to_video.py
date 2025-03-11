import os
import math
import tempfile
import requests
import random
import re
from collections import Counter
from enum import Enum
from dataclasses import dataclass
from dotenv import load_dotenv

from pydub import AudioSegment
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
import speech_recognition as sr
import g4f

# Load environment variables
load_dotenv()

# ----- Definitions for missing types and functions -----

@dataclass
class MaterialInfo:
    provider: str = ""
    url: str = ""
    duration: float = 0.0

class VideoAspect(Enum):
    landscape = 1
    portrait = 2

    def to_resolution(self):
        # Define the desired resolution for each aspect
        if self == VideoAspect.landscape:
            return (1280, 720)
        elif self == VideoAspect.portrait:
            return (720, 1280)
        # Default fallback resolution
        return (1280, 720)

class VideoConcatMode(Enum):
    random = 1
    sequential = 2

def save_video(video_url: str, save_dir: str) -> str:
    """Download video from the provided URL to save_dir and return the local path."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Create a file name based on the URL (using a simple hash here)
    url_without_query = video_url.split("?")[0]
    video_id = f"vid-{abs(hash(url_without_query))}"
    video_path = os.path.join(save_dir, f"{video_id}.mp4")
    
    # If already downloaded, return it.
    if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
        return video_path
    
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(video_url, headers=headers, stream=True)
    if response.status_code == 200:
        with open(video_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        return video_path
    return ""

# ----- End of definitions -----

class VideoCreator:
    def __init__(self, audio_path, pexels_api_key, video_dir='downloaded_videos', video_aspect=VideoAspect.landscape):
        self.audio_path = audio_path
        self.pexels_api_key = pexels_api_key  # For Pexels API requests
        self.video_dir = video_dir
        self.video_aspect = video_aspect
        self.temp_dir = tempfile.mkdtemp()
        self.downloaded_video_ids = set()  # Track downloaded video IDs
        if not os.path.exists(self.video_dir):
            os.makedirs(self.video_dir)
        self.recognizer = sr.Recognizer()
        self.interval = 8000  # 8 seconds in milliseconds
        
    def split_audio(self):
        """Split audio into 8-second intervals."""
        audio = AudioSegment.from_mp3(self.audio_path)
        duration = len(audio)
        interval = self.interval  # 8 seconds in milliseconds
        segments = []
        
        for start in range(0, duration, interval):
            end = start + interval if start + interval < duration else duration
            segment = audio[start:end]
            temp_path = os.path.join(self.temp_dir, f'segment_{start//interval}.wav')
            segment.export(temp_path, format='wav')
            segments.append(temp_path)
            
        return segments
    
    def transcribe_segment(self, segment_path):
        """Transcribe a single audio segment."""
        with sr.AudioFile(segment_path) as source:
            audio = self.recognizer.record(source)
            try:
                text = self.recognizer.recognize_google(audio)
                return text
            except sr.UnknownValueError:
                return ""
            except sr.RequestError:
                return ""
    
    def get_keywords(self, text):
        """Get keywords using G4F with fallback to basic text analysis."""
        if not text:
            return []
        try:
            prompt = f"Given this text: '{text}', provide 3-5 relevant keywords for visual content. Response should be comma-separated keywords only."
            response = g4f.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            return [keyword.strip() for keyword in response.split(',')]
        except Exception as e:
            print(f"G4F keyword generation failed, using fallback method: {e}")
            cleaned_text = re.sub(r'[^\w\s]', '', text.lower())
            words = cleaned_text.split()
            stop_words = {'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have',
                          'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you',
                          'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they',
                          'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one',
                          'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out',
                          'if', 'about', 'who', 'get', 'which', 'go', 'me'}
            content_words = [w for w in words if w not in stop_words and len(w) > 2]
            freq = Counter(content_words)
            keywords = [word for word, _ in freq.most_common(5)]
            return keywords if keywords else ['nature']
    
    def download_pexels_video(self, keywords):
        """Download a relevant video from Pexels using the given keywords.
           Falls back to neutral keywords if necessary."""
        video_path = None
        # Try each provided keyword
        for keyword in keywords:
            video_path = self._download_video_by_keyword(keyword)
            if video_path:
                return video_path
        # Fallback neutral keywords if none found
        fallback_keywords = ["nature", "landscape", "abstract"]
        for keyword in fallback_keywords:
            video_path = self._download_video_by_keyword(keyword)
            if video_path:
                return video_path
        return None

    def _download_video_by_keyword(self, keyword):
        headers = {"Authorization": self.pexels_api_key}
        url = f"https://api.pexels.com/videos/search?query={keyword}&per_page=10&min_duration={self.interval // 1000}"
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('videos'):
                # Retrieve the desired resolution for filtering
                desired_width, desired_height = self.video_aspect.to_resolution()
                for video in data['videos']:
                    video_id = str(video.get('id'))
                    # Skip if we've already downloaded this video
                    if video_id in self.downloaded_video_ids:
                        continue
                        
                    duration = video.get('duration', 0)
                    if duration < 15:
                        continue
                    video_files = video.get('video_files', [])
                    # Check each video file for matching resolution
                    for vf in video_files:
                        w = vf.get('width', 0)
                        h = vf.get('height', 0)
                        if w == desired_width and h == desired_height:
                            video_url = vf.get('link')
                            video_path = os.path.join(self.video_dir, f'video_{video_id}.mp4')
                            if os.path.exists(video_path):
                                self.downloaded_video_ids.add(video_id)
                                return video_path
                            video_response = requests.get(video_url, stream=True)
                            if video_response.status_code == 200:
                                with open(video_path, 'wb') as f:
                                    for chunk in video_response.iter_content(chunk_size=1024):
                                        if chunk:
                                            f.write(chunk)
                                self.downloaded_video_ids.add(video_id)
                                return video_path
                return None  # No suitable videos found
            else:
                print("No videos returned from Pexels.")
                return None
        elif response.status_code == 429:
            print("Rate limit exceeded for Pexels API. Please try again later.")
            return None
        elif response.status_code == 401:
            print("Invalid Pexels API key. Please check your credentials.")
            return None
        return None
    
    def create_video(self):
        """Create the final video by matching audio segments with video clips."""
        audio_segments = self.split_audio()
        final_clips = []
        
        for i, segment_path in enumerate(audio_segments):
            text = self.transcribe_segment(segment_path)
            keywords = self.get_keywords(text)
            video_path = self.download_pexels_video(keywords)
            
            if video_path:
                try:
                    # Load video and audio clips
                    video = VideoFileClip(video_path)
                    audio = AudioFileClip(segment_path)
                    
                    # Loop the video if it's shorter than the audio segment
                    if video.duration < audio.duration:
                        repeats = math.ceil(audio.duration / video.duration)
                        video = concatenate_videoclips([video] * repeats)
                    
                    # Trim video to match the audio segment exactly
                    video = video.subclip(0, audio.duration)
                    
                    # Set the audio to the video clip
                    final_clip = video.set_audio(audio)
                    final_clips.append(final_clip)
                except Exception as e:
                    print(f"Error processing segment {i}: {e}")
                    continue
            else:
                print(f"No video found for segment {i} even after fallback.")
        
        if final_clips:
            try:
                final_video = concatenate_videoclips(final_clips, method="compose")
                final_video.write_videofile('output.mp4', codec='libx264', audio_codec='aac')
                # Clean up downloaded videos after successful video creation
                if os.path.exists(self.video_dir) and os.path.isdir(self.video_dir):
                    for file in os.listdir(self.video_dir):
                        file_path = os.path.join(self.video_dir, file)
                        try:
                            if os.path.isfile(file_path):
                                os.remove(file_path)
                        except Exception as e:
                            print(f"Error removing file {file_path}: {e}")
                    print("Downloaded videos cleaned up successfully.")
                print("Video creation completed successfully!")
            except Exception as e:
                print(f"Error during final video creation: {e}")
        else:
            print("No clips to concatenate. Video creation aborted.")

def main():
    # Load Pexels API key from environment variable
    PEXELS_API_KEY = os.getenv('PEXELS_API_KEY')
    if not PEXELS_API_KEY:
        print("Error: PEXELS_API_KEY not found in environment variables")
        return
    
    audio_path = input('Enter the path to your MP3 file: ')
    if not os.path.exists(audio_path):
        print(f"Error: File {audio_path} does not exist")
        return
    
    video_dir = 'downloaded_videos'
    # You can choose the video aspect; here we default to landscape.
    creator = VideoCreator(audio_path, PEXELS_API_KEY, video_dir, video_aspect=VideoAspect.landscape)
    creator.create_video()
    print(f"Video creation completed! Check output.mp4\nDownloaded videos are stored in {video_dir}/")

if __name__ == '__main__':
    main()