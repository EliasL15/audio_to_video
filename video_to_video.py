import os
import math
import tempfile
import requests
import re
import json
from collections import Counter
from enum import Enum
from dataclasses import dataclass
from dotenv import load_dotenv

import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip
import moviepy.video.fx.all as vfx
from pydub import AudioSegment, silence
import whisper  # Import Whisper for transcription
import g4f

# Fix for Pillow 10.0 compatibility:
from PIL import Image
Image.ANTIALIAS = Image.Resampling.LANCZOS


@dataclass
class MaterialInfo:
    provider: str = ""
    url: str = ""
    duration: float = 0.0

class VideoAspect(Enum):
    landscape = 1
    portrait = 2

    def to_resolution(self):
        if self == VideoAspect.landscape:
            return (1280, 720)
        elif self == VideoAspect.portrait:
            return (720, 1280)
        return (1280, 720)


def save_video(video_url: str, save_dir: str) -> str:
    """Download video from the provided URL to save_dir and return the local path."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    url_without_query = video_url.split("?")[0]
    video_id = f"vid-{abs(hash(url_without_query))}"
    video_path = os.path.join(save_dir, f"{video_id}.mp4")
    
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


def scale_and_crop_to_fill(clip, new_size):
    """
    Resizes and center-crops `clip` so that it exactly fills new_size
    without letterboxing/pillarboxing.

    new_size: (target_width, target_height)
    """
    target_w, target_h = new_size
    orig_w, orig_h = clip.size

    # Scale factor so the clip fills at least one dimension fully
    scale_w = target_w / orig_w
    scale_h = target_h / orig_h
    scale = max(scale_w, scale_h)

    # Resize the clip
    clip = clip.resize(scale)

    # Crop overflow so the result is exactly target_w x target_h
    resized_w, resized_h = clip.size
    x1 = (resized_w - target_w) / 2
    y1 = (resized_h - target_h) / 2
    x2 = x1 + target_w
    y2 = y1 + target_h

    return clip.crop(x1=x1, y1=y1, x2=x2, y2=y2)


class VideoCreator:
    def __init__(self, video_path, pexels_api_key, video_dir='downloaded_videos',
                 video_aspect=VideoAspect.landscape, interactive=False):
        """
        video_path: Path to your speaking video.
        pexels_api_key: API key for Pexels.
        video_aspect: VideoAspect.landscape or VideoAspect.portrait.
        interactive: If True, allow interactive adjustment of cue timing and keyword.
        """
        self.video_path = video_path
        self.pexels_api_key = pexels_api_key  
        self.video_dir = video_dir
        self.video_aspect = video_aspect
        self.interactive = interactive
        self.temp_dir = tempfile.mkdtemp()
        self.downloaded_video_ids = set()  # Track downloaded video IDs
        if not os.path.exists(self.video_dir):
            os.makedirs(self.video_dir)
        # Load Whisper model (choose size: tiny, base, small, medium, or large)
        self.whisper_model = whisper.load_model("base")
        # Fixed interval of 10 seconds.
        self.interval_sec = 10.0

    def transcribe_segment(self, audio_path):
        """Transcribe a single audio segment using Whisper."""
        try:
            result = self.whisper_model.transcribe(audio_path)
            return result["text"].strip()
        except Exception as e:
            print("Error transcribing using Whisper:", e)
            return ""
    
    def extract_json(self, text):
        """
        Extract JSON content from a GPT response.
        First, try to extract a JSON code block delimited by ```json and ``` .
        If not found, try to extract content between the first '{' and the last '}'.
        """
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return match.group(1)
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            return text[start:end+1]
        return text

    def get_keywords_with_relevance(self, transcript, interval_index, start, end):
        """
        Use GPT (via g4f) to return a JSON array of candidate keyword objects.
        Each object should have:
          - "keyword": the candidate keyword.
          - "timestamp": when (in seconds relative to the segment) the keyword is spoken.
          - "relevance": a float between 0 and 1 indicating how relevant the keyword is.
        """
        prompt = (
            f"Given the following transcript from segment {interval_index} "
            f"({start:.1f}-{end:.1f} seconds):\n\n"
            f"\"{transcript}\"\n\n"
            "Please return a JSON array of candidate keywords for a visual cue. "
            "Each element should be a JSON object with keys 'keyword', 'timestamp' (in seconds relative to the segment), "
            "and 'relevance' (a float between 0 and 1 where 1 is the highest relevance). "
            "Return only the JSON array. For example:\n"
            '[{"keyword": "basketball", "timestamp": 6.0, "relevance": 0.9}, '
            '{"keyword": "sports", "timestamp": 3.0, "relevance": 0.6}]'
        )
        try:
            response = g4f.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            print("Raw GPT response:", repr(response))
            match = re.search(r"```json\s*(\[\s*.*?\s*\])\s*```", response, re.DOTALL)
            if match:
                json_str = match.group(1)
            else:
                start_idx = response.find('[')
                end_idx = response.rfind(']')
                json_str = response[start_idx:end_idx+1] if start_idx != -1 and end_idx != -1 else response
            print("Extracted JSON:", repr(json_str))
            candidates = json.loads(json_str)
            print(f"Segment {interval_index} candidates:", candidates)
            return candidates
        except Exception as e:
            print(f"Error obtaining keywords for segment {interval_index}: {e}")
            return []

    def choose_best_candidate(self, candidates):
        if not candidates:
            return None
        stopwords = {"but", "and", "or", "the", "a", "an"}
        filtered = [c for c in candidates if c.get("keyword", "").lower() not in stopwords]
        if filtered:
            candidates = filtered
        sorted_candidates = sorted(candidates, key=lambda x: (-x.get("relevance", 0), x.get("timestamp", float('inf'))))
        return sorted_candidates[0]

    def refine_keyword_query(self, keyword):
        return f"{keyword} HD"

    def download_top_videos(self, keyword, n=5):
        """
        Download up to the top n videos from Pexels for the given keyword.
        Returns a list of local video file paths.
        """
        headers = {"Authorization": self.pexels_api_key}
        orientation = "landscape" if self.video_aspect == VideoAspect.landscape else "portrait"
        url = f"https://api.pexels.com/videos/search?query={keyword}&per_page={n}&min_duration={int(self.interval_sec)}&orientation={orientation}"
        response = requests.get(url, headers=headers)
        video_paths = []
        if response.status_code == 200:
            data = response.json()
            if data.get('videos'):
                desired_width, desired_height = self.video_aspect.to_resolution()
                for video in data['videos']:
                    video_id = str(video.get('id'))
                    local_path = os.path.join(self.video_dir, f'video_{video_id}.mp4')
                    if os.path.exists(local_path):
                        video_paths.append(local_path)
                        continue
                    duration = video.get('duration', 0)
                    if duration < 15:
                        continue
                    video_files = video.get('video_files', [])
                    for vf in video_files:
                        w = vf.get('width', 0)
                        h = vf.get('height', 0)
                        if w >= desired_width and h >= desired_height:
                            video_url = vf.get('link')
                            video_response = requests.get(video_url, stream=True)
                            if video_response.status_code == 200:
                                with open(local_path, 'wb') as f:
                                    for chunk in video_response.iter_content(chunk_size=1024):
                                        if chunk:
                                            f.write(chunk)
                                self.downloaded_video_ids.add(video_id)
                                video_paths.append(local_path)
                            break
            else:
                print("No videos returned from Pexels for keyword:", keyword)
        elif response.status_code == 429:
            print("Rate limit exceeded for Pexels API. Please try again later.")
        elif response.status_code == 401:
            print("Invalid Pexels API key. Please check your credentials.")
        return video_paths

    def download_pexels_video(self, keyword):
        videos = self.download_top_videos(keyword, n=5)
        if videos:
            return videos[0]
        return None

    def segment_video_fixed(self, video):
        total_duration = video.duration
        segments = []
        current = 0.0
        while current < total_duration:
            segments.append((current, min(current + self.interval_sec, total_duration)))
            current += self.interval_sec
        print("Fixed segments:", segments)
        return segments

    # def adjust_cue_timing(self, audio_path, candidate_time, search_window=1000):
    #     try:
    #         audio_seg = AudioSegment.from_wav(audio_path)
    #     except Exception as e:
    #         print("Error loading audio for cue adjustment:", e)
    #         return candidate_time
    #     candidate_ms = candidate_time * 1000
    #     start_search = max(0, candidate_ms - search_window)
    #     segment = audio_seg[start_search:candidate_ms]
    #     silent_parts = silence.detect_silence(segment, min_silence_len=300, silence_thresh=audio_seg.dBFS - 16)
    #     if silent_parts:
    #         last_silence = silent_parts[-1]
    #         new_candidate_ms = start_search + last_silence[1]
    #         adjusted_time = new_candidate_ms / 1000.0
    #         print(f"Adjusted cue timing from {candidate_time:.2f}s to {adjusted_time:.2f}s based on silence detection.")
    #         return adjusted_time
    #     return candidate_time

    def create_video(self):
        """
        Process the video by splitting it into fixed intervals.
        For each segment:
         1. Transcribe the audio using Whisper.
         2. Use GPT to obtain candidate keywords with timestamps and relevance.
         3. Choose the best candidate.
         4. Optionally adjust cue timing.
         5. In interactive mode, allow adjustment of cue timing and/or keyword.
         6. Build a composite segment with:
             - Original video before the cue.
             - Cue segment: either the downloaded stock footage trimmed to a fixed cue window or the original segment.
             - Original video after the cue.
         7. Join segments without transition effects.
         Additionally, the input video is scaled/cropped to fill the target resolution.
         Note: The cue is always inserted from 3.5 seconds to 8.0 seconds into each segment.
        """
        target_size = self.video_aspect.to_resolution()

        # Scale and crop the original speaking video to fill the target frame
        original_video = VideoFileClip(self.video_path)
        original_video = scale_and_crop_to_fill(original_video, target_size)

        segments = self.segment_video_fixed(original_video)
        final_clips = []

        for idx, (start, end) in enumerate(segments):
            print(f"\nProcessing segment {idx} from {start:.2f} to {end:.2f} seconds.")
            interval_clip = original_video.subclip(start, end)
            # Compute segment length
            segment_length = end - start
            temp_audio_path = os.path.join(self.temp_dir, f'interval_{idx}.wav')
            try:
                audio_clip = interval_clip.audio
                if not hasattr(audio_clip, 'fps'):
                    audio_clip.fps = 44100
                audio_clip.write_audiofile(temp_audio_path, verbose=False, logger=None)
            except Exception as e:
                print(f"Error exporting audio for segment {idx}: {e}")
                final_clips.append(interval_clip)
                continue

            transcript = self.transcribe_segment(temp_audio_path)
            print(f"Segment {idx} transcript: {transcript}")
            candidates = self.get_keywords_with_relevance(transcript, idx, start, end)
            best_candidate = self.choose_best_candidate(candidates)
            if best_candidate:
                keyword = best_candidate.get("keyword", "").strip()
            else:
                keyword = ""
            # Always insert cue from 3.5s to 8.0s into the segment.
            fixed_cue_start = 3.5
            fixed_cue_duration = 4.5

            if self.interactive:
                user_input = input(f"Segment {idx}: Candidate keyword '{keyword}' (cue will be inserted from 3.5s to 8.0s). Accept? (y/n) ")
                if user_input.lower().strip() == 'n':
                    new_input = input("Enter new keyword or type 'none' to skip cue: ").strip()
                    if new_input.lower() in ['none', 'skip', 'no video']:
                        print(f"Segment {idx}: No cue will be inserted. Using original segment.")
                        final_clips.append(interval_clip)
                        continue
                    else:
                        # If only a new keyword is provided, use fixed cue timing.
                        keyword = new_input.strip()

            cue_start = fixed_cue_start
            cue_duration = fixed_cue_duration
            print(f"Segment {idx}: Inserting cue with keyword '{keyword}' from {cue_start:.1f}s to {cue_start + cue_duration:.1f}s.")

            stock_video_path = None
            if keyword:
                if self.interactive:
                    skip_cue = False
                    while True:
                        top_videos = self.download_top_videos(keyword, n=5)
                        if top_videos:
                            print("Top video candidates:")
                            for i, path in enumerate(top_videos):
                                print(f"[{i}] {path}")
                            choice = input("Enter the number of the video to use, type 'change' to change keyword, or type 'none' to skip cue (or press Enter for first): ").strip()
                            if choice.lower() in ['none', 'skip', 'no video']:
                                print(f"Segment {idx}: No cue will be inserted. Using original segment.")
                                skip_cue = True
                                break
                            if choice.lower() in ['change', 'c']:
                                keyword = input("Enter new keyword: ").strip()
                                continue
                            try:
                                index = int(choice) if choice != "" else 0
                                if 0 <= index < len(top_videos):
                                    stock_video_path = top_videos[index]
                                else:
                                    print("Invalid choice; defaulting to first candidate.")
                                    stock_video_path = top_videos[0]
                            except:
                                stock_video_path = top_videos[0]
                            break
                        else:
                            print("No candidate videos found for keyword:", keyword)
                            change = input("Would you like to change the keyword? (y/n): ").strip().lower()
                            if change == 'y':
                                keyword = input("Enter new keyword: ").strip()
                                continue
                            else:
                                break
                    if skip_cue:
                        final_clips.append(interval_clip)
                        continue
                else:
                    stock_video_path = self.download_pexels_video(keyword)

            stock_clip = None
            if stock_video_path:
                try:
                    stock_clip = VideoFileClip(stock_video_path)
                    # If the stock clip is shorter than the cue duration, repeat it
                    if stock_clip.duration < cue_duration:
                        repeats = math.ceil(cue_duration / stock_clip.duration)
                        stock_clip = concatenate_videoclips([stock_clip] * repeats)
                    stock_clip = stock_clip.subclip(0, cue_duration)

                    # Fill the entire frame without black bars
                    stock_clip = scale_and_crop_to_fill(stock_clip, target_size)

                    # Set the audio for the stock clip to match the segment’s audio portion
                    audio_stock = interval_clip.audio.subclip(cue_start, cue_start + cue_duration)
                    stock_clip = stock_clip.set_audio(audio_stock)
                except Exception as e:
                    print(f"Error processing stock footage for segment {idx}: {e}")
                    stock_clip = None
            else:
                print(f"No stock footage found for keyword '{keyword}' in segment {idx}.")

            clips_to_concat = []
            if cue_start > 0:
                clips_to_concat.append(interval_clip.subclip(0, cue_start))
            if stock_clip:
                clips_to_concat.append(stock_clip)
            else:
                # If no stock clip, just use the original portion
                clips_to_concat.append(interval_clip.subclip(cue_start, cue_start + cue_duration))
            remaining = segment_length - (cue_start + cue_duration)
            if remaining > 0:
                clips_to_concat.append(interval_clip.subclip(cue_start + cue_duration, segment_length))
            composite_clip = concatenate_videoclips(clips_to_concat)
            final_clips.append(composite_clip)

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
                print("Video creation completed successfully! Output saved as output.mp4")
            except Exception as e:
                print(f"Error during final video creation: {e}")
        else:
            print("No clips to concatenate. Video creation aborted.")


def main():
    load_dotenv()
    PEXELS_API_KEY = os.getenv('PEXELS_API_KEY')
    if not PEXELS_API_KEY:
        print("Error: PEXELS_API_KEY not found in environment variables")
        return
    
    video_path = input('Enter the path to your video file (with you speaking): ').strip()
    if not os.path.exists(video_path):
        print(f"Error: File {video_path} does not exist")
        return

    aspect_input = input("Enter video aspect (landscape/portrait): ").strip().lower()
    video_aspect = VideoAspect.portrait if aspect_input == "portrait" else VideoAspect.landscape

    interactive_input = input("Run in interactive mode? (y/n): ").strip().lower()
    interactive_mode = interactive_input == "y"

    video_dir = 'downloaded_videos'
    creator = VideoCreator(video_path, PEXELS_API_KEY, video_dir, video_aspect=video_aspect, interactive=interactive_mode)
    creator.create_video()
    print(f"Video creation process completed!\nDownloaded videos are stored in {video_dir}/")


if __name__ == '__main__':
    main()
