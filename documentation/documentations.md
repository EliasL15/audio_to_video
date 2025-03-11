# Video Cue Insertion and Composition Tool Documentation

## 1. Overview

This tool automates the process of creating a composite video by inserting “cue” segments into a speaking video. Each cue is a short clip (stock footage) that corresponds to keywords extracted from the speaker’s audio. The tool uses transcription, GPT-based keyword extraction, and external video sources to generate a final composed video.

The script performs the following tasks:

* **Input:** A video file where a person is speaking.
* **Audio Processing:** Splits the video into fixed-length segments (10 seconds each) and transcribes each segment using Whisper.
* **Keyword Extraction:** Uses GPT (via the g4f library) to extract candidate keywords with associated timestamps and relevance scores from the transcript.
* **Cue Insertion:** Downloads relevant stock footage from the Pexels API based on the best keyword candidate. In each segment, a “cue” clip is inserted (typically from 3.5s to 8.0s).
* **Video Composition:** The original segment is split into parts (before cue, cue, and after cue) and recombined. If no suitable stock footage is found, the original segment is used.
* **Output:** The final composed video is saved as `output.mp4`.
* **Interactive Mode:** Optionally, the tool can run in an interactive mode, allowing the user to manually adjust keywords and select video candidates.

## 2. Setup and Requirements

### Environment Variables

* `PEXELS_API_KEY`: The Pexels API key must be set in the environment. This key is required to fetch stock videos.

### Dependencies

* **Python Libraries:** `os`, `math`, `tempfile`, `requests`, `re`, `json`
* `collections.Counter`, `enum.Enum`, `dataclasses.dataclass`
* `dotenv` (for loading environment variables)
* `numpy` (for numerical operations)
* `moviepy` (for video editing and processing)
* `pydub` (for audio processing and silence detection)
* `whisper` (for audio transcription)
* `g4f` (to interface with GPT-4 for keyword extraction)
* `PIL` (for image processing adjustments)

Ensure that these dependencies are installed in your Python environment.

## 3. Key Components and Their Functionality

### A. Helper Enums

1.  **VideoAspect Enum**

    * **Purpose:** Defines two video orientations: landscape and portrait.
    * **Method:** `to_resolution()`: Returns a target resolution tuple. For example, landscape returns `(1280, 720)`.

### B. Helper Functions

1.  **`save_video(video_url: str, save_dir: str) -> str`**

    * **What it does:** Downloads a video from a given URL.
    * Saves the video in a specified directory.
    * Uses a hash of the URL (without query parameters) to generate a unique filename.
    * **Key Points:** Checks if the video already exists and avoids re-downloading. Streams and writes video content in chunks.

2.  **`scale_and_crop_to_fill(clip, new_size)`**

    * **What it does:** Resizes a video clip so that it completely fills the target resolution.
    * Performs a center-crop if necessary to avoid any letterboxing or pillarboxing.
    * **Key Steps:** Computes scale factors for both width and height. Resizes the clip by the larger scale factor. Crops the center portion to match the exact target dimensions.

### C. The VideoCreator Class

This is the core class that orchestrates the video processing.

1.  **Initialization (`__init__`)**

    * **Parameters:** `video_path`, `pexels_api_key`, `video_dir`, `video_aspect`, `interactive`.
    * **Actions:** Creates a temporary directory for intermediate files. Initializes the Whisper transcription model. Sets a fixed segment duration (`interval_sec = 10.0`).

2.  **`transcribe_segment(audio_path)`**

    * **Purpose:** Transcribes an audio segment using the Whisper model.
    * **Process:** Loads the audio file and returns the transcribed text. Catches and prints errors if transcription fails.

3.  **`extract_json(text)`**

    * **Purpose:** Extracts JSON data from a text string.
    * **Process:** Looks for a code block formatted as `json ...`. If not found, it looks for a JSON object by finding the first `{` and last `}`.

4.  **`get_keywords_with_relevance(transcript, interval_index, start, end)`**

    * **Purpose:** Uses GPT (via `g4f.ChatCompletion`) to generate candidate keywords from a transcript.
    * **Process:** Prepares a prompt and sends it to GPT. Extracts a JSON array containing objects with "keyword", "timestamp", and "relevance".
    * **Error Handling:** Prints an error and returns an empty list if the GPT request fails.

5.  **`choose_best_candidate(candidates)`**

    * **Purpose:** Filters and sorts the list of candidate keywords.
    * **Process:** Removes common stopwords. Sorts by relevance (descending) and then by timestamp (ascending). Returns the top candidate.

6.  **`refine_keyword_query(keyword)`**

    * **Purpose:** Modifies the keyword to include a quality hint (e.g., appending "HD") before searching for stock footage.

7.  **`download_top_videos(keyword, n=5)`**

    * **Purpose:** Downloads up to n videos from Pexels that match the provided keyword.
    * **Process:** Makes a request to the Pexels API. Filters videos based on orientation and minimum duration. Checks that the video resolution meets the target resolution. Downloads and saves the video locally.
    * **Error Handling:** Prints messages if no videos are found, or if there are issues such as rate limits or invalid API keys.

8.  **`download_pexels_video(keyword)`**

    * **Purpose:** Downloads the top (first) video from the list of top videos for a given keyword.

9.  **`segment_video_fixed(video)`**

    * **Purpose:** Splits the video into fixed segments of 10 seconds.
    * **Process:** Iterates over the total duration of the video. Creates a list of tuples with start and end times for each segment.

10. **`adjust_cue_timing(audio_path, candidate_time, search_window=1000)`**

    * **Purpose:** Adjusts the cue insertion time based on silence detection in the audio.
    * **Process:** Loads the audio segment using `pydub`. Analyzes the portion before the candidate time for silence. Adjusts the candidate time if silence is found.

11. **`create_video()`**

    * **Purpose:** Orchestrates the entire process of creating the final composite video.
    * **Step-by-Step Process:**
        * Loads and scales the original video.
        * Splits the video into 10-second segments.
        * For each segment:
            * Extracts audio and transcribes it.
            * Uses GPT to obtain candidate keywords.
            * Chooses the best candidate.
            * Interactive mode prompts.
            * Downloads stock footage.
            * Processes the stock clip.
            * Constructs a composite segment.
        * Concatenates all segments.
        * Writes the final video to `output.mp4`.
        * Cleanup.

### D. The `main()` Function

* **Purpose:** Acts as the entry point for the script.
* **Process:**
    * Loads environment variables.
    * Prompts the user for input video path, aspect ratio, and interactive mode.
    * Instantiates a `VideoCreator` object.
    * Calls `create_video()`.
    * Prints a success message.

## 4. Execution Flow Summary

* **Setup:** Loads environment variables and user inputs.
* **Video Loading and Preprocessing:** Loads and scales the input video.
* **Segment Processing:** Splits the video into 10-second intervals.
* **Final Composition:** Concatenates all processed segments.
* **Cleanup:** Removes temporary files.

## 5. Error Handling and Cleanup

* **Error Checks:** If transcription fails, the original segment is used. If no stock footage is found, the original video segment is used.
* **Resource Cleanup:** Temporary files and downloaded stock footage are removed after processing.