import ffmpeg
import os
import logging
import moviepy
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)
# Set the FFmpeg path explicitly
FFMPEG_PATH = "/opt/homebrew/bin/ffmpeg"  # Update if necessary
os.environ["PATH"] += os.pathsep + os.path.dirname(FFMPEG_PATH)
os.environ["IMAGEIO_FFMPEG_EXE"] = FFMPEG_PATH
import subprocess
import moviepy.config


import ffmpeg
import os
import logging
import moviepy.config
import subprocess
import shutil
import cv2
import easyocr
import assemblyai as aai
import librosa
import numpy as np
import whisper
import torch
import soundfile as sf
from pydub import AudioSegment
from pydub.effects import speedup
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import speech_recognition as sr
from googletrans import Translator
import torchaudio
from moviepy.editor import VideoFileClip, CompositeAudioClip, AudioFileClip
import spleeter
from spleeter.separator import Separator
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import Response
import uvicorn
import uuid
import translate
from translate import Translator

# Configuration (customize these)
FFMPEG_PATH = "/opt/homebrew/bin/ffmpeg"  # Update if necessary
UPLOAD_DIR = "uploads"
TEMP_DIR = "temp_audio"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
aai.settings.api_key = "4413298478384dc883b39283bf0ca832"  # Replace with your actual API key

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Video-Debugger - AI Video Generator")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Initialize EasyOCR reader
reader = easyocr.Reader(['nl'])

#mp4 to mp3 Conversion
def convert_mp4_to_wav(ffmpeg_path, input_file, output_file):
    try:
        command = [ffmpeg_path, "-i", input_file, "-acodec", "pcm_s16le", "-ar", "16000", output_file]
        subprocess.run(command, check=True)
        print("Conversion successful!")
    except Exception as e:
        print(f"Error: {e}")
    

# Transcribe WAV to Dutch text
def transcribe_audio_to_dutch(audio_file_path, output_txt_path):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file_path) as source:
            audio = recognizer.record(source)

        text = recognizer.recognize_google(audio, language="nl-NL")

        # Write the transcribed text to a file
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"Transcription saved to: {output_txt_path}")
        return text
    except sr.UnknownValueError:
        print(f"Could not understand the audio in {audio_file_path}")
    except sr.RequestError as e:
        print(f"Error with Google Speech Recognition API: {e}")
    except Exception as e:
        print(f"Error processing {audio_file_path}: {e}")
    return ""

def translate_transcription(input_file_path, output_file_path):
    """
    Translate and refine transcription from a text file using advanced NLP models.

    Args:
        input_file_path (str): Path to the input transcription file.
        output_file_path (str): Path to save the refined translated text file.

    Returns:
        None
    """
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

    # Load translation model
    print("Loading translation model...")
    translation_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-nl-en")
    translation_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-nl-en")

    # Load paraphrasing model
    print("Loading paraphrasing model...")
    paraphrase_pipeline = pipeline("text2text-generation", model="facebook/bart-large-cnn")

    # Read the transcription file
    print(f"Reading transcription file: {input_file_path}")
    try:
        with open(input_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: File not found - {input_file_path}")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Process each line: translate and refine
    refined_lines = []
    for i, line in enumerate(lines):
        line = line.strip()
        print(f"Processing line {i + 1}: {line}")
        if "]" in line:
            try:
                # Split the line into timestamp and text
                parts = line.split("]", 1)
                timestamp = parts[0] + "]"
                text = parts[1].strip()

                # Translate the text
                inputs = translation_tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
                translated = translation_model.generate(inputs, max_length=512, num_beams=4, early_stopping=True)
                translated_text = translation_tokenizer.decode(translated[0], skip_special_tokens=True)

                # Refine (paraphrase) the translated text
                print(f"Refining translation for line {i + 1}...")
                refined_text = paraphrase_pipeline(translated_text, max_length=512, num_return_sequences=1)[0]['generated_text']

                # Append the refined text with the timestamp
                refined_lines.append(f"{timestamp} {refined_text}\n")
            except Exception as e:
                print(f"Error processing line {i + 1}: {e}")
        else:
            print(f"Skipping malformed line {i + 1}: {line}")

    # Save the refined transcription to a new file
    try:
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.writelines(refined_lines)
        print(f"Translation and refinement completed. Refined file saved at: {output_file_path}")
    except Exception as e:
        print(f"Error saving refined file: {e}")

def detect_gender_with_whisper(audio_file_path):
    print("Detecting gender using Whisper...")
    
    # Load the Whisper model
    model = whisper.load_model("small")  # You can use "small", "medium", or "large" for better accuracy
    
    # Load the audio file
    audio = whisper.load_audio(audio_file_path)
    audio = whisper.pad_or_trim(audio)
    
    # Extract embeddings
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)
    language = max(probs, key=probs.get)
    
    # Use embeddings for gender detection (example logic)
    if language == "nl":  # Example: Dutch speakers
        return "Male"  # Replace with actual logic
    else:
        return "Female"

def generate_tts_audio(translated_text_file, output_wav, temp_wav):
    print("Generating TTS audio with precise timing...")
    
    # Load the SpeechT5 model and tokenizer
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    
    # Load the speaker embeddings dataset
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    
    # Detect gender from the original audio
    gender = detect_gender_with_whisper(temp_wav)
    print(f"Detected Gender: {gender}")
    
    # Select speaker embedding based on gender
    speaker_id = 0 if gender == "Male" else 7306  # Replace with actual speaker IDs
    speaker_embeddings = torch.tensor(embeddings_dataset[speaker_id]["xvector"]).unsqueeze(0)
    
    # Read the translated text
    with open(translated_text_file, "r", encoding="utf-8") as file:
        lines = file.readlines()
    
    # Initialize an empty audio segment
    final_audio = AudioSegment.silent(duration=0)
    current_time = 0  # In milliseconds
    
    for line in lines:
        parts = line.strip().split("]")
        if len(parts) == 2:
            timestamp = parts[0].strip("[]")
            text = parts[1].strip()
            start_time, end_time = map(float, timestamp.split(" - "))
            
            start_time_ms = int(start_time * 1000)
            end_time_ms = int(end_time * 1000)
            
            if current_time < start_time_ms:
                silence_duration = start_time_ms - current_time
                silence = AudioSegment.silent(duration=silence_duration)
                final_audio += silence
                current_time = start_time_ms
            
            inputs = processor(text=text, return_tensors="pt")
            speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
            sf.write("temp_tts.wav", speech.numpy(), samplerate=16000)
            audio_segment = AudioSegment.from_file("temp_tts.wav")
            print(f"Generated TTS audio for: {text}")
            
            audio_segment = speedup(audio_segment, playback_speed=1.25)
            tts_duration = len(audio_segment)  # In milliseconds
            
            if tts_duration < (end_time_ms - start_time_ms):
                silence_duration = (end_time_ms - start_time_ms) - tts_duration
                silence = AudioSegment.silent(duration=silence_duration)
                audio_segment += silence
            
            final_audio += audio_segment
            current_time += len(audio_segment)
    
    final_audio.export(output_wav, format="wav")
    print(f"TTS audio saved to: {output_wav}")

def get_audio_duration(AUDIO_FILE_PATH):
    """Get the duration of the audio in milliseconds."""
    audio = AudioSegment.from_file(AUDIO_FILE_PATH)
    return len(audio)

def detect_dutch_audio_segments(AUDIO_FILE_PATH):
    """Detect segments in the audio that contain Dutch speech."""
    logger.info("Detecting Dutch audio segments...")
    recognizer = sr.Recognizer()
    audio = AudioSegment.from_file(AUDIO_FILE_PATH)
    segment_duration = 1000  # Process in 5-second chunks
    detected_segments = []

    for i in range(0, len(audio), segment_duration):
        segment = audio[i:i + segment_duration]
        temp_file = os.path.join(TEMP_DIR, f"temp_segment_{i}.wav")
        segment.export(temp_file, format="wav")

        try:
            with sr.AudioFile(temp_file) as source:
                audio_data = recognizer.record(source)
                detected_text = recognizer.recognize_google(audio_data, language="nl-NL")  # Dutch language code
                logger.info(f"Dutch speech detected from {i}ms to {min(i + segment_duration, len(audio))}ms: {detected_text}")
                detected_segments.append((i, min(i + segment_duration, len(audio))))
        except sr.UnknownValueError:
            logger.info(f"No recognizable speech in segment {i}ms to {min(i + segment_duration, len(audio))}ms")
        except Exception as e:
            logger.error(f"Error processing segment {i}ms to {min(i + segment_duration, len(audio))}ms: {str(e)}")

    logger.info(f"Final detected Dutch segments: {detected_segments}")
    return detected_segments

def mute_dutch_audio(AUDIO_FILE_PATH, output_path):
    """Mute detected Dutch audio segments in the given audio file."""
    try:
        audio = AudioSegment.from_file(AUDIO_FILE_PATH)
        detected_segments = detect_dutch_audio_segments(AUDIO_FILE_PATH)

        for start_ms, end_ms in detected_segments:
            logger.info(f"Muting Dutch audio from {start_ms}ms to {end_ms}ms...")
            silent_segment = AudioSegment.silent(duration=(end_ms - start_ms))
            audio = audio[:start_ms] + silent_segment + audio[end_ms:]

        audio.export(output_path, format="wav")
        logger.info(f"Background audio with muted Dutch segments saved to: {output_path}")
    except Exception as e:
        logger.error(f"Error muting Dutch audio: {str(e)}", exc_info=True)
        raise

def extract_background_audio(VIDEO_FILE_PATH, background_audio):
    """Extract background audio from the video using Spleeter."""
    try:
        logger.info("Extracting background audio using Spleeter...")
        video = VideoFileClip(VIDEO_FILE_PATH)
        original_audio = video.audio
        temp_wav = os.path.join(TEMP_DIR, "temp_audio.wav")
        original_audio.write_audiofile(temp_wav, codec="pcm_s16le")

        separator = Separator('spleeter:2stems', stft_backend='tensorflow', multiprocess=False)
        separator.separate_to_file(temp_wav, TEMP_DIR)
        
        bg_audio = os.path.join(TEMP_DIR, "temp_audio", "accompaniment.wav")
        if not os.path.exists(bg_audio):
            raise FileNotFoundError(f"Background audio file not found at {bg_audio}")
        
        mute_dutch_audio(bg_audio, background_audio)
        logger.info(f"Background audio with Dutch speech removed saved to: {background_audio}")
    except Exception as e:
        logger.error(f"Error extracting background audio: {str(e)}", exc_info=True)
        raise

def mix_audio(bg_audio_path, tts_output_path, mixed_output_path):
    try:
        # Load both audio files as AudioSegment objects
        bg_audio = AudioSegment.from_file(bg_audio_path)
        tts_audio = AudioSegment.from_file(tts_output_path)
        
        # Ensure both audio files are synced in channels and duration
        if bg_audio.channels != tts_audio.channels:
            tts_audio = tts_audio.set_channels(bg_audio.channels)
        
        if bg_audio.frame_rate != tts_audio.frame_rate:
            tts_audio = tts_audio.set_frame_rate(bg_audio.frame_rate)
        
        # Mix the audio files
        mixed_audio = bg_audio.overlay(tts_audio)
        
        # Export the mixed audio to the specified path
        mixed_audio.export(mixed_output_path, format="wav")
        print(f"Mixed audio saved to: {mixed_output_path}")
    
    except Exception as e:
        print(f"Error mixing audio: {e}")


def replace_audio_in_video(VIDEO_FILE_PATH, AUDIO_FILE_PATH, dubbed_video):
    """Replace the audio track in the video with the mixed audio."""
    try:
        logger.info("Replacing audio in video...")
        video = VideoFileClip(VIDEO_FILE_PATH)
        new_audio = AudioFileClip(AUDIO_FILE_PATH)
        final_video = video.set_audio(new_audio)
        final_video.write_videofile(dubbed_video, codec="libx264", audio_codec="aac")
        logger.info(f"Final video saved to: {dubbed_video}")
    except Exception as e:
        logger.error(f"Error replacing audio in video: {str(e)}", exc_info=True)
        raise


# Initialize EasyOCR reader
reader = easyocr.Reader(['nl'])  # Replace 'nl' with the desired language code


def get_subtitle_region(frame):
    """Define subtitle region based on video resolution."""
    height, width, _ = frame.shape
    y_min = int(height * 0.85)  # Starting height (85% of the frame height)
    y_max = height             # Bottom of the frame
    x_min = int(width * 0.1)   # Left (10% of the frame width)
    x_max = int(width * 0.9)   # Right (90% of the frame width)
    return x_min, y_min, x_max, y_max


def process_frame(frame):
    """Blur detected subtitles in a frame."""
    frame_copy = frame.copy()
    x_min, y_min, x_max, y_max = get_subtitle_region(frame_copy)
    subtitle_roi = frame_copy[y_min:y_max, x_min:x_max]
    results = reader.readtext(subtitle_roi)

    for (bbox, text, prob) in results:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        x1, y1 = map(int, top_left)
        x2, y2 = map(int, bottom_right)
        x1 += x_min
        x2 += x_min
        y1 += y_min
        y2 += y_min
        roi = frame_copy[y1:y2, x1:x2]
        blurred_roi = cv2.GaussianBlur(roi, (51, 51), 0)
        frame_copy[y1:y2, x1:x2] = blurred_roi
    return frame_copy


def video_callback(frame):
    """Callback function for processing each frame in the video."""
    return process_frame(frame)


def embed_subtitles(blur_video, subtitle_path, blur_and_subtitled_dubbed_video):
    """Embed subtitles into the blurred video using FFmpeg."""
    try:
        ffmpeg_command = [
            "ffmpeg",
            "-i", blur_video,
            "-vf", f"subtitles={subtitle_path}:force_style='Fontname=Arial,Fontsize=24,PrimaryColour=&H00FFFF,SecondaryColour=&H000000,OutlineColour=&H000000,BackColour=&H000000,BorderStyle=3,Outline=1,Shadow=0,Alignment=2,MarginL=20,MarginR=20,MarginV=20,Encoding=UTF-8'",
            "-c:v", "libx264",
            "-crf", "23",
            "-preset", "medium",
            "-c:a", "aac",
            "-b:a", "192k",
            blur_and_subtitled_dubbed_video
        ]
        print("Running FFmpeg command:", " ".join(ffmpeg_command))
        result = subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise Exception(f"FFmpeg command failed with error:\n{result.stderr.decode()}")
        print(f"Subtitles embedded successfully. Output saved to: {blur_and_subtitled_dubbed_video}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def process_video_with_subtitles(dubbed_video, blur_video, subtitle_path, blur_and_subtitled_dubbed_video):
    """Process the video by blurring existing subtitles and embedding new ones."""

    # Step 1: Blur the subtitles (This is working now!)
    video = VideoFileClip(dubbed_video)
    processed_video = video.fl_image(video_callback)
    processed_video = processed_video.set_audio(video.audio) # Keep the dubbed audio
    processed_video.write_videofile(blur_video, codec='libx264', audio_codec='aac', logger=None) # Suppress MoviePy logs
    print("Video processing (blurring) completed with audio.")

    # Step 2: Transcribe the video using AssemblyAI and generate subtitles (FIXED!)
    try:
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(blur_video) # Transcribe the BLURRED video

        if transcript and transcript.text:  # Check if transcription was successful
            subtitles = transcript.export_subtitles_srt()
            with open(subtitle_path, 'w', encoding='utf-8') as srt_file:
                srt_file.write(subtitles)
            print("Transcription and subtitle generation complete. Subtitles saved.")
        else:
            print("AssemblyAI transcription failed or returned empty.")
            # Create an empty SRT file to avoid FFmpeg errors if transcription fails
            with open(subtitle_path, 'w', encoding='utf-8') as srt_file:
                srt_file.write("") # Write empty string
            # Handle the error appropriately, e.g., return an error response
            # raise HTTPException(status_code=500, detail="Transcription failed.") # Or continue with empty subtitles

    except Exception as e:
        print(f"AssemblyAI Error: {e}")
        # Create an empty SRT file to avoid FFmpeg errors if transcription fails
        with open(subtitle_path, 'w', encoding='utf-8') as srt_file:
            srt_file.write("") # Write empty string
        # Handle the error appropriately
        # raise HTTPException(status_code=500, detail=f"Transcription failed: {e}") # Or continue with empty subtitles


    # Step 3: Embed the subtitles into the video (FIXED!)
    embed_subtitles(blur_video, subtitle_path, blur_and_subtitled_dubbed_video)


#Below Code Is for Enhance Video Quality, Specially Audio in Voice
def enhance_video(input_video, output_video):
    extracted_audio = "extracted_audio.wav"
    enhanced_audio = "enhanced_audio.wav"
    noise_reduced_audio = "noise_reduced_audio.wav"
    normalized_audio = "final_audio.wav"

    # Step 1: Extract Audio
    logger.info("Extracting audio...")
    cmd = [FFMPEG_PATH, "-i", input_video, "-q:a", "0", "-map", "a", extracted_audio, "-y"]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Step 2: Enhance Speech
    logger.info("Enhancing speech...")
    wav, sr = torchaudio.load(extracted_audio)
    enhanced_wav = torchaudio.functional.bandpass_biquad(wav, sr, 85, 255)
    torchaudio.save(enhanced_audio, enhanced_wav, sr)

    # Step 3: Reduce Background Noise
    logger.info("Reducing noise...")
    cmd = [FFMPEG_PATH, "-i", enhanced_audio, "-af", "afftdn=nf=-20", noise_reduced_audio, "-y"]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Step 4: Normalize Audio
    logger.info("Normalizing audio...")
    cmd = [FFMPEG_PATH, "-i", noise_reduced_audio, "-af", "loudnorm", normalized_audio, "-y"]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Step 5: Merge Enhanced Audio Back into Video
    logger.info("Merging enhanced audio back into video...")
    cmd = [
        FFMPEG_PATH, "-i", input_video, "-i", normalized_audio,
        "-c:v", "copy", "-c:a", "aac", "-strict", "experimental", output_video, "-y"
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    logger.info(f"\nProcessing Complete! Check the output video: {output_video}")    

@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)  # No Content


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index_2.html", {"request": request})



@app.post("/upload")
async def upload_video(request: Request, video: UploadFile = File(...)):
    unique_id = str(uuid.uuid4())
    video_path = os.path.join(UPLOAD_DIR, f"{unique_id}_{video.filename}")  # Use UUIDs!
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    try:
        base_name = os.path.splitext(video.filename)[0]
        temp_wav = os.path.join(TEMP_DIR, f"{unique_id}_{base_name}.wav") # Use UUIDs!
        temp_txt = os.path.join(TEMP_DIR, f"{unique_id}_{base_name}_transcription.txt")
        translated_txt = os.path.join(TEMP_DIR, f"{unique_id}_{base_name}_translated.txt")
        tts_wav = os.path.join(TEMP_DIR, f"{unique_id}_{base_name}_tts.wav")
        background_audio = os.path.join(TEMP_DIR, f"{unique_id}_{base_name}_background.wav")
        mixed_audio = os.path.join(TEMP_DIR, f"{unique_id}_{base_name}_mixed.wav")
        dubbed_video = os.path.join(TEMP_DIR, f"{unique_id}_{base_name}_dubbed.mp4")
        blur_video = os.path.join(TEMP_DIR, f"{unique_id}_{base_name}_blurred.mp4")
        subtitle_path = os.path.join(TEMP_DIR, f"{unique_id}_{base_name}_subtitles.srt")
        blur_and_subtitled_video = os.path.join(UPLOAD_DIR, f"{unique_id}_subtitled_debugged_{video.filename}")
        enhanced_video = os.path.join(UPLOAD_DIR, f"{unique_id}_debugged_{video.filename}")

        # ... (Call all your processing functions in the correct order as before, using the UUID filenames) ...

        convert_mp4_to_wav(FFMPEG_PATH, video_path, temp_wav)
        transcribe_audio_to_dutch(temp_wav, temp_txt)
        translate_transcription(temp_txt, translated_txt)
        generate_tts_audio(translated_txt, tts_wav, temp_wav)
        extract_background_audio(video_path, background_audio)
        mix_audio(background_audio, tts_wav, mixed_audio)
        replace_audio_in_video(video_path, mixed_audio, dubbed_video)
        process_video_with_subtitles(dubbed_video, blur_video, subtitle_path, blur_and_subtitled_video)
        enhance_video(blur_and_subtitled_video, enhanced_video)

        return FileResponse(enhanced_video, media_type="video/mp4", filename=enhanced_video.split("/")[-1])

    except Exception as e:
        logger.error(f"Error processing video: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ... (rest of the code)
if __name__ == '__main__':
    uvicorn.run("19-feb-app_version-2:app", host="127.0.0.1", port=8080, reload=True)