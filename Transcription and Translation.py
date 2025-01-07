import os
from pydub import AudioSegment
from pydub.utils import which

#specify the path to ffmpeg and ffprobe
AudioSegment.converter = which("ffmpeg")
AudioSegment.converter = which("ffprobe")


AudioSegment.converter = "/opt/homebrew/bin/ffmpeg//ffmpeg.exe"
AudioSegment.ffmpeg = "/opt/homebrew/bin/ffmpeg//ffmpeg.exe"
AudioSegment.ffprobe ="/opt/homebrew/bin/ffprobe//ffprobe.exe"


import speech_recognition as sr
from googletrans import Translator


# Convert .mp3 to .wav
def mp3_to_wav(mp3_file, wav_file):
    try:
        audio = AudioSegment.from_mp3(mp3_file)
        audio.export(wav_file, format="wav")
        print(f"Converted {mp3_file} to {wav_file}")
    except Exception as e:
        print(f"Error converting {mp3_file} to WAV: {e}")


# Transcribe WAV to Dutch text
def transcribe_audio_to_dutch(audio_file_path):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file_path) as source:
            audio = recognizer.record(source)

        text = recognizer.recognize_google(audio, language="nl-NL")
        return text
    except sr.UnknownValueError:
        print(f"Could not understand the audio in {audio_file_path}")
    except sr.RequestError as e:
        print(f"Error with Google Speech Recognition API: {e}")
    except Exception as e:
        print(f"Error processing {audio_file_path}: {e}")
    return ""


# Translate Dutch text to English
def translate_text(text, target_lang="en"):
    translator = Translator()
    try:
        translated = translator.translate(text, dest=target_lang)
        return translated.text
    except Exception as e:
        print(f"Error translating text: {e}")
        return ""


# Process files in the folder
def process_files(input_folder, output_folder, process_function, file_ext, ext_suffix):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(file_ext):
            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, os.path.splitext(filename)[0] + ext_suffix)

            # Apply the process function
            result = process_function(input_file)
            if result:
                with open(output_file, 'w', encoding='utf-8') as file:
                    file.write(result)
                print(f"Processed {filename} -> {os.path.basename(output_file)}")


# Main Workflow
if __name__ == "__main__":
    input_mp3_folder = '/Users/talharauf/Desktop/Bots/Transcription/mp3 videos/'
    wav_folder = '/Users/talharauf/Desktop/Bots/Transcription/Dutch wav/'
    transcription_folder = '/Users/talharauf/Desktop/Bots/Transcription/Dutch Transcription/'
    translation_folder = '/Users/talharauf/Desktop/Bots/Transcription/English Translation/'

    # Step 1: Convert MP3 to WAV
    if not os.path.exists(wav_folder):
        os.makedirs(wav_folder)
    for filename in os.listdir(input_mp3_folder):
        if filename.endswith('.mp3'):
            mp3_file_path = os.path.join(input_mp3_folder, filename)
            wav_file_path = os.path.join(wav_folder, os.path.splitext(filename)[0] + '.wav')
            mp3_to_wav(mp3_file_path, wav_file_path)

    # Step 2: Transcribe WAV to Dutch Text
    process_files(
        wav_folder, transcription_folder,
        lambda audio_file: transcribe_audio_to_dutch(audio_file),
        '.wav', '.txt'
    )

    # Step 3: Translate Dutch Text to English
    process_files(
        transcription_folder, translation_folder,
        lambda text_file: translate_text(open(text_file, 'r', encoding='utf-8').read(), target_lang="en"),
        '.txt', '_ENGLISH.txt'
    )
