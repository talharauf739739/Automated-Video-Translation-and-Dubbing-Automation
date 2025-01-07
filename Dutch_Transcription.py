import os
import speech_recognition as sr

def wav_to_text(wav_dir, txt_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(txt_dir):
        os.mkdir(txt_dir)
    
    # Initialize recognizer
    recognizer = sr.Recognizer()
    
    # Iterate over files in the wav directory
    for file in os.listdir(wav_dir):
        # Process only .wav files
        if file.endswith(".wav"):
            wav_path = os.path.join(wav_dir, file)
            
            # Open the .wav file for recognition
            with sr.AudioFile(wav_path) as source:
                audio_data = recognizer.record(source)
                
                try:
                    # Use Google Web Speech API for transcription
                    text = recognizer.recognize_google(audio_data)
                    
                    # Path for saving transcribed text file
                    txt_path = os.path.join(txt_dir, file[:-4] + ".txt")
                    
                    # Save the transcribed text to a file
                    with open(txt_path, "w") as text_file:
                        text_file.write(text)
                    
                    print(f"Status: Successfully transcribed {file} to {file[:-4]}.txt")
                
                except sr.UnknownValueError:
                    print(f"Error: Could not understand audio in {file}")
                except sr.RequestError as e:
                    print(f"Error: Could not request results from Google Speech Recognition service for {file}; {e}")
    
    return

if __name__ == "__main__":
    # Directory where your .wav files are stored
    wav_dir = r"/Users/talharauf/Desktop/Bots/Transcription/Dutch wav/"
    
    # Directory where you want to save the transcribed text files
    txt_dir = r"/Users/talharauf/Desktop/Bots/Transcription/Dutch Transcription/"
    
    wav_to_text(wav_dir, txt_dir)
