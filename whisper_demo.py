import whisper

def transcribe_audio(file_path):
    # Load the Whisper model
    model = whisper.load_model("base")

    # Transcribe the audio file
    result = model.transcribe(file_path)

    # Print the transcription
    print("Transcription:")
    print(result['text'])

if __name__ == "__main__":
    # Replace 'your_audio_file.mp3' with the path to your mp3 file
    audio_file = "shinki77.m4a"  
    transcribe_audio(audio_file)