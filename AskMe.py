import os
import speech_recognition as sr
from pydub import AudioSegment
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def recognize_speech(audio_file_path):
    recognizer = sr.Recognizer()
    audio = AudioSegment.from_file(audio_file_path)
    audio.export("temp.wav", format="wav")
    with sr.AudioFile("temp.wav") as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            text = "Sorry, I did not understand that."
        except sr.RequestError:
            text = "Sorry, the service is unavailable."
    os.remove("temp.wav")
    return text

def main():
    audio_file_path = "/Users/shanyu/Desktop/testing1.mp3"
    text = recognize_speech(audio_file_path)
    print("Trasnscibed Text: " + text)

if __name__ == "__main__":
    main()