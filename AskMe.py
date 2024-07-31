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

# NLP Question Generation Function
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

def generate_question(user_input):
    tokens = tokenizer.tokenize(user_input)
    if len(tokens) > 900:
        tokens = tokens[-900:]
        user_input = tokenizer.convert_tokens_to_string(tokens)

    prompt = f"Generate a nice question to ask someone who said: {user_input}"
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(input_ids, max_length=50, num_return_sequences=1)
    question = tokenizer.decode(output[0], skip_special_tokens=True)
    return question

def main():
    audio_file_path = "/Users/shanyu/Desktop/testing1.mp3"
    text = recognize_speech(audio_file_path)
    print("Trasnscibed Text: " + text)

    # Generate a question based on the transcribed text
    question = generate_question(text)
    print(f"Generated Question: {question}")

if __name__ == "__main__":
    main()