import pyttsx3
import speech_recognition as sr
from transformers import BlenderbotSmallTokenizer, BlenderbotSmallForConditionalGeneration
import torch
import datetime

your_name = "put your nick name here"
bot_name = "put you chat bots name here"

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Initialize the speech recognizer
r = sr.Recognizer()

def speak(text):
    """Speak the given text."""
    engine.say(text)
    engine.runAndWait()

def initialize_chatbot():
    """Initialize the chatbot model and tokenizer."""
    tokenizer = BlenderbotSmallTokenizer.from_pretrained("facebook/blenderbot_small-90M")
    model = BlenderbotSmallForConditionalGeneration.from_pretrained("facebook/blenderbot_small-90M")
    return tokenizer, model

def get_timestamp():
    """Get the current timestamp."""
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

def main():
    tokenizer, model = initialize_chatbot()
    
    print("Chatbot initialized. Say " + bot_name +" to start a chat or 'Exit' to exit.")
    chat_active = False
    conversation_history = []

    while True:
        if not chat_active:
            print("Listening...")
            with sr.Microphone() as source:
                r.adjust_for_ambient_noise(source)
                audio = r.listen(source)
                try:
                    user_input = r.recognize_google(audio).lower()
                    print(f"{get_timestamp()} You said:", user_input)  # Add timestamp here
                    if bot_name in user_input:
                        speak("Hello! " + your_name + " How can I help you?")
                        chat_active = True
                        conversation_history = []
                    if "exit" in user_input:
                        speak("Goodbye for now.")
                        break
                except sr.UnknownValueError:
                    pass  # Ignore unrecognizable audio
                except sr.RequestError as e:
                    print(f"Error during audio recognition: {str(e)}")

        elif chat_active:
            print(">> User:", end=' ')
            with sr.Microphone() as source:
                audio = r.listen(source)
                try:
                    user_input = r.recognize_google(audio).lower()
                    print(f"{get_timestamp()} {user_input}")  # Add timestamp here
                    if any(keyword in user_input for keyword in ["exit", "quit", "bye"]):
                        speak("Goodbye, talk to you later.")
                        chat_active = False
                    else:
                        # Use the BlenderBot model to generate a response
                        input_ids = tokenizer(user_input, return_tensors="pt").input_ids
                        bot_response_ids = model.generate(input_ids, max_length=100, num_return_sequences=1)
                        bot_response = tokenizer.decode(bot_response_ids[0], skip_special_tokens=True)

                        print(f"{get_timestamp()} Bot:", bot_response)  # Add timestamp here
                        speak(bot_response)

                        # Append user input and bot response to the conversation history
                        conversation_history.append(user_input)
                        conversation_history.append(bot_response)

                except sr.UnknownValueError:
                    pass  # Ignore unrecognizable audio
                except sr.RequestError as e:
                    print(f"Error during audio recognition: {str(e)}")

if __name__ == "__main__":
    main()
