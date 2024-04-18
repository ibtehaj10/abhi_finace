import telebot
from pydub import AudioSegment
import requests
import os
import json

TOKEN = '6860861824:AAFM0Ox2SJFUYOeT3OtkxnsXhMGMD8a7Y3E'
bot = telebot.TeleBot(TOKEN)

# Replace these with your actual API endpoints
TEXT_API_ENDPOINT = "http://localhost:5000/chat"
AUDIO_API_ENDPOINT = "http://localhost:5000/audio"

@bot.message_handler(content_types=['text'])
def handle_text_messages(message):
    # Send received text to the API
    id = message.chat.id
    response = requests.post(TEXT_API_ENDPOINT, json={
        "prompt": message.text,
        'user_id':str(id)
        }
        )
    reply = response.json()
    print(reply)
    reply = reply['message']
    # Check if the request was successful
    if response.status_code == 200:
        # Send the response back to the user
        bot.send_message(message.chat.id, reply)
    else:
        bot.send_message(message.chat.id, "Sorry, there was an error processing your request.")




@bot.message_handler(content_types=['voice'])
def handle_voice_messages(message):
    try:
        file_info = bot.get_file(message.voice.file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        user_id = message.chat.id
        
        # Paths for the original and converted files
        ogg_file_path = 'voice_note.ogg'
        wav_file_path = 'voice_note.wav'
        
        # Save the voice note locally in .ogg format
        with open(ogg_file_path, 'wb') as new_file:
            new_file.write(downloaded_file)
        
        # Convert to WAV format
        AudioSegment.from_file(ogg_file_path).export(wav_file_path, format="wav")
        
        # Prepare files and data for the POST request
        files = {
            'audio_file': (wav_file_path, open(wav_file_path, 'rb'), 'audio/wav'),
        }
        data = {'user_id': json.dumps(user_id)}
        
        # Sending POST request to the Flask server with the audio file and user_id
        response = requests.post(AUDIO_API_ENDPOINT, files=files, data=data)
        
        if response.status_code == 200:
            # Assuming the API returns a direct audio file, save and send it back to the user
            response_audio_path = 'response_audio.wav'
            with open(response_audio_path, 'wb') as out_file:
                out_file.write(response.content)
            
            with open(response_audio_path, 'rb') as audio_response:
                bot.send_audio(message.chat.id, audio_response)
        else:
            bot.send_message(message.chat.id, "Sorry, there was an error processing your audio.")
    except Exception as e:
        bot.send_message(message.chat.id, f"An error occurred: {str(e)}")
    # finally:
        # Cleanup temporary files
        # cleanup_files(ogg_file_path, wav_file_path, 'response_audio.wav')

def cleanup_files():
    if os.path.exists("voice_note.ogg"):
        os.remove("voice_note.ogg")
    if os.path.exists("voice_note.wav"):
        os.remove("voice_note.wav")
    if os.path.exists("response_audio.ogg"):
        os.remove("response_audio.ogg")

    # You might want to call this function after handling each voice message or periodically

bot.polling()
