from api import apikey
from flask import Flask, request, jsonify
import pandas as pd
from csv import writer
import tempfile
import os
import time
from flask import Flask, send_file, after_this_request
import openai
import json
import jsonpickle
from langchain.document_loaders import PyPDFLoader
# from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from transformers import VitsModel, AutoTokenizer, pipeline
import torch
import soundfile as sf
from qna import text 
apikeys = apikey

app = Flask(__name__)


embeddings = OpenAIEmbeddings(openai_api_key=apikeys)
db = Chroma(persist_directory="mydb", embedding_function=embeddings)
# db.get()
# embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
# db = Chroma.from_documents(docs, embedding_function)
############################## 

#################################MODEL LOADING
from faster_whisper import WhisperModel
print('Loading Whisper...')
model_size = "large-v3"
model = WhisperModel(model_size, device="cuda", compute_type="int8")
print('Loading TTS...')
tts_model = VitsModel.from_pretrained("facebook/mms-tts-urd-script_arabic")
tts_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-urd-script_arabic")

print('Models Loaded Successfully...')


def transcribe(audio_file_path, beam_size=5):   
    print('transcribing...')
    segments, info = model.transcribe(audio_file_path, beam_size=beam_size,language='ur')
    transcribed_text = ""

    for segment in segments:
        transcribed_text += segment.text + " "

    return transcribed_text.strip()

def tts(text,filename):

    inputs = tts_tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        tts_output = tts_model(**inputs).waveform
    audio_data = tts_output.squeeze().cpu().numpy()
    sf.write(filename, audio_data, 16000)

    return  filename



def retrieve_combined_documents(query, max_combined_docs=4):
    retriever = db.as_retriever(search_type="mmr")

    rev_doc = retriever.get_relevant_documents(query)
    lim_rev_doc = rev_doc[:max_combined_docs]

    docs = db.similarity_search(query)
    lim_docs = docs[:max_combined_docs]

    combined_docs = str(lim_rev_doc) + str(lim_docs)
    # combined_docs=db.similarity_search(query)

    return combined_docs


############## GPT PROMPT ####################
def gpt(inp, prompt):
    systems = {"role":"system","content":"""YYou are an Assistant. Answer in detail in the language the question was asked such as if user ask question in urdu give him answer in urdu and if the Question is in Enlgish then answer him English and if in Roman Urdu then asnwer in him Roman Urdu, and so on dont try yo make up the amswer be concise as possibile."""}
    rcd = retrieve_combined_documents(prompt)
    print("rcd  hai ye &&&&&&& : ",rcd)
    systems1 = {"role":"system","content":str(rcd)}
    new_inp = inp
    new_inp.insert(0,systems)
    new_inp.insert(1,systems1)
    print("inp : \n ",new_inp)
    openai.api_key = apikeys
    completion = openai.ChatCompletion.create(
    model="gpt-4-1106-preview", 
    messages=new_inp
    )
    return completion

############    GET CHATS BY USER ID ##################
def get_chats(id):
    path = id
    isexist = os.path.exists(path)
    if isexist:
        data = pd.read_json(path)
        chats = data.chat
        return  list(chats)
    else:
        return "No Chat found on this User ID."





############### APPEND NEW CHAT TO USER ID JSON FILE #################
def write_chat(new_data, id):
    with open(id,'r+') as file:
          # First we load existing data into a dict.
        file_data = json.load(file)
        # Join new_data with file_data inside emp_details
        file_data["chat"].append(new_data)
        # Sets file's current position at offset.
        file.seek(0)
        # convert back to json.
        json.dump(file_data, file, indent = 4)



################################ CHECK IF USER IS ALREADY EXIST IF NOT CREATE ONE ELSE RETURN GPT REPLY ##################

def check_user(ids,prompt):
    

    print("asd")
    path = str(os.getcwd())+'//chats//'+ids+'.json'
    # path = str(os.getcwd())+'\\'+"5467484.json"
    isexist = os.path.exists(path)
    if isexist:
        # try:
        print(path," found!")
        write_chat({"role":"user","content":prompt},path)
        # print()
        chats = get_chats(path)
        print(chats)
        send = gpt(chats,prompt)
        reply = send.choices[0].message
        print("reply    ",reply.content)
        write_chat({"role":"assistant","content":reply.content},path)
        return {"message":reply,"status":"OK"}
        # except:
        #     return {"message":"something went wrong!","status":"404"}

    else:
        print(path," Not found!")
        dictionary = {
        "user_id":ids,
        "chat":[]


        }
        
        # Serializing json
        json_object = json.dumps(dictionary, indent=4)
        
        # Writing to sample.json
        with open(path, "w") as outfile:
            outfile.write(json_object)
        reply = check_user(ids,prompt)
        return reply
    

#################################handling chats
@app.route('/chat', methods=['POST'])
def chats():
    ids = request.json['user_id']
    prompt = request.json['prompt']
    reply = check_user(ids,prompt)
    return reply


UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)





def tensor_to_wav(tensor, sample_rate, output_path):
    audio_data = tensor.numpy()  # Convert to numpy array
    sf.write(output_path, audio_data, sample_rate)

@app.route('/audio', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'audio_file' not in request.files:
            return 'No file part'
        file = request.files['audio_file']
        name = file.filename
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            return 'No selected file'
        if file:
            # Save the file to the uploads directory
 
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], name))
            text = transcribe('uploads/'+name,beam_size=5)
            print(text)
            gpt_reply = check_user('test',text+"give very short answser")
            print('gpt_reply : ',gpt_reply)
            waveform = tts(gpt_reply['message']['content'],'audios/output.wav')  # Your tts function is called here with the input text
            # audio = open(waveform, 'rb')
            return send_file(waveform, as_attachment=True, download_name='downloaded_audio.wav')


            # return gpt_reply


####################   NEW ENPOINT GET CHAT ##############################
@app.route('/get_chats', methods=['POST'])
def get_chatss():
    ids = request.json['user_id']
    return jsonpickle.encode(get_chats(ids))





if __name__ == '__main__':
    app.run()
    
