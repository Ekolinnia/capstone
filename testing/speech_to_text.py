#pip install sounddevice - Micorphone du PC
#pip install vosk - convert input mic to text 
#pip install queue
#https://stackoverflow.com/questions/79253154/use-vosk-speech-recognition-with-python


import pickle
import vosk
import sounddevice as sd
import json
#store the transcript
import queue

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


#print out available device to speak on 
print(sd.query_devices())

#Load the tokenizer and the trained model
#read in byte 
with open("testing/tokenizer.pickle", "rb") as handling:
    commmand_tokenizer = pickle.load(handling)
    
tensor_model = tf.keras.models.load_model("testing/command_classification_model.h5")



# Initialize Vosk model dowloaded in 
try:
    model = vosk.Model("vosk-model-small-en-us-0.15") 
    print("Model has loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)
# Initilaize regonization model 
recognizer = vosk.KaldiRecognizer(model, 16000)

#Queue to store all transcription 
trasncript_queue = queue.Queue()


#Process captured input
def audio_callback(indata, frames, time, status):
    if status:
        print(f"Stream status error: {status}")
        return
    try:
        # Convert indata to bytes
        if recognizer.AcceptWaveform(bytes(indata)):
            result = json.loads(recognizer.Result())
            #Get the transcription 
            transcription = result.get('text', '')
            print(f"Transcription: {transcription}")
            if transcription:
                #add transcription to queue
                trasncript_queue.put(transcription)
    except Exception as e:
        print(f"Error in callback: {e}")


#transcribe captured input into text
def process_transcript():
    print("Listening... Speak into macbook microphone!")
    try:
        with sd.RawInputStream(device=0,samplerate=16000, blocksize=1024, dtype='int16',
                               channels=1, callback=audio_callback):
            print("Press Ctrl+C to stop.")
            #open a file 
            with open("transcriptions.txt", "a") as file:  
                while True:
                    #retrieve from the queue
                    transcription = trasncript_queue.get()
                    if transcription:
                        #process the transcript
                        #test
                        #sample_command = ["Take me to the nearest coffee shop"]
                        print("This is test : ")
                        print(transcription)
                        # convert trasncription as a single string
                        new_sequence = commmand_tokenizer.texts_to_sequences([transcription])
                        new_padded_sequences = pad_sequences(new_sequence,maxlen=10, padding="post")
                        
                        #predict the command with th emodel
                        prediction = tensor_model.predict(new_padded_sequences)
                        #take the one with the highest predcition %
                        predicted_label = tf.argmax(prediction[0]).numpy()
                        
                        print(f"Predicted command: {predicted_label} " )
                        
                        
                        #save transcription in a file
                        file.write(transcription + "\n")
    except KeyboardInterrupt:
        print("Stopped listening.")
        
        

    

# Start transcription
process_transcript()





