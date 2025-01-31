#pip install sounddevice - Micorphone du PC
#pip install vosk - convert input mic to text 
#pip install queue
#pip install pyttsx3
#https://stackoverflow.com/questions/79253154/use-vosk-speech-recognition-with-python
#/opt/miniconda3/envs/pytorch/bin/python -m pip install <package-name>

#pi@raspberrypi:~ $ cd ~/Documents/capstone-1 
#pi@raspberrypi:~/Documents/capstone-1 $ python3 -m venv myenv
#pi@raspberrypi:~/Documents/capstone-1 $ source myenv/bin/activate
#(myenv) pi@raspberrypi:~/Documents/capstone-1 $ pip install vosk

import pickle
import vosk
import sounddevice as sd
import json
#store the transcript
import queue
import pandas as pd
import time

#model
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#text to speech
import pyttsx3

#Setting up the text to speech
#/opt/miniconda3/envs/pytorch/bin/python -m pip install <package-name>
engine = pyttsx3.init()



#print out available device to speak on 
print(sd.query_devices())

#Load the tokenizer and the trained model
#read in byte 
with open("user_commands_recognition/tokenizer_activation.pickle", "rb") as handling:
    commmand_tokenizer = pickle.load(handling)
    
tensor_model = tf.keras.models.load_model("user_commands_recognition/activation_classification_model.h5")

# Initialize Vosk model dowloaded in 
try:
    model = vosk.Model("vosk-model-small-en-us-0.15") 
    print("Model has loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)
# Initilaize regonization model 
recognizer = vosk.KaldiRecognizer(model, 48000)

#Queue to store all transcription 
trasncript_queue = queue.Queue()

#function to read transcription and compare with the location file
def location_detector(transcription,location):
    #iterate through the rows of the file
    for _,row in location.iterrows():
        if row["places"].lower() in transcription.lower().strip():
            return row["coordinatesx"], row["coordinatesy"]
    return None

    
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

#Save the coordinates of desired location
coordinatesx = 0
coordinatesy = 0

#acquire the location file 
locations = pd.read_csv("user_commands_recognition/places.csv")



#transcribe captured input into text
def process_transcript():
    print("Listening... Speak into macbook microphone!")
    try:
        with sd.RawInputStream(device=0,samplerate=48000, blocksize=1024, dtype='int16',
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
                        highest_probability = prediction[0][predicted_label] * 100 
                        
                        #Start processing command if hear "hey rebecca"
                        if transcription.lower().strip() == "hey rebecca":
                            print("Hi! How can I help you? ")
                            engine.say("Hi! How can I help you? ")
                            engine.runAndWait()
                            time.sleep(6)
                            
                            while True:
                            # Wait for the next command 
                                transcription = trasncript_queue.get()
                                if transcription:
                                
                                    # Check if the user says "cancel"
                                    if transcription.lower().strip() == "cancel":
                                        print("Cancelled. Waiting for 'hey Rebecca' again...")
                                        engine.say("Cancelled. Waiting for 'hey Rebecca' again...")
                                        engine.runAndWait()
                                        break 
                                
                                    # convert trasncription as a single string
                                    new_sequence = commmand_tokenizer.texts_to_sequences([transcription])
                                    new_padded_sequences = pad_sequences(new_sequence,maxlen=10, padding="post")
                    
                                    #predict the command with th emodel
                                    prediction = tensor_model.predict(new_padded_sequences)
                                    #take the one with the highest predcition %
                                    predicted_label = tf.argmax(prediction[0]).numpy()
                                    highest_probability = prediction[0][predicted_label] * 100   
                                      
                        
                                    if highest_probability >= 40:
                                        print(f"Predicted command: {predicted_label} " )
                                        #Try to detect which location it is...
                                        #For now, set like 5 locations for the GPS....(which is saved in places.csv)
                                        coordinates = location_detector(transcription,locations)
                            
                                        if(coordinates):
                                            coordinatesx,coordinatesy = coordinates
                                            print(f"User has requested to go to {coordinatesx}, {coordinatesy}")
                                            engine.say("User has requested to go to {coordinatesx}, {coordinatesy}")
                                            engine.runAndWait()
                                            #go back to lsitening for hey walking
                                            break
                                        elif(predicted_label == 2):
                                            print("User requested to see whats in front")
                                            engine.say("User requested to see whats in front")
                                            engine.runAndWait()
                                            
                                            break
                            
                                        else:
                                            count = 1
                                            if count ==1: 
                                                print(" No Location found/not recognize try again?... or say cancel")
                                                continue
                                                    
                                    else:
                                        print(f"Please repeat....")  
                        
                                #save transcription in a file
                                file.write(transcription + "\n")
                        else: 
                            print("Listening for hey Rebecca")
    except KeyboardInterrupt:
        print("Stopped listening.")
        
        

    

# Start transcription
process_transcript()
