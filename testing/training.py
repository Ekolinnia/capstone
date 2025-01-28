import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
#save token 
import pickle

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers

# Get datasheet (headers on the datasheet)
data_train = pd.read_csv("commands1.csv", header=0)
print(data_train.head())

# Seperate Commands and Label for training
commands = data_train["Command"].tolist()
labels = data_train["Label"].tolist()

#Convert the data (text command) into numbers (token) so that machine can understand it
#https://docs.python.org/3/library/tokenize.html 
#1.Create a Token dictionary that will fit with datasheet
commands_token = Tokenizer(num_words = 1000,oov_token="<OOV>")
commands_token.fit_on_texts(commands)

#2.Converting the text into the sequence with the created token
commands_sequences = commands_token.texts_to_sequences(commands)

#3.Padding them so they have the same length of 10
commands__padded_sequences = pad_sequences(commands_sequences,maxlen=10, padding="post")

#printing the badded sequeces of command
print( " \n Padded sequences: ")
print(commands__padded_sequences)

#NN model for text 
model = tf.keras.Sequential([
    #Text Embedding input layer (input 10, cuz command sequence is 10)
    tf.keras.layers.Embedding(1000,16,input_length=10),
    #Reduce Dimensions
    tf.keras.layers.GlobalAveragePooling1D(),
    #hidden layer
    tf.keras.layers.Dense(16,activation = "relu"),
    #ouput layer
    tf.keras.layers.Dense(len(set(labels)), activation="softmax")
])

#Opdimize model
model.compile(optimizer= "adam", loss="sparse_categorical_crossentropy",metrics=["accuracy"])

#Training the model
print("\nTraining the model...")
#30 epochs
model.fit(commands__padded_sequences, np.array(labels), epochs=10, batch_size=2)

#Save the trained model
model.save("testing/command_classification_model.h5")
print("saved model")

#save the token 
with open("testing/tokenizer.pickle", "wb") as handle:
    pickle.dump(commands_token, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Tokenizer saved to 'tokenizer.pickle'")

#test the model with a command

#command given
new_command = [" drive me to hospital"]
#convert the command into the token
new_sequence = commands_token.texts_to_sequences(new_command)
#padd the sequece
new_padded = pad_sequences(new_sequence, maxlen=10, padding="post")


# Predict
prediction = model.predict(new_padded)
predicted_label = tf.argmax(prediction[0]).numpy()
highest_probability = prediction[0][predicted_label] * 100 

print("\nPrediction:")
print(f"Command: {new_command[0]}")
print(f"Predicted Label: {predicted_label}")
print(f"With a percentage of: {highest_probability}")