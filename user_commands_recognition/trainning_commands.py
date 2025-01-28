import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
#save token 
import pickle

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

# Training model
import tensorflow as tf
from tensorflow.keras import layers

# Get the datasheet for activation
data_train = pd.read_csv("user_commands_recognition/activation.csv", header= 0)

# Seperate the activation quotes and label for trainning
commands = data_train["command"].tolist()
labels = data_train["label"].tolist()

# Convert the text commands into array of numbers with token
# https://docs.python.org/3/library/tokenize.html 
#Create the token 
commands_token = Tokenizer(num_words = 1000,oov_token="<OOV>")
# Create Token
commands_token.fit_on_texts(commands)

#Convert the text with the token
s_commands = commands_token.texts_to_sequences(commands)
p_s_commands = pad_sequences(s_commands, maxlen = 10, padding = "post")

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
#10 epochs
model.fit(p_s_commands, np.array(labels), epochs=10, batch_size=2)

#Save the trained model
model.save("user_commands_recognition/activation_classification_model.h5")
print("saved model")

#save the token 
with open("user_commands_recognition/tokenizer_activation.pickle", "wb") as handle:
    pickle.dump(commands_token, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Tokenizer saved")

# Testing the model

#activation command and converting it into padded sequence
new_command = ["What's in front of me"]
s_new_command = commands_token.texts_to_sequences(new_command)
p_s_new_command = pad_sequences(s_new_command, maxlen = 10, padding = 'post')

#prediction
prediction = model.predict(p_s_new_command)
predicted_label = tf.argmax(prediction[0]).numpy()
highest_probability = prediction[0][predicted_label] * 100 


print("\nPrediction:")
print(f"Command: {new_command[0]}")
print(f"Predicted Label: {predicted_label}")
print(f"With a percentage of: {highest_probability} ")




