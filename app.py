from flask import Flask, render_template, request, jsonify
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np
from sklearn.preprocessing import LabelEncoder
import emoji
import pandas as pd
import pickle
from flask_cors import CORS
train = pd.read_csv('training.csv', header=None)
X_train = train.values[:, 0]
app = Flask(__name__)
CORS(app)
app.config['JSON_AS_ASCII'] = False
# Load the trained model
model = load_model('matched.h5')

# Load label encoder classes
label_encoder_classes = np.load('label_encoder_classes.npy', allow_pickle=True)

# Initialize label encoder and fit it with classes
label_encoder = LabelEncoder()
label_encoder.classes_ = label_encoder_classes

# Initialize a Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

# Load pre-trained embeddings
embeddings_index = {}
with open('./glove.6B.50d.txt', encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# Map emoji indices to names
emoji_dict = {'heart': "\u2764\uFE0F", 'baseball': ":baseball:", 'smile': ":beaming_face_with_smiling_eyes:",
              'sad': ":downcast_face_with_sweat:", 'food': ":fork_and_knife:"}
        
"""emoji_dict = {'heart': "\u2764\uFE0F", 'baseball': "&#x26be;", 'smile': ":beaming_face_with_smiling_eyes:",
              'sad': ":downcast_face_with_sweat:", 'food': ":fork_and_knife:"}"""

# Map label indices to names
#emoji_names = ["Love", "Sports", "Happy", "Sad", "Food"]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/emojify', methods=['POST'])
def emojify():
    try:
        data = request.json
        text = data.get('text')
        # Preprocess the input text
        #print("Original Text:", text)
        sequence = tokenizer.texts_to_sequences([text])
        #print("Tokenized Sequence:", sequence)
        padded_sequence = pad_sequences(sequence, maxlen=10)
        #print("Padded Sequence:", padded_sequence)
        reshaped_input = np.zeros((1, 10, 50))

        for i in range(min(10, len(padded_sequence[0]))):
            word_index = padded_sequence[0][i]
            if word_index != 0:
                word = tokenizer.index_word[word_index]
                reshaped_input[0][i] = embeddings_index.get(word, np.zeros((50,)))

        # Make predictions
        print("Reshaped Input:", reshaped_input)
        predictions = model.predict(reshaped_input)
        print("Prediction array:", predictions)
        predicted_label = label_encoder.inverse_transform(np.argmax(predictions, axis=1))[0]
        # Convert predictions to emoji name
        emoji_index = np.argmax(predictions)
        predicted_emoji = emoji.emojize(emoji_dict[predicted_label])
        print("Predicted Label:", predicted_label)
        print("Predicted Emoji:", predicted_emoji)
        return jsonify({"emoji_name": predicted_emoji}), 200



    except Exception as e:
        print(str(e))
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5500)
