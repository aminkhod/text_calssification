from flask import Flask, render_template, request
import nltk
import numpy as np


import spacy

import pandas as pd 
from nltk.corpus import stopwords



from time import time

# import warnings
# warnings.filterwarnings("ignore")

from nltk.stem import WordNetLemmatizer


import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Embedding, Layer, Dense, Dropout, MultiHeadAttention, LayerNormalization, Input, GlobalAveragePooling1D
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

app = Flask(__name__)
def process_text(text):
    # Replace email addresses with 'email'
    processed = text.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$', 'emailaddress', regex=True)

    # Replace URLs with 'webaddress'
    processed = processed.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$',
                                    'webaddress', regex=True)

    # Replace money symbols with 'moneysymb' (£ can by typed with ALT key + 156)
    processed = processed.str.replace(r'£|\$', 'moneysymb', regex=True)
        
    # Replace 10 digit phone numbers (formats include paranthesis, spaces, no spaces, dashes) with 'phonenumber'
    processed = processed.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$',
                                    'phonenumbr', regex=True)
        
    # Replace numbers with 'numbr'
    processed = processed.str.replace(r'\d+(\.\d+)?', 'numbr', regex=True)

    # Removeing useless characters like whitespace, punctuation and so on
    processed = processed.str.replace(r'[^\w\d\s]', ' ', regex=True)

    # Replace whitespace between terms with a single space
    processed = processed.str.replace(r'\s+', ' ', regex=True)

    # Remove leading and trailing whitespace
    processed = processed.str.replace(r'^\s+|\s+?$', '', regex=True)
    processed = processed.str.lower()
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    processed = processed.apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))
    ps = nltk.PorterStemmer()
    processed = processed.apply(lambda x: ' '.join(ps.stem(term) for term in x.split()))

    return processed
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, heads, neurons):
        super(TransformerEncoder, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=heads, key_dim=embed_dim)
        self.ffn = Sequential(
            [layers.Dense(neurons, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(0.5)
        self.dropout2 = layers.Dropout(0.5)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
    
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the input string from the textbox
        input_string = request.form['input_string']
        # data = pd.DataFrame(input_string, columns= 'Synopsis')
        print(pd.DataFrame([input_string]))
        X_test = process_text(pd.Series([input_string]))
        max_len = 2000     
        oov_token = '00_V' 
        padding_type = 'post'
        trunc_type = 'post'  

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(X_train)
        vocab_size = len(tokenizer.word_index) + 1
        print("Vocab Size: ",vocab_size)
        embed_dim = 20 
        heads = 2  
        neurons = 10
        max_len = 2000
        # vocab_size = vocab_size
        vocab_size =26070

        inputs = layers.Input(shape=(max_len,))
        embedding_layer = TokenAndPositionEmbedding(max_len, vocab_size, embed_dim)
        x = embedding_layer(inputs)
        transformer_block = TransformerEncoder(embed_dim, heads, neurons)
        x = transformer_block(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = Dropout(0.2)(x)
        outputs = layers.Dense(3, activation="sigmoid")(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()
        model.load_weights('../Model weights/transformer_weights.h5')
        y_pred = model.predict(X_test)
        cult = [int(x) for x in y_pred[:,0]>.6]

        print(sum(y_pred[:,0]>.6))
        y_pred1 = y_pred.copy()
        y_pred1[:,0] = [int(x) for x in y_pred[:,0]>.6]

        print(sum(y_pred[:,1]>.26))
        y_pred1[:,1] = [int(x) for x in y_pred[:,1]>.15]

        print(sum(y_pred[:,2]>.33))
        y_pred1[:,2] = [int(x) for x in y_pred[:,2]>.23]

        y_pred2 = [1 if x[1]==1 else 2 if x[2]==1 else 0 for x in y_pred1]
        x = ''
        if y_pred2[0] == 0:
            x = 'cult'
        elif y_pred2[0] ==1:
            x='paranormal'

        else:
            x= 'dramatic'
        # Process the input list (here, we'll just join the elements with '-')
        processed_string = '-'.join(x)
        
        # Render the template with the processed string
        return render_template('index.html', processed_string=processed_string)
    else:
        # If it's a GET request, render the template without any processed string
        return render_template('index.html', processed_string=None)

if __name__ == '__main__':
    app.run(debug=True)
