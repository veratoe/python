import numpy as np
import random
import sys
import os
import keras
import json
import time

from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, Dropout
from keras.callbacks import ModelCheckpoint, Callback

os.system('clear')
print('We gaan beginnen')
text = open('nietzsche.txt').read().lower()
print('text gelezen uit bestand..')
chars = sorted(list(set(text)))

int_to_char = dict((k, v) for k, v in enumerate(chars))
char_to_int = dict((v, k) for k, v in enumerate(chars))

maxlen = 40
step = 3
sentences = []
next_chars = []

for i in range(0, len(text) - maxlen, step):
    sentences.append([char_to_int[c] for c in text[i:i + maxlen]])
    next_chars.append(char_to_int[text[i + maxlen]])

print('fragmenten gegenereerd: %s sentences' % len(sentences))

x = np.reshape(sentences, (len(sentences), maxlen, 1))
print('Normalizeren..')
x = x / float(len(chars))
y = keras.utils.to_categorical(next_chars);

print('vectors gemaakt..')

model = Sequential()
model.add(LSTM(256, input_shape = (maxlen, 1), return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(256, input_shape = (maxlen, 1)))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')
model.load_weights('weights-1.4465.hdf5')
filepath="weights-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=2, save_best_only=True, mode='min')

print('model geladen en gecompileerd..')

last_loss = 0
start_time = time.time()

for iteration in (range(1, 600)):

    class NBatchLogger(Callback):
        def __init__(self,display=10):
            '''
            display: Number of batches to wait before outputting loss
            '''
            self.seen = 0
            self.display = display
            self.time = time.time()
            self.loss = 0
            self.wub = 0

        def on_batch_end(self,batch,logs={}):
            self.seen += logs.get('size', 0)
            if self.seen % self.display == 0:
                    
                self.loss += logs.get('loss')
                self.wub += 1
                d = dict(logs.items())
                pf = open('status.json', 'w')
                pf.write(json.dumps({
                    'batch': str(d['batch']),
                    'total_batches': len(sentences) / 128,
                    'loss': str(self.loss / self.wub),
                    'iteration': iteration,
                    'time': time.time() - self.time
                }))
                pf.close()

    diede = NBatchLogger(display=10)
    callbacks_list = [checkpoint, diede]

    print()
    print('-' * 50)
    print('Iteratie: ', iteration)
    history = model.fit(x, y, batch_size = 128, epochs = 1, callbacks = callbacks_list)
    loss = history.history['loss'][0]

    start_index = random.randint(0, len(sentences) - 1)
    seed = sentences[start_index]
    print("Seed:")
    print(''.join([int_to_char[c] for c in seed]))
    print()
    print("Output:")
    print()

    initial_seed = seed
    result_text = ''

    for i in range(400):
        s = np.reshape(seed, (1, len(seed), 1))
        s = s / float(len(chars))
        prediction = model.predict(s, verbose = 0)
        index = np.argmax(prediction)
        result = int_to_char[index]
        result_text += result
        sys.stdout.write(result)
        seed.append(index)
        seed = seed[1:len(seed)]

    f = open('resultaten.json', 'a')
    f.write(json.dumps({
        'seed': ''.join([int_to_char[c] for c in initial_seed]), 
        'iteration': iteration,
        'loss': loss,
        'result_text': result_text,
        'model_improved': "true" if loss < last_loss else "false",
        'time': time.time() - start_time

    }))
    f.write("#@#") # marker om json blobs te splitten
    f.write("\n")
    f.close()

    start_time = time.time()
    last_loss = loss
