import numpy as np
import random
import sys
import os
import keras
import json
import time

from pathlib import Path

from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, Dropout
from keras.callbacks import ModelCheckpoint, Callback
from keras.optimizers import Adam

filepath = "weights-best.hdf5"

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
    sentences.append(text[i:i + maxlen])
    next_chars.append(text[i + maxlen])

print('fragmenten gegenereerd: %s sentences' % len(sentences))

x = np.zeros((len(sentences), maxlen, len(chars)), dtype = np.bool)
y = np.zeros((len(sentences), len(chars)), dtype = np.bool)

for i, sentence in enumerate(sentences):
    for j, c in enumerate(sentence):
        x[i, j, char_to_int[c]] = 1
    y[i, char_to_int[next_chars[i]]] = 1


print('vectors gemaakt..')

model = Sequential()
model.add(LSTM(128, input_shape = (maxlen, len(chars)), return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(len(chars), activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = Adam(lr = 0.0005))

model.summary()

if Path(filepath).is_file():
    print('model geladen vanuit file ', filepath)
    model.load_weights(filepath)

checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=2, save_best_only=True, mode='min')

print('model compileerd..')

best_loss = 10000
start_time = time.time()

for iteration in (range(1, 600)):

    class NBatchLogger(Callback):
        def __init__(self,display=10):
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

    if iteration % 50 == 0:
        model.save('weights-iteration-' + str(iteration) + '.hdf5')

    print()
    print('-' * 50)
    print('Iteratie: ', iteration)
    history = model.fit(x, y, batch_size = 128, epochs = 1, callbacks = callbacks_list)
    loss = history.history['loss'][0]

    start_index = random.randint(0, len(sentences) - 1)
    seed = sentences[start_index]
    print("Seed:")
    print(seed)
    print()
    print("Output:")
    print()

    initial_seed = seed
    result_text = ''
    certainties = []

    for i in range(400):
        # seed omzetten in juiste format voor netwerk
        s = np.zeros((1, maxlen, len(chars)), dtype = np.bool)
        for i, c in enumerate(seed):
            s[0, i, char_to_int[c]] = 1

        prediction = model.predict(s, verbose = 0)
        index = np.argmax(prediction)
        certainties.append(np.amax(prediction))
        print(certainties)
        result = int_to_char[index]
        result_text += result
        sys.stdout.write(result)
        seed = seed[1:] + result

    f = open('resultaten.json', 'a')

    f.write(json.dumps({
        'seed': initial_seed, 
        'iteration': iteration,
        'loss': loss,
        'result_text': result_text,
        'model_improved': "true" if loss < best_loss else "false",
        'time': time.time() - start_time,
        'certainties': [str(x) for x in certainties]

    }))
    f.write("#@#") # marker om json blobs te splitten
    f.write("\n")
    f.close()

    start_time = time.time()
    if loss < best_loss:
        best_loss = loss
