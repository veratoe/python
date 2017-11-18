from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing import image
import numpy


# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.load_weights('cnn.h5')

print('loaded model')

test_image = image.load_img('Dogge_Odin.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
#06-train-cat-shake-hands.jpg
#75552.ngsversion.1422285553360.adapt.1900.1.jpg
#audi-sq7.jpg
#cat-pet-animal-domestic-104827.jpeg
#cnn.h5
#cnn.py
#cnn_test.py
#Convolutional_Neural_Networks
#Convolutional_Neural_Networks.zip
#dataset
#file_23012_beagle.jpg
#gold-fish-250x250.jpg
#hondje.jpg
#istock-526142422.jpg
#rMyStss.jpg
test_image = numpy.expand_dims(test_image, axis = 0)
p = classifier.predict(test_image, batch_size = 1)



if p[0][0] == 1:
    print('het is een katje')
else:
    print('het is een katje')
# Part 2 - Fitting the CNN to the images

#from keras.preprocessing.image import ImageDataGenerator
#
#train_datagen = ImageDataGenerator(rescale = 1./255,
#                                   shear_range = 0.2,
#                                   zoom_range = 0.2,
#                                   horizontal_flip = True)
#
#test_datagen = ImageDataGenerator(rescale = 1./255)
#
#training_set = train_datagen.flow_from_directory('dataset/training_set',
#                                                 target_size = (64, 64),
#                                                 batch_size = 32,
#                                                 class_mode = 'binary')
#
#test_set = test_datagen.flow_from_directory('dataset/test_set',
#                                            target_size = (64, 64),
#                                            batch_size = 32,
#                                            class_mode = 'binary')
#
#classifier.fit_generator(training_set,
#                         steps_per_epoch = 800,
#                         epochs = 20,
#                         validation_data = test_set,
#                         validation_steps = 200)
#
#classifier.save('cnn.h5')
