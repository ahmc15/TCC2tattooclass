import keras

import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.applications import InceptionV3
from keras.layers import Conv2D, MaxPooling2D,ZeroPadding2D
from keras.layers import Flatten, Dense, Dropout


#preprocessing




##model

model = Sequential()
keras.applications.InceptionV3(include_top=False, weights='imagenet',input_shape=(224,224,3) , pooling='avg')

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(1, 1)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(1, 1)))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(Y_train, activation='softmax'))
model.summary()

#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

#model.fit(X_train, Y_train, epochs=2, batch_size=32)



