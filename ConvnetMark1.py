import tensorflow
import keras
#import parser import load_data
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D,ZeroPadding2D
from keras.layers import Flatten, Dense, Dropout

img_width,img_height = 256, 256
train_data_dir = 'data/train'
validation_data_dir='data/validation'
train_samples = 1144
validation_samples = 122
epoch = 4
num_classes=2


#Collect data
#training_data = load_data('data/training')
#validation_data = load_data ('data/validation')


# Model Structure
model = Sequential()

model.add(Convolution2D(64, 3, 3, input_shape=(img_width, img_height, 3), activation='relu', name='conv1_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, 3, 3,  activation='relu', name='conv1_3'))
model.add(ZeroPadding2D((1, 1)))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_3'))
model.add(ZeroPadding2D((1, 1)))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
model.add(ZeroPadding2D((1, 1)))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))




model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.summary()



#  compile
model.compile(optimizer='rmsprop',loss='binary_crossentropy', metrics=['accuracy'] )
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=(img_width, img_height),
                                                    batch_size=32,
                                                    )
validation_generator = test_datagen.flow_from_directory(validation_data_dir,
                                                        target_size=(img_width, img_height),
                                                        batch_size=32,
                                                        )
model.fit_generator(train_generator,
                    )

model.save_weights('covnetmark1.h5')




