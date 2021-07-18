import tensorflow as tf
import keras 

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, save_model
from keras.layers import Conv2D,MaxPooling2D,Dropout,Flatten,Dense
import numpy as np 

train_data_gen=ImageDataGenerator(rescale=1/255.,rotation_range=40,shear_range=0.2, zoom_range=0.2,horizontal_flip=True,fill_mode="nearest")

val_data_gen=ImageDataGenerator(rescale=1/255. )

train_data=train_data_gen.flow_from_directory(directory="The path to the train file is required",
                                              target_size=(224,224), color_mode="grayscale",
                                              class_mode="categorical",batch_size=32)


val_data=val_data_gen.flow_from_directory(directory="The path to the validation file is required",
                                          target_size=(224,224),color_mode="grayscale",
                                          class_mode="categorical",batch_size=32)

#SWISH
model=Sequential()
model.add(Conv2D(32,(3,3), input_shape=(224,224,1),activation="swish"))

model.add(Conv2D(64,(3,3),activation="swish"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(128,(3,3),activation="swish"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(128,(3,3),activation="swish"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(256,activation="swish"))
model.add(Dropout(0.5))
model.add(Dense(3,activation="softmax"))
model.summary()

model.compile(optimizer="rmsprop",loss="categorical_crossentropy",metrics=["accuracy"])
hist_swish=model.fit(train_data,batch_size=32,epochs=6,validation_data=val_data,steps_per_epoch=14630//32,validation_steps=1500//32)

#SIGMOID
model=Sequential()
model.add(Conv2D(32,(3,3), input_shape=(224,224,1),activation="sigmoid"))

model.add(Conv2D(64,(3,3),activation="sigmoid"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(128,(3,3),activation="sigmoid"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(128,(3,3),activation="sigmoid"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(256,activation="sigmoid"))
model.add(Dropout(0.5))
model.add(Dense(3,activation="softmax"))
model.summary()

model.compile(optimizer="rmsprop",loss="categorical_crossentropy",metrics=["accuracy"])
hist_sigmoid=model.fit(train_data,batch_size=32,epochs=6,validation_data=val_data,steps_per_epoch=14630//32,validation_steps=1500//32)

#TANH
model=Sequential()
model.add(Conv2D(32,(3,3), input_shape=(224,224,1),activation="tanh"))

model.add(Conv2D(64,(3,3),activation="tanh"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(128,(3,3),activation="tanh"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(128,(3,3),activation="tanh"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(256,activation="tanh"))
model.add(Dropout(0.5))
model.add(Dense(3,activation="softmax"))
model.summary()

model.compile(optimizer="rmsprop",loss="categorical_crossentropy",metrics=["accuracy"])
hist_tanh=model.fit(train_data,batch_size=32,epochs=6,validation_data=val_data,steps_per_epoch=14630//32,validation_steps=1500//32)

#RELU
model=Sequential()
model.add(Conv2D(32,(3,3), input_shape=(224,224,1),activation="relu"))

model.add(Conv2D(64,(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(128,(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(128,(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(256,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(3,activation="softmax"))
model.summary()

model.compile(optimizer="rmsprop",loss="categorical_crossentropy",metrics=["accuracy"])
hist_relu=model.fit(train_data,batch_size=32,epochs=6,validation_data=val_data,steps_per_epoch=14630//32,validation_steps=1500//32)

#LEAKY RELU
model=Sequential()
model.add(Conv2D(32,(3,3), input_shape=(224,224,1),activation="linear"))

model.add(Conv2D(64,(3,3),activation="linear"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(128,(3,3),activation="linear"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(128,(3,3),activation="linear"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(256,activation="linear"))
model.add(Dropout(0.5))
model.add(Dense(3,activation="softmax"))
model.summary()

model.compile(optimizer="rmsprop",loss="categorical_crossentropy",metrics=["accuracy"])
hist_leakyrelu=model.fit(train_data,batch_size=32,epochs=6,validation_data=val_data,steps_per_epoch=14630//32,validation_steps=1500//32)






















