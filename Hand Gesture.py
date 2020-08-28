#!/usr/bin/env python
# coding: utf-8

# ![alt](./images/gestures.png)

# In[1]:


import numpy as np
import os
import matplotlib.pyplot as plt
import cv2


# In[2]:




def signal_handler(signal, frame):
    # KeyboardInterrupt detected, exiting
    global is_interrupted
    is_interrupted = True


# In[3]:


def make_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


# In[4]:


def put_text(frame,text,position):
     cv2.putText(frame, text, position, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)


# In[5]:


path = "./handgestures/"


def record_gesture(frame, gesture_number, image_count, roi, is_train=True):
    current_type = "train" if is_train else "test"
    put_text(
        frame,
        "Recording " + str(gesture_number) + " gesture for - " + current_type,
        (100, 100))
    put_text(frame, str(image_count), (400, 400))
    current_path = path + current_type + "/" + str(gesture_number) + "/"
    make_directory(current_path)
    cv2.imwrite(current_path + str(image_count) + ".jpg", roi)


# In[6]:


label_to_symbol_map = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E',
    5: 'F',
    6: 'G',
    7: 'H',
    8: 'I',
    9: 'K',
    10: 'L',
    11: 'M',
    12: 'N',
    13: 'O',
    14: 'P',
    15: 'Q',
    16: 'R',
    17: 'S',
    18: 'T',
    19: 'U',
    20: 'V',
    21: 'W',
    22: 'X',
    23: 'Y'
}


def map_label_to_symbol(label):
    if label in label_to_symbol_map:
        return label_to_symbol_map[label]
    else:
        return "Did not get"


# In[7]:


def generate_data(number_of_gestures=2):
    cap = cv2.VideoCapture(0)
    try:
        image_count = 0
        gesture_count = 0
        gesture_recording_started = False
        is_train = True
        while gesture_count < number_of_gestures:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            roi = frame[100:400, 320:620]
            cv2.imshow('roi', roi)
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)

            cv2.imshow('roi scaled and gray', roi)
            copy = frame.copy()
            cv2.rectangle(copy, (320, 100), (620, 400), (255, 0, 0), 5)

            if not gesture_recording_started:
                image_count = 0
                put_text(copy, "Hit Enter to Record new gesture", (100, 100))
            else:
                image_count += 1
                put_text(copy, "Hit Enter to change", (100, 50))
                record_gesture(copy, gesture_count, image_count, roi, is_train)

            cv2.imshow('frame', copy)

            if cv2.waitKey(1) == 13:
                image_count = 0
                if gesture_recording_started:
                    is_train = not is_train
                    if is_train:
                        gesture_recording_started = False
                        print("changing gesture", str(gesture_count))
                        gesture_count += 1
                else:
                    gesture_recording_started = True
                    is_train = True
    finally:
        cap.release()
        cv2.destroyAllWindows()


# In[8]:


#generate_data(2)


# In[9]:


import tensorflow.keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D


# In[10]:


num_classes = 2
img_rows, img_cols = 28, 28
batch_size = 32

train_data_dir = './handgestures/train'
validation_data_dir = './handgestures/test'


# In[11]:


train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=30,
      width_shift_range=0.3,
      height_shift_range=0.3,
      horizontal_flip=True,
      fill_mode='nearest')
 
validation_datagen = ImageDataGenerator(rescale=1./255)
 
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        color_mode = 'grayscale',
        class_mode='binary')
 
validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        color_mode = 'grayscale',
        class_mode='binary')


# In[12]:


model = Sequential()
model.add(
    Conv2D(64, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.20))


model.add(Dense(1, activation = 'sigmoid'))

print(model.summary())


# In[ ]:


model.compile(loss = 'binary_crossentropy',
              optimizer = 'rmsprop',
              metrics = ['accuracy'])

nb_train_samples = 1109 
nb_validation_samples = 231 
epochs = 10

history = model.fit(
    train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs = epochs,
    validation_data = validation_generator,
    validation_steps = nb_validation_samples // batch_size)



model.save("my_gestures_cnn.h5")

from tensorflow.keras.models import load_model

classifier = load_model('my_gestures_cnn.h5')

import cv2
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    
    ##############################
    frame=cv2.flip(frame, 1)

    #define region of interest
    roi = frame[100:400, 320:620]
    cv2.imshow('roi', roi)
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(roi, (28, 28), interpolation = cv2.INTER_AREA)
    
    cv2.imshow('roi scaled and gray', roi)
    copy = frame.copy()
    cv2.rectangle(copy, (320, 100), (620, 400), (255,0,0), 5)
    
    roi = roi.reshape(1,28,28,1) 
    roi = roi/255
    result = str(classifier.predict_classes(roi, 1, verbose = 0)[0])
    cv2.putText(copy, str(result), (300 , 100), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
    cv2.imshow('frame', copy)    
    
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break
        
cap.release()
cv2.destroyAllWindows()

# In[ ]:




