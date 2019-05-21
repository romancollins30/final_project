#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
import pickle
import os
import keras.preprocessing as kp


# In[11]:


filename = "./models/face_model"
model = pickle.load(open(filename, "rb"))


# In[13]:


path = "testing/"
filelist = os.listdir(path)
for x in filelist:
    if x.endswith(".png"):
        img = kp.image.load_img(path+x, target_size=(227, 227))
        y = kp.image.img_to_array(img)
        y = y.transpose(2,0,1).reshape(1,-1)
        guess = model.predict(y)
        if guess[0] == 1:
            print('real')
        else:
            print('fake')
    if x.endswith(".PNG"):
        img = kp.image.load_img(path+x, target_size=(227, 227))
        y = kp.image.img_to_array(img)
        y = y.transpose(2,0,1).reshape(1,-1)
        guess = model.predict(y)
        if guess[0] == 1:
            print('real')
        else:
            print('fake')

