
# coding: utf-8

# In[1]:


from oneshot import model


# In[2]:


from oneshot import network


# In[3]:


training_dir = "./clean_data"
testing_dir = "./test"
batch_size = 40


# In[4]:


from keras.preprocessing.image import ImageDataGenerator


# In[5]:


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)


# In[6]:


train_generator = train_datagen.flow_from_directory(
        training_dir,
        target_size=(200, 200),
        batch_size=batch_size,
        class_mode='binary')


# In[20]:


import matplotlib.pyplot as plt
train = train_generator.next()
#train


# In[21]:


import collections
collections.Counter(train[1])


# In[22]:


train_x,train_y = train[0],train[1]


# In[23]:


import numpy as np


# In[24]:


train_left_input = np.zeros((20,200,200,3))
train_right_input = np.zeros((20,200,200,3))
for i in range(20):
    #print(train_x[i].shape)
    train_left_input[i] = train_x[i]
    train_right_input[i] = train_x[i+20]


# In[25]:


from keras.optimizers import Adam
network.compile(optimizer=Adam(lr=0.000001),loss='binary_crossentropy')


# In[26]:


train_output = np.zeros((20,))
for i in range(20):
    if train_y[i] == train_y[i+20]:
        train_output[i] = 1
    else:
        train_output[i] = 0


# In[27]:


# for i in range(20):
#     plt.imshow(train_left_input[i])
#     plt.show()
#     plt.imshow(train_right_input[i])
#     plt.show()
#     print(train_output[i])

from keras.callbacks import ModelCheckpoint

filepath = "model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

network.fit([train_left_input,train_right_input],train_output,epochs=20,callbacks=callbacks_list)

