# %% [markdown]
# I trained the model on Kaggle
# 
# https://www.kaggle.com/code/yutong0807/img-classification-with-tf-resnet50/notebook

# %%
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import os
import os.path as path
import cv2
import numpy as np

# %% [markdown]
# ## Load Data

# %%
import random
def knuthShuffle(list1, list2):
    for (i, e) in enumerate(list1):
        j = random.randint(0, i)
        if i != j:
            list1[i], list1[j] = list1[j], list1[i]
            list2[i], list2[j] = list2[j], list2[i]

# %%
def loadData(folder, shuffle=False):
    catFolder = path.join(folder, 'Cat/')
    dogFolder = path.join(folder, 'Dog/')

    catsPath = [path.join(catFolder, f) for f in os.listdir(catFolder) if path.isfile(path.join(catFolder, f))]
    dogsPath = [path.join(dogFolder, f) for f in os.listdir(dogFolder) if path.isfile(path.join(dogFolder, f))]

    cats = []
    dogs = []
    for cat in catsPath:
        cat = cv2.imdecode(np.fromfile(cat, dtype=np.uint8), 1)
        if cat is not None:
            cat = cv2.cvtColor(cat, cv2.COLOR_BGR2RGB)
            cat = cv2.resize(cat, (224, 224))
            cats.append(cat)
    
    for dog in dogsPath:
        dog = cv2.imdecode(np.fromfile(dog, dtype=np.uint8), 1)
        if dog is not None:
            dog = cv2.cvtColor(dog, cv2.COLOR_BGR2RGB)
            dog = cv2.resize(dog, (224, 224))
            dogs.append(dog)
    x = cats+dogs
    y = [0]*len(cats) + [1]*len(dogs)
    
    if shuffle:
        knuthShuffle(x, y)
        
    return np.array(x), np.array(y, dtype=np.int8)

# %%
x_train, y_train = loadData('/kaggle/input/picture-of-cats-and-dogs/training_dataset', shuffle=True)
x_test, y_test = loadData('/kaggle/input/picture-of-cats-and-dogs/validation_dataset')

# %%
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# %% [markdown]
# ## Load Model

# %%
import gc
gc.collect()

# %%
model = tf.keras.Sequential()
resnet = tf.keras.applications.resnet50.ResNet50(include_top = False, input_shape=(224,224,3), weights='imagenet')
model.add(resnet)
model.add(tf.keras.layers.GlobalAveragePooling2D())
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.summary()

# %%
model1 = model
model2 = tf.keras.models.clone_model(model1)

# %%
# early stop
earlyStop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)

# %% [markdown]
# ## Train Model 1

# %%
model1.compile(optimizer='adam',loss=tfa.losses.SigmoidFocalCrossEntropy(alpha=.4, gamma=1.0), metrics=['accuracy'])
history1 = model1.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test), batch_size=16, callbacks=[earlyStop])

# %%
plt.plot(history1.history['accuracy'], label='training')
plt.plot(history1.history['val_accuracy'], label = 'testing')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')

# %%
model1.save("resnet50-SigmoidFocalCrossEntropy")

# %% [markdown]
# ## Train Model 2

# %%
model2.compile(optimizer='adam',loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
history2 = model2.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test), batch_size=16, callbacks=[earlyStop])

# %%
plt.plot(history2.history['accuracy'], label='training')
plt.plot(history2.history['val_accuracy'], label = 'testing')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')

# %%
model2.save("resnet50-BinaryCrossEntrop")


