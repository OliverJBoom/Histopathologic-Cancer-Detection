import cv2
import gc
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from keras.applications.resnet50 import ResNet50
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, ZeroPadding2D
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from skimage.io import imread

    
def df_init(path, train_dir):
    """Returns the dataframe containing the ids, path and 
    labels for the training dataset"""
    offset = len(path + "train/")
    df = pd.DataFrame({'path': glob(os.path.join(train_dir,'*.tif'))})
    df['id'] = df['path'].map(lambda x: str(x)[offset:-4])
    labels = pd.read_csv(path+"train_labels.csv")
    df = df.merge(labels, on = "id")
    # df[df['id'] != 'dd6dfed324f9fcb6f93f46f32fc800f2ec196be2']
    return df
    
    
def df_test_init(test_dir):
    """Returns the dataframe containing the ids, path and labels
    for the test dataset"""
    df = pd.DataFrame({'path': glob(os.path.join(test_dir,'*.tif'))})
    df['filename'] = df['path'].map(lambda x: str(x)[len(test_dir):])
    df['id'] = df['path'].map(lambda x: str(x)[len(test_dir):-4])
    return df


def dir_check(directory):
    """Prints the directory"""
    print(os.listdir(directory))
    
    
def history_display():
    """Displays the models training/validation performance"""
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.savefig("History.png")
    
    
def show_images(df, path):
    """Prints out a selection of random images from the target directory"""
    fig = plt.figure(figsize=(20, 8))
    index = 1
    for i in np.random.randint(low=0, high=df.shape[0], size=10):
        file = path + df.iloc[i]['id'] + '.tif'
        img = cv2.imread(file)
        ax = fig.add_subplot(2, 5, index)
        ax.imshow(img, cmap='gray')
        index = index + 1
        color = ['green' if df.iloc[i].label == 1 else 'red'][0]
        ax.set_title(df.iloc[i].label, fontsize=18, color=color)
    plt.tight_layout()
    plt.show()
    

def data_label_view(df):
    """Returns information on the quantity of labels in the dataset"""
    value_count = df['label'].value_counts()
    sns.countplot(df['label'], alpha=0.8)
    plt.title('Split between positive and negative')
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel('Labels', fontsize=12)
    plt.ylim(0, 1.2 * df.label.value_counts()[0])
    plt.annotate("No tumour: %d" % (df.label.value_counts()[0]),
                 xy=(0, df.label.value_counts()[0] + 2000),
                 va='bottom',
                 ha='center',
                 fontsize=12)
    plt.annotate("Tumour: %d" % (df.label.value_counts()[1]),
                 xy=(1, df.label.value_counts()[1] + 2000),
                 va='bottom',
                 ha='center',
                 fontsize=12)
    plt.show()
        
    
class cnn:

    def __init__(self, dimensions, path):
        """
        :param width: Input width of test images
        :param height: Input height of test images
        :param channels: Number of spectrums channels. For RBG = 3 channels
        """
        self.width = dimensions[0]
        self.height = dimensions[1]
        self.channels = dimensions[2]
        self.path = path

        
    def train_test_generators(self, df_train, df_val):
        """Batch generators to shuttle the train/val data into the CNN"""
        datagen = ImageDataGenerator(rescale=1./255)
                                    
        # Want generators for both training and validation set
        train_gen = datagen.flow_from_dataframe(df_train, 
                                                directory="../input/train/", 
                                                x_col='filename',
                                                y_col='label', 
                                                target_size=(self.width, self.height),
                                                class_mode='binary', 
                                                batch_size=32, 
                                                shuffle=False)
                                              
        val_gen = datagen.flow_from_dataframe(df_val, 
                                            directory="../input/train/", 
                                            x_col='filename',
                                            y_col='label', 
                                            target_size=(self.width, self.height),
                                            class_mode='binary', 
                                            batch_size=32, 
                                            shuffle=False)
        
        return train_gen, val_gen
        
        
# Defining key initial paths
path = "../input/"
train_dir = path + "train/"
test_dir = path + "test/"

# Stitching togehter the df
df = df_init(path, train_dir)
df['filename'] = df['path'].map(lambda x: str(x)[len(train_dir):])
# In new update binary classification requires strings not ints
df['label'] = df['label'].astype(str)

# Some initial data exploration
# show_images(df, train_dir)
# data_label_view(df)
# df.head()

# Building the CNN
df_train, df_val = train_test_split(df, test_size=0.10, random_state=42, stratify=df['label'])
dimensions = cv2.imread(df_train['path'][0]).shape
print(dimensions)

net = cnn(dimensions, 'base_dir/')
model = Sequential()

# model.add(ZeroPadding2D((51, 51), input_shape=dimensions))

# Can try with and without preloaded weights
# Can try with variations on pooling
model.add(ResNet50(weights='imagenet', include_top=False, input_shape=(96, 96, 3), pooling=None))
model.add(Flatten())
# model.add(Dense(units=128, activation='softmax'))
model.add(Dense(units=1, activation='sigmoid'))

print("Model Built")
# ResNet50(weights='imagenet', include_top=False, input_shape=(198, 198, 3)).summary()
model.summary()

# Train test split occurs within image generator
train_gen, val_gen = net.train_test_generators(df_train, df_val)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
del df
gc.collect()

# Training CNN
train_sample = len(df_train)
val_sample = len(df_val)
batch_size = 32

filepath = "model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

early = EarlyStopping(monitor='val_loss', 
                     min_delta=0, 
                     patience=2, 
                     verbose=1, 
                     mode='auto')
        
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=2, min_lr=0.001)
                              
callbacks_list = [checkpoint, early, reduce_lr]

print('Beginning training')
history = model.fit_generator(train_gen,
                         steps_per_epoch= np.ceil(train_sample / batch_size),
                         epochs=10,
                         validation_data=val_gen,
                         validation_steps=np.ceil(val_sample / batch_size),
                         callbacks=callbacks_list)

history_display()

# Predicting the outcome on the testing set
df_test = df_test_init(test_dir)
datagen = ImageDataGenerator(rescale=1./255)
test_gen = datagen.flow_from_dataframe(
    dataframe = df_test,
    directory = test_dir,
    target_size = (96, 96),
    shuffle=False,
    batch_size=32,
    class_mode = None)
    
pred = model.predict_generator(test_gen, steps= np.ceil(len(df_test) / batch_size), verbose=1) 

# Compiling the results into a submission csv
submission = pd.DataFrame()
submission['id'] = df_test['id']
submission['label'] = np.round(pred)
submission['label'] = submission['label'].astype(int)
submission.head()
submission.to_csv("submission.csv",index=False)

print('Prediction complete')