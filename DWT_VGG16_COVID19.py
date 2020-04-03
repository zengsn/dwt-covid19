# coding=utf-8
# Copyright 2020 Shaoning Zeng
#
# Discriminative Wavelet Tuning Deep Neural Networks for Detecting COVID-19
# 
# Cite our paper: 

# import the necessary packages
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Reshape
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D
from keras.layers.core import Lambda
from keras.models import Model
from keras.optimizers import Adam
#from keras import backend as K
#from keras import regularizers
from keras.utils import to_categorical

# resize input to 48x48 at least
import cv2  

import tensorflow as tf
#from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from WaveletDeconvolution import WaveletDeconvolution

class DWTVGG16COVID19:
  def __init__(self, hps, train=True): 
    self.dataset      = hps["dataset"]
    self.num_classes  = hps["num_classes"] #10
    self.learning_rate= hps["learning_rate"] #1e-3
    self.weight_decay = hps["weight_decay"] #0.0005
    self.wavelet      = hps["wavelet"]
    self.batch_size   = hps["batch_size"]
    self.max_epochs  = hps["max_epochs"]
    self.x_train = hps["x_train"]
    self.x_test  = hps["x_test"]
    self.y_train = hps["y_train"]
    self.y_test  = hps["y_test"]
    self.x_shape = self.x_train.shape[1:]
    self.hps = hps # other hp
    if self.wavelet:
      self.name = "dwt_vgg16_%s_wavelet_%d" % (
        self.dataset, self.batch_size)
    else:
      self.name = "dwt_vgg16_%s_%d" % (
        self.dataset, self.batch_size)

    self.model_dir = os.path.join(os.getcwd(), "%s_model" % self.name)
    self.final_weights_file = "%s_%d.h5" % (self.name, self.max_epochs)
    self.model = self.build_model()
    if train:
      self.model = self.train(self.model)
    else: # load the saved model
      self.model.load_weights(os.path.join(
        self.model_dir, self.final_weights_file))


  def build_model(self):
    baseModel = VGG16( # Pre-trained VGG16
      weights="imagenet", include_top=False, 
      input_tensor=Input(shape=self.x_shape))
    # loop over all layers in the base model and freeze them so they will
    # *not* be updated during the first training process
    for layer in baseModel.layers:
      layer.trainable = False
    # construct the head of the model 
    # that will be placed on top of the the base model
    headModel = baseModel.output
    if self.hps["wavelet"]: # add WaveletDeconv       
      print(headModel.shape) 
      w = headModel.shape[1].value 
      h = headModel.shape[2].value 
      z = headModel.shape[3].value 
      # TypeError: float() argument must be a string or a number, not 'Dimension' keras
      # Solution: https://blog.csdn.net/qq_40774175/article/details/105196387
      # When using Reshape, make sure the shape is using int values. 
      headModel = Reshape([w*h,z])(headModel)
      # Error: Variable is unhashable if Tensor equality is enabled
      #headModel = Lambda(lambda x: tf.reshape(x, [-1,7*7,512]))(headModel) 
      headModel = WaveletDeconvolution(
        5, kernel_length=500, padding='same', 
        input_shape=[w*h,z], data_format='channels_first')(headModel)
      headModel = Activation('tanh')(headModel)
      headModel = Conv2D(5, (3, 3), padding='same')(headModel)
      headModel = Activation('relu')(headModel)
      print(headModel.shape) 
      headModel = Lambda(lambda x: tf.reduce_min(x, 3))(headModel) 
      headModel = Reshape([w,h,z])(headModel)
      print(headModel.shape) 

    # Normal fine-tuning
    headModel = Flatten()(headModel)
    #headModel = Dense(512,kernel_regularizer=regularizers.l2(self.weight_decay))(headModel)
    headModel = Dense(64)(headModel)
    headModel = Activation('relu')(headModel)
    #headModel = BatchNormalization()(headModel)

    headModel = Dropout(0.5)(headModel)
    headModel = Dense(self.num_classes)(headModel)
    headModel = Activation('softmax')(headModel)
    
    # place the head FC model on top of the base model
    # (this will become the actual model we will train)
    model = Model(inputs=baseModel.input, outputs=headModel)    
    return model 

  def predict(self):
    x = self.x_test
    y = self.y_test
    predIdxs = self.model.predict(x, self.batch_size)

    # for each image in the testing set we need to find the index of the
    # label with corresponding largest predicted probability
    predIdxs = np.argmax(predIdxs, axis=1)
    
    # show a nicely formatted classification report
    print(classification_report(y.argmax(axis=1), predIdxs, target_names=le.classes_))
    
    # compute the confusion matrix and and use it to derive the raw
    # accuracy, sensitivity, and specificity
    cm = confusion_matrix(y.argmax(axis=1), predIdxs)
    total = sum(sum(cm))
    acc = (cm[0, 0] + cm[1, 1]) / total
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    
    # show the confusion matrix, accuracy, sensitivity, and specificity
    print(cm)
    print("acc: {:.4f}".format(acc))
    print("sensitivity: {:.4f}".format(sensitivity))
    print("specificity: {:.4f}".format(specificity))
    result_path = os.path.join(
        self.model_dir, "%s_result_%04d_%.4f_%.4f_%.4f.txt" % (
          self.name,self.max_epochs,acc,sensitivity,specificity))
    fo = open(result_path, "w")
    fo.write(str(self.hps))
    fo.close()
  
  def evaluate(self):
    x = self.x_test
    y = self.y_test
    test_loss, test_acc = self.model.evaluate(x, y)
    print("test_loss: {:.4f}".format(test_loss))
    print("test_acc: {:.4f}".format(test_acc))
    result_path = os.path.join(
        self.model_dir, "%s_result_%04d_%.4f_%.4f.txt" % (
          self.name,self.max_epochs,test_acc,test_loss))
    fo = open(result_path, "w")
    fo.write(str(self.hps))
    fo.close()

  def train(self, model):
    #training parameters
    batch_size = self.batch_size # 8
    max_epochs = self.max_epochs # 25
    learning_rate = self.learning_rate # 0.001
    lr_decay = learning_rate / max_epochs #1e-6
    #lr_drop = 20
    
    # The data, shuffled and split between train and test sets:
    #(x_train, y_train), (x_test, y_test) = cifar10.load_data()
    #x_train = x_train.astype('float32')
    #x_test = x_test.astype('float32')
    x_train = self.x_train
    x_test  = self.x_test
    #x_train, x_test = self.normalize(x_train, x_test)

    #y_train = keras.utils.to_categorical(y_train, self.num_classes)
    #y_test = keras.utils.to_categorical(y_test, self.num_classes)
    y_train = self.y_train
    y_test  = self.y_test

    # data augmentation
    datagen = ImageDataGenerator(
      #featurewise_center=False, # set input mean to 0 over the dataset
      #samplewise_center=False,  # set each sample mean to 0
      #featurewise_std_normalization=False, # divide inputs by std of the dataset
      #samplewise_std_normalization=False,  # divide each input by its std
      #zca_whitening=False,   # apply ZCA whitening
      rotation_range=15,      # randomly rotate images in the range (degrees, 0 to 180)
      #width_shift_range=0.1, # randomly shift images horizontally (fraction of total width)
      #height_shift_range=0.1,# randomly shift images vertically (fraction of total height)
      #horizontal_flip=True,  # randomly flip images
      #vertical_flip=False,   # randomly flip images
      fill_mode="nearest")  
    # (std, mean, and principal components if ZCA whitening is applied).
    #datagen.fit(x_train) # featurewise_center, featurewise_std_normalization and zca_whitening.

    # optimization details
    opt = Adam(lr=learning_rate, decay=lr_decay)
    if self.num_classes == 2:
      model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    else: #self.num_classes == 3
      model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])    
    
    # check the last training steps
    checkpoint_dir = self.model_dir
    if not os.path.exists(checkpoint_dir):
      os.mkdir(checkpoint_dir)
    filepath = os.path.join(checkpoint_dir, '%s_{epoch:04d}.h5' % self.name)
    period = 5
    assert self.max_epochs % period == 0, "please set max epochs as n*%d" % period
    checkpoint_cb = ModelCheckpoint( # Save weights, every 10-epochs. 
        filepath, save_weights_only=True, verbose=1, period=period) 
    # TODO add customized history callback
    # https://github.com/keras-team/keras/blob/master/keras/callbacks/callbacks.py#L614
    # history_db = keras.callbacks.callbacks.History()
    last_epochs = 0
    for epoch in range(self.max_epochs,0,-1*period):
      #print(filepath)
      value = {"epoch": epoch}
      if os.path.isfile(filepath.format(**value)):
        print("Load saved weights from %s" % filepath.format(**value))
        model.load_weights(filepath.format(**value))
        # TODO: load history
        # history_db.history.append(v)
        last_epochs = epoch
        break

    # training process in a for loop with learning rate drop every 25 epochs.
    #def lr_scheduler(epoch):
    #    return learning_rate * (0.5 ** (epoch // lr_drop))
    #reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)
    
    # Continue training
    H = model.fit_generator(
      datagen.flow(x_train, y_train,batch_size=batch_size),
      steps_per_epoch=x_train.shape[0] // batch_size,
      epochs=max_epochs,
      initial_epoch=last_epochs,
      validation_data=(x_test, y_test),
      callbacks=[checkpoint_cb],
      verbose=1)
    
    # Save the final weights
    model.save_weights(os.path.join(self.model_dir, self.final_weights_file))
    if last_epochs<max_epochs: # plot the training loss and accuracy
      print(H.history)
      N = max_epochs
      plt.style.use("ggplot")
      plt.figure()
      plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
      plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
      if "accuracy" in H.history:
        plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
      else: 
        plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
      plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
      plt.title("Training Loss and Accuracy (%s)" % self.name)
      plt.xlabel("Epoch #")
      plt.ylabel("Loss/Accuracy")
      plt.legend(loc="lower left")
      plot_path = os.path.join(
        self.model_dir, "%s_plot_%04d.png" % (self.name,self.max_epochs))
      plt.savefig(plot_path)
    else: # 
      print("Train from 1st epoch if needing to plot H.history!")
    return model

if __name__ == '__main__':
  # 1. Construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument("-d", "--dataset", type=str, default="covid19",
    help="dataset name, default is covid19")
  ap.add_argument("-dd", "--dataset_dir", type=str, required=True,
    help="directory containing COVID-19 images")
  #ap.add_argument("-nc", "--num_classes", type=int, default=2,
  #  help="number of classes, default is 2, covid19 and normal")
  #ap.add_argument("-p", "--plot", type=str, default="plot.png",
  #  help="path to output loss/accuracy plot")
  #ap.add_argument("-m", "--model", type=str, default="keras_covid19",
  #  help="path to output loss/accuracy plot")
  ap.add_argument("-bs", "--batch_size", type=int, default=8,
    help="batch size, default is 8")
  ap.add_argument("-lr", "--learning_rate", type=float, default=1e-3,
    help="learning rate, default is 1e-3")
  ap.add_argument("-wd", "--weight_decay", type=float, default=0.0005,
    help="weight decay, default is 0.0005")
  ap.add_argument("-me", "--max_epochs", type=int, default=25,
    help="max epoches, default is 25")
  ap.add_argument('--wavelet', dest='wavelet', action='store_true')
  ap.add_argument('--no-wavelet', dest='wavelet', action='store_false')
  ap.set_defaults(wavelet=False)
  args = vars(ap.parse_args())
  
  in_size = 224
  
  # 2. Load images
  # grab the list of images in our dataset directory, then initialize
  # the list of data (i.e., images) and class images
  print("[INFO] loading images...")
  imagePaths = list(paths.list_images(args["dataset_dir"]))
  data = []
  labels = []

  # loop over the image paths
  for imagePath in imagePaths:
    # extract the class label from the filename
    label = imagePath.split(os.path.sep)[-2]
  
    # load the image, swap color channels, and resize it to be a fixed
    # 224x224 pixels while ignoring aspect ratio
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (in_size, in_size))
  
    # update the data and labels lists, respectively
    data.append(image)
    labels.append(label)

  # convert the data and labels to NumPy arrays while scaling the pixel
  # intensities to the range [0, 255]
  data = np.array(data) / 255.0
  #print(data)
  labels = np.array(labels)
  print(labels)
  
  # perform one-hot encoding on the labels
  #lb = LabelBinarizer()
  #labels = lb.fit_transform(labels)
  le = LabelEncoder()
  labels = le.fit_transform(labels)
  labels = to_categorical(labels)  
  #print(labels)
  #print(le.classes_)
  num_classes = len(le.classes_) # 2 or 3
  args["num_classes"] = num_classes

  # partition the data into training and testing splits using 80% of
  # the data for training and the remaining 20% for testing
  (x_train, x_test, y_train, y_test) = train_test_split(data, labels,
    test_size=0.20, stratify=labels, random_state=42)
  #print(y_train)
  #print(y_test)
  args['x_train'] = x_train
  args['x_test']  = x_test
  args['y_train'] = y_train
  args['y_test']  = y_test
  print(y_train[:10,:])
  print(y_test[:10,:])

  # 3. Create and train the model
  model = DWTVGG16COVID19(args, True)

  # 4. Predict the result
  predicted_x = model.predict()

  # 5. Evaluate the result
  model.evaluate()

