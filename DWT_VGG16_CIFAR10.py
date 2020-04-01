# coding=utf-8
# Copyright 2020 Shaoning Zeng
#
# Discriminative Wavelet Tuning Deep Neural Networks for Detecting COVID-19
# 
# Cite our paper: 

# import the necessary packages
from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.layers import Input, Reshape
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, BatchNormalization
from keras.layers.core import Lambda
from keras.models import Model
from keras import optimizers
from keras import backend as K
from keras import regularizers

# resize input to 48x48 at least
import cv2  

import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from WaveletDeconvolution import WaveletDeconvolution
from keras.callbacks import ModelCheckpoint

class DWTVGG16Cifar10:
  def __init__(self, hps, train=True): 
    self.dataset      = hps["dataset"]
    self.num_classes  = hps["num_classes"] #10
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

    self.final_weights_file = "%s_%d.h5" % (self.name, self.max_epochs)
    self.model = self.build_model()
    if train:
      self.model = self.train(self.model)
    else: # load the saved model
      self.model.load_weights(os.path.join(os.getcwd(), self.final_weights_file))


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
    headModel = Dense(512,kernel_regularizer=regularizers.l2(self.weight_decay))(headModel)
    headModel = Activation('relu')(headModel)
    headModel = BatchNormalization()(headModel)

    headModel = Dropout(0.5)(headModel)
    headModel = Dense(self.num_classes)(headModel)
    headModel = Activation('softmax')(headModel)
    
    # place the head FC model on top of the base model
    # (this will become the actual model we will train)
    model = Model(inputs=baseModel.input, outputs=headModel)    
    return model


  def normalize(self,X_train,X_test):
    #this function normalize inputs for zero mean and unit variance
    # it is used when training a model.
    # Input: training set and test set
    # Output: normalized training set and test set according to the trianing set statistics.
    mean = np.mean(X_train,axis=(0,1,2,3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    X_train = (X_train-mean)/(std+1e-7)
    X_test = (X_test-mean)/(std+1e-7)
    return X_train, X_test

  def normalize_production(self,x):
    #this function is used to normalize instances in production according to saved training set statistics
    # Input: X - a training set
    # Output X - a normalized training set according to normalization constants.

    #these values produced during first training and are general for the standard cifar10 training set normalization
    mean = 120.707
    std = 64.15
    return (x-mean)/(std+1e-7)

  def predict(self, normalize=True, batch_size=50):
    x = self.x_test
    y = self.y_test
    if normalize:
        x = self.normalize_production(x)
    predIdxs = self.model.predict(x, batch_size)

    # for each image in the testing set we need to find the index of the
    # label with corresponding largest predicted probability
    predIdxs = np.argmax(predIdxs, axis=1)
    
    # show a nicely formatted classification report
    print(classification_report(y.argmax(axis=1), predIdxs))
    
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
        os.getcwd(), "%s_result_%04d_%.4f_%.4f_%.4f.txt" % (
          self.name,self.max_epochs,acc,sensitivity,specificity))
    fo = open(result_path, "w")
    fo.write(self.hps)
    fo.close()

  def train(self, model):
    #training parameters
    batch_size = self.batch_size # 128
    max_epochs = self.max_epochs # 250
    learning_rate = 0.1
    lr_decay = 1e-6
    lr_drop = 20
    
    # The data, shuffled and split between train and test sets:
    #(x_train, y_train), (x_test, y_test) = cifar10.load_data()
    #x_train = x_train.astype('float32')
    #x_test = x_test.astype('float32')
    x_train = self.x_train
    x_test  = self.x_test
    x_train, x_test = self.normalize(x_train, x_test)

    #y_train = keras.utils.to_categorical(y_train, self.num_classes)
    #y_test = keras.utils.to_categorical(y_test, self.num_classes)
    y_train = self.y_train
    y_test  = self.y_test

    # data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # optimization details
    sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
    
    # check the last training steps
    checkpoint_dir = os.path.join(os.getcwd(), "%s_train" % self.name)
    if not os.path.exists(checkpoint_dir):
      os.mkdir(checkpoint_dir)
    filepath = os.path.join(checkpoint_dir, '%s_{epoch:04d}.h5' % self.name)
    period = 10
    assert self.max_epochs % period == 0, "please set max epochs as n*%d" % period
    checkpoint_cb = ModelCheckpoint( # Save weights, every 10-epochs. 
        filepath, save_weights_only=True, verbose=1, period=period) 
    last_epochs = 0
    for epoch in range(self.max_epochs,0,-1*period):
      #print(filepath)
      value = {"epoch": epoch}
      if os.path.isfile(filepath.format(**value)):
        print("Load saved weights from %s" % filepath.format(**value))
        model.load_weights(filepath.format(**value))
        last_epochs = epoch
        break

    # training process in a for loop with learning rate drop every 25 epochs.
    def lr_scheduler(epoch):
        return learning_rate * (0.5 ** (epoch // lr_drop))
    reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)
    
    # Continue training
    H = model.fit_generator(
      datagen.flow(x_train, y_train,batch_size=batch_size),
      steps_per_epoch=x_train.shape[0] // batch_size,
      epochs=max_epochs,
      initial_epoch=last_epochs,
      validation_data=(x_test, y_test),
      callbacks=[reduce_lr,checkpoint_cb],
      verbose=1)
    
    # Save the final weights
    model.save_weights(os.path.join(os.getcwd(), self.final_weights_file))
    # plot the training loss and accuracy
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
      os.getcwd(), "%s_plot_%04d.png" % (self.name,self.max_epochs))
    plt.savefig(plot_path)
    return model

if __name__ == '__main__':
  # 1. Construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument("-d", "--dataset", type=str, default="cifar10",
    help="the dataset name, default is cifar10")
  #ap.add_argument("-p", "--plot", type=str, default="plot.png",
  #  help="path to output loss/accuracy plot")
  #ap.add_argument("-m", "--model", type=str, default="keras_covid19",
  #  help="path to output loss/accuracy plot")
  ap.add_argument("-b", "--batch_size", type=int, default=128,
    help="batch size, default is 128")
  ap.add_argument("-e", "--max_epochs", type=int, default=250,
    help="max epoches, default is 250")
  ap.add_argument('--wavelet', dest='wavelet', action='store_true')
  ap.add_argument('--no-wavelet', dest='wavelet', action='store_false')
  ap.set_defaults(wavelet=False)
  args = vars(ap.parse_args())
  
  in_size = 64
  
  # 2. Load CIFAR-10
  (x_train, y_train), (x_test, y_test) = cifar10.load_data()
  #x_train = x_train.astype('float32')
  #x_test  = x_test.astype('float32')
  x_train = [cv2.resize(i,(in_size,in_size)) for i in x_train]
  x_test  = [cv2.resize(i,(in_size,in_size)) for i in x_test]
  x_train = np.concatenate([arr[np.newaxis] for arr in x_train] ).astype('float32')
  x_test  = np.concatenate([arr[np.newaxis] for arr in x_test] ).astype('float32')
  y_train = keras.utils.to_categorical(y_train, 10)
  y_test  = keras.utils.to_categorical(y_test, 10)
  args['x_train'] = x_train
  args['x_test']  = x_test
  args['y_train'] = y_train
  args['y_test']  = y_test
  args["num_classes"]  = 10
  args["weight_decay"] = 0.0005

  # 3. Create and train the model
  model = DWTVGG16Cifar10(args, True)

  # 4. Predict the result
  predicted_x = model.predict()

