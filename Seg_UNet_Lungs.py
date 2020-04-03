# coding=utf-8
# Copyright 2020 Shaoning Zeng
#
# Implement and train U-Net for lungs medical image segmentation,
#   on two x-rays datasets: Montgomery County and Shenzhen Hospital.
# 
# Code of U-Net training is referred from:
# https://www.kaggle.com/eduardomineo/u-net-lung-segmentation-montgomery-shenzhen
#  

# import the necessary packages
from __future__ import print_function
#import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Conv2D
from keras.layers.convolutional import Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K

#import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os

from glob import glob
from tqdm import tqdm

DILATE_KERNEL = np.ones((15, 15), np.uint8)

class SegUNetLungs:
  def __init__(self, hps, x_shape=(512,512,1)): 
    self.input_dir    = hps["input_dir"]
    self.prepare_data_dir() # prepare data directories under input_dir
    self.batch_size   = hps["batch_size"]
    self.max_epochs  = hps["max_epochs"]
    self.hps = hps # other hp

    self.name = "segmentation_unet_lungs"
    self.model_dir = os.path.join(self.input_dir, "%s_model" % self.name)
    self.model = self.build_model(x_shape)
    
    # Go train but it will load the best weights if already trained
    self.model = self.train()
  
  def prepare_data_dir(self):
    """
    !mkdir ../input/segmentation
    !mkdir ../input/segmentation/test
    !mkdir ../input/segmentation/train
    !mkdir ../input/segmentation/train/augmentation
    !mkdir ../input/segmentation/train/image
    !mkdir ../input/segmentation/train/mask
    !mkdir ../input/segmentation/train/dilate
    """
    self.seg_dir              = os.path.join(self.input_dir, "segmentation")
    self.seg_test_dir         = os.path.join(self.seg_dir, "test")    
    self.seg_train_dir        = os.path.join(self.seg_dir, "train")
    self.seg_train_aug_dir    = os.path.join(self.seg_train_dir, "augmentation")
    self.seg_train_img_dir    = os.path.join(self.seg_train_dir, "image")
    self.seg_train_mask_dir   = os.path.join(self.seg_train_dir, "mask")
    self.seg_train_dilate_dir = os.path.join(self.seg_train_dir, "dilate")
    # ../input/    
    self.seg_src_dir  = os.path.join(self.input_dir,"pulmonary-chest-xray-abnormalities")
    self.seg_src_shenzhen_train_dir       = os.path.join(self.seg_src_dir, "ChinaSet_AllFiles")
    self.seg_src_shenzhen_train_img_dir   = os.path.join(self.seg_src_shenzhen_train_dir, "CXR_png")
    self.seg_src_shenzhen_mask_dir        = os.path.join(self.input_dir, "shcxr-lung-mask", "mask")
    self.seg_src_montgomery_train_dir     = os.path.join(self.seg_src_dir, "MontgomerySet")
    self.seg_src_montgomery_train_img_dir = os.path.join(self.seg_src_montgomery_train_dir, "CXR_png")
    self.seg_src_montgomery_left_mask_dir = os.path.join(self.seg_src_montgomery_train_dir, 
                                                         "ManualMask", "leftMask")
    self.seg_src_montgomery_right_mask_dir= os.path.join(self.seg_src_montgomery_train_dir,
                                                         "ManualMask", "rightMask")
  
  def prepare_data_montgomery(self):
    """
    1. Combine left and right lung segmentation masks of Montgomery chest x-rays
    2. Resize images to 512x512 pixels
    3. Dilate masks to gain more information on the edge of lungs
    4. Split images into training and test datasets
    5. Write images to /segmentation directory
    """
    montgomery_left_mask_dir = glob(os.path.join(self.seg_src_montgomery_left_mask_dir, '*.png'))
    montgomery_test = montgomery_left_mask_dir[0:50]
    montgomery_train= montgomery_left_mask_dir[50:] # TODO: make it all as train ?

    for left_image_file in tqdm(montgomery_left_mask_dir):
      base_file = os.path.basename(left_image_file)
      image_file = os.path.join(self.seg_src_montgomery_train_img_dir, base_file)
      right_image_file = os.path.join(self.seg_src_montgomery_right_mask_dir, base_file)
  
      image = cv2.imread(image_file)
      left_mask = cv2.imread(left_image_file, cv2.IMREAD_GRAYSCALE)
      right_mask = cv2.imread(right_image_file, cv2.IMREAD_GRAYSCALE)
      
      image = cv2.resize(image, (512, 512))
      left_mask = cv2.resize(left_mask, (512, 512))
      right_mask = cv2.resize(right_mask, (512, 512))
      
      mask = np.maximum(left_mask, right_mask)
      mask_dilate = cv2.dilate(mask, DILATE_KERNEL, iterations=1)
    
      if (left_image_file in montgomery_train):
        cv2.imwrite(os.path.join(self.seg_train_img_dir, base_file), image)
        cv2.imwrite(os.path.join(self.seg_train_mask_dir, base_file), mask)
        cv2.imwrite(os.path.join(self.seg_train_dilate_dir, base_file), mask_dilate)
      else:
        filename, fileext = os.path.splitext(base_file)
        cv2.imwrite(os.path.join(self.seg_test_dir, base_file), image)
        cv2.imwrite(os.path.join(self.seg_test_dir, "%s_mask%s" % (filename, fileext)), mask)
        cv2.imwrite(os.path.join(self.seg_test_dir, "%s_dilate%s" % (filename, fileext)), mask_dilate)
    
    # after processing, show some samples 
    self.show_montgomery(montgomery_train,montgomery_test)
  
  def add_colored_dilate(self, image, mask_image, dilate_image):
    mask_image_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    dilate_image_gray = cv2.cvtColor(dilate_image, cv2.COLOR_BGR2GRAY)    
    mask = cv2.bitwise_and(mask_image, mask_image, mask=mask_image_gray)
    dilate = cv2.bitwise_and(dilate_image, dilate_image, mask=dilate_image_gray)    
    mask_coord = np.where(mask!=[0,0,0])
    dilate_coord = np.where(dilate!=[0,0,0])
    mask[mask_coord[0],mask_coord[1],:]=[255,0,0]
    dilate[dilate_coord[0],dilate_coord[1],:] = [0,0,255]
    ret = cv2.addWeighted(image, 0.7, dilate, 0.3, 0)
    ret = cv2.addWeighted(ret, 0.7, mask, 0.3, 0)
    return ret

  def add_colored_mask(self, image, mask_image):
    mask_image_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)    
    mask = cv2.bitwise_and(mask_image, mask_image, mask=mask_image_gray)    
    mask_coord = np.where(mask!=[0,0,0])
    mask[mask_coord[0],mask_coord[1],:]=[255,0,0]
    ret = cv2.addWeighted(image, 0.7, mask, 0.3, 0)
    return ret

  def diff_mask(self, ref_image, mask_image):
    mask_image_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)    
    mask = cv2.bitwise_and(mask_image, mask_image, mask=mask_image_gray)    
    mask_coord = np.where(mask!=[0,0,0])
    mask[mask_coord[0],mask_coord[1],:]=[255,0,0]
    ret = cv2.addWeighted(ref_image, 0.7, mask, 0.3, 0)
    return ret
  
  def show_montgomery(self, montgomery_train, montgomery_test):
    """
    Show some Montgomery chest x-rays and its lung segmentation masks 
    from training and test dataset to verify the procedure above. 
    In merged image it is possible to see the difference between 
    the dilated mask in blue and the original mask in red.
    """
    base_file = os.path.basename(montgomery_train[0])

    image_file = os.path.join(self.seg_train_img_dir, base_file)
    mask_image_file = os.path.join(self.seg_train_mask_dir, base_file)
    dilate_image_file = os.path.join(self.seg_train_dilate_dir, base_file)
    
    image = cv2.imread(image_file)
    mask_image = cv2.imread(mask_image_file)
    dilate_image = cv2.imread(dilate_image_file)
    merged_image = self.add_colored_dilate(image, mask_image, dilate_image)
                              
    fig, axs = plt.subplots(2, 4, figsize=(15, 8))
    fig.tight_layout()
    
    axs[0, 0].set_title("X-Ray")
    axs[0, 0].imshow(image)    
    axs[0, 1].set_title("Mask")
    axs[0, 1].imshow(mask_image)    
    axs[0, 2].set_title("Dilate")
    axs[0, 2].imshow(dilate_image)    
    axs[0, 3].set_title("Merged")
    axs[0, 3].imshow(merged_image)
    
    base_file = os.path.basename(montgomery_test[0])
    filename, fileext = os.path.splitext(base_file)
    image_file = os.path.join(self.seg_test_dir, base_file)
    mask_image_file = os.path.join(self.seg_test_dir, "%s_mask%s" % (filename, fileext))
    dilate_image_file = os.path.join(self.seg_test_dir, "%s_dilate%s" % (filename, fileext))
    
    image = cv2.imread(image_file)
    mask_image = cv2.imread(mask_image_file)
    dilate_image = cv2.imread(dilate_image_file)
    merged_image = self.add_colored_dilate(image, mask_image, dilate_image)
    
    axs[1, 0].set_title("X-Ray")
    axs[1, 0].imshow(image)    
    axs[1, 1].set_title("Mask")
    axs[1, 1].imshow(mask_image)    
    axs[1, 2].set_title("Dilate")
    axs[1, 2].imshow(dilate_image)    
    axs[1, 3].set_title("Merged")
    axs[1, 3].imshow(merged_image)
  
  def prepare_data_shenzhen(self):
    """
    1. Resize Shenzhen Hospital chest x-ray images to 512x512 pixels
    2. Dilate masks to gain more information on the edge of lungs
    3. Split images into training and test datasets
    4. Write images to /segmentation directory
    """
    shenzhen_mask_dir = glob(os.path.join(self.seg_src_shenzhen_mask_dir, '*.png'))
    shenzhen_test = shenzhen_mask_dir[0:50]
    shenzhen_train= shenzhen_mask_dir[50:]

    for mask_file in tqdm(shenzhen_mask_dir):
      base_file = os.path.basename(mask_file).replace("_mask", "")
      image_file = os.path.join(self.seg_src_shenzhen_train_img_dir, base_file)
  
      image = cv2.imread(image_file)
      mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
          
      image = cv2.resize(image, (512, 512))
      mask = cv2.resize(mask, (512, 512))
      mask_dilate = cv2.dilate(mask, DILATE_KERNEL, iterations=1)
    
      if (mask_file in shenzhen_train):
        cv2.imwrite(os.path.join(self.seg_train_img_dir, base_file), image)
        cv2.imwrite(os.path.join(self.seg_train_mask_dir, base_file), mask)
        cv2.imwrite(os.path.join(self.seg_train_dilate_dir, base_file), mask_dilate)
      else:
        filename, fileext = os.path.splitext(base_file)
        cv2.imwrite(os.path.join(self.seg_test_dir, base_file), image)
        cv2.imwrite(os.path.join(self.seg_test_dir, "%s_mask%s" % (filename, fileext)), mask)
        cv2.imwrite(os.path.join(self.seg_test_dir, "%s_dilate%s" % (filename, fileext)), mask_dilate)
    
    # After that, show some samples
    self.show_shenzhen(shenzhen_train, shenzhen_test)
  
  def show_shenzhen(self,shenzhen_train,shenzhen_test):
    """
    Show some Shenzhen Hospital chest x-rays and its lung segmentation masks 
    from training and test dataset to verify the procedure above. 
    In merged image it is possible to see the difference between 
    the dilated mask in blue and the original mask in red.
    """
    base_file = os.path.basename(shenzhen_train[0].replace("_mask", ""))

    image_file = os.path.join(self.seg_train_img_dir, base_file)
    mask_image_file = os.path.join(self.seg_train_mask_dir, base_file)
    dilate_image_file = os.path.join(self.seg_train_dilate_dir, base_file)
    
    image = cv2.imread(image_file)
    mask_image = cv2.imread(mask_image_file)
    dilate_image = cv2.imread(dilate_image_file)
    merged_image = self.add_colored_dilate(image, mask_image, dilate_image)
                              
    fig, axs = plt.subplots(2, 4, figsize=(15, 8))
    fig.tight_layout()
    
    axs[0, 0].set_title("X-Ray")
    axs[0, 0].imshow(image)    
    axs[0, 1].set_title("Mask")
    axs[0, 1].imshow(mask_image)    
    axs[0, 2].set_title("Dilate")
    axs[0, 2].imshow(dilate_image)    
    axs[0, 3].set_title("Merged")
    axs[0, 3].imshow(merged_image)
    
    base_file = os.path.basename(shenzhen_test[0].replace("_mask", ""))
    image_file = os.path.join(self.seg_test_dir, base_file)
    filename, fileext = os.path.splitext(base_file)
    mask_image_file = os.path.join(self.seg_test_dir, "%s_mask%s" % (filename, fileext))
    
    filename, fileext = os.path.splitext(base_file)
    image_file = os.path.join(self.seg_test_dir, base_file)
    mask_image_file = os.path.join(self.seg_test_dir, "%s_mask%s" % (filename, fileext))
    dilate_image_file = os.path.join(self.seg_test_dir, "%s_dilate%s" % (filename, fileext))
    
    image = cv2.imread(image_file)
    mask_image = cv2.imread(mask_image_file)
    dilate_image = cv2.imread(dilate_image_file)
    merged_image = self.add_colored_dilate(image, mask_image, dilate_image)
    
    axs[1, 0].set_title("X-Ray")
    axs[1, 0].imshow(image)    
    axs[1, 1].set_title("Mask")
    axs[1, 1].imshow(mask_image)    
    axs[1, 2].set_title("Dilate")
    axs[1, 2].imshow(dilate_image)    
    axs[1, 3].set_title("Merged")
    axs[1, 3].imshow(merged_image)
    
  def prepare_data(self):
    """
    Print the count of images and segmentation lung masks available 
    to test and train the model.
    """
    train_files = glob(os.path.join(self.seg_train_img_dir, "*.png"))
    test_files = glob(os.path.join(self.seg_test_dir, "*.png"))
    mask_files = glob(os.path.join(self.seg_train_mask_dir, "*.png"))
    dilate_files = glob(os.path.join(self.seg_train_dilate_dir, "*.png"))    
    (len(train_files), len(test_files), len(mask_files), len(dilate_files))
    return train_files # ,test_files,mask_files,dilate_files
    
  # From: https://github.com/zhixuhao/unet/blob/master/data.py
  def train_generator(self, batch_size, train_path, image_folder, mask_folder, aug_dict,
        image_color_mode="grayscale",
        mask_color_mode="grayscale",
        image_save_prefix="image",
        mask_save_prefix="mask",
        save_to_dir=None,
        target_size=(256,256),
        seed=1):
    '''
    can generate image and mask at the same time use the same seed for
    image_datagen and mask_datagen to ensure the transformation for image
    and mask is the same if you want to visualize the results of generator,
    set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)

    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)

    train_gen = zip(image_generator, mask_generator)
    
    def adjust_data(img,mask):
      img = img / 255
      mask = mask / 255
      mask[mask > 0.5] = 1
      mask[mask <= 0.5] = 0      
      return (img, mask)
    
    for (img, mask) in train_gen:
        img, mask = adjust_data(img, mask)
        yield (img,mask)


  def build_model(self, input_size=(256,256,1)):
    inputs = Input(input_size)    
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    return Model(inputs=[inputs], outputs=[conv10])

  # From: https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py
  def dice_coef(self, y_true, y_pred):
    y_true_f = (y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

  def dice_coef_loss(self, y_true, y_pred):
    return -self.dice_coef(y_true, y_pred) 
  
  def test_load_image(self, test_file, target_size=(256,256)):
    """https://github.com/zhixuhao/unet/blob/master/data.py """
    img = cv2.imread(test_file, cv2.IMREAD_GRAYSCALE)
    img = img / 255
    img = cv2.resize(img, target_size)
    img = np.reshape(img, img.shape + (1,))
    img = np.reshape(img,(1,) + img.shape)
    return img
  
  def predict(self, test_dir=None):
    if not test_dir:
      test_dir = self.seg_test_dir
    # Helper functions to load test chest x-ray images  
    def test_generator(test_files, target_size=(256,256)):
      for test_file in test_files:
        yield self.test_load_image(test_file, target_size)
        
    def save_result(save_path, npyfile, test_files):
      for i, item in enumerate(npyfile):
        result_file = test_files[i]
        img = (item[:, :, 0] * 255.).astype(np.uint8)
        filename, fileext = os.path.splitext(os.path.basename(result_file))
        result_file = os.path.join(save_path, "%s_predict%s" % (filename, fileext))
        cv2.imwrite(result_file, img)
    
    test_files = [test_file for test_file in glob(os.path.join(test_dir, "*.png")) 
                  if ("_mask" not in test_file and "_dilate" not in test_file and "_predict" not in test_file)]
    test_gen = test_generator(test_files, target_size=(512,512))
    results = self.model.predict_generator(test_gen, len(test_files), verbose=1)
    save_result(test_dir, results, test_files)

  def train(self):
    # Prepare data
    self.prepare_data_montgomery()
    self.prepare_data_shenzhen()
    train_files = self.prepare_data()
    
    # Select test and validation files        
    def add_suffix(base_file, suffix):
      filename, fileext = os.path.splitext(base_file)
      return "%s_%s%s" % (filename, suffix, fileext)
    test_files = [test_file for test_file in glob(os.path.join(self.seg_test_dir, "*.png")) 
                  if ("_mask" not in test_file and "_dilate" not in test_file and "_predict" not in test_file)]
    validation_data = (self.test_load_image(test_files[0], target_size=(512, 512)),
                    self.test_load_image(add_suffix(test_files[0], "dilate"), target_size=(512, 512)))
    len(test_files), len(validation_data)

    # Prepare the U-Net model and train the model.
    train_generator_args = dict(
      rotation_range=0.2, 
      width_shift_range=0.05, 
      height_shift_range=0.05, 
      shear_range=0.05, 
      zoom_range=0.05,
      horizontal_flip=True,
      fill_mode='nearest')
    train_gen = self.train_generator(
      self.batch_size, self.seg_train_dir, 'image','dilate', 
      train_generator_args,target_size=(512,512),
      save_to_dir=os.path.abspath(self.seg_train_aug_dir))
    
    model= self.model

    # optimization details
    model.compile(optimizer=Adam(lr=1e-5), 
                  loss=self.dice_coef_loss, 
                  metrics=[self.dice_coef, 'binary_accuracy'])
    model.summary()
    
    # check the last training steps
    checkpoint_dir = self.model_dir
    if not os.path.exists(checkpoint_dir):
      os.mkdir(checkpoint_dir)
    filepath = os.path.join(checkpoint_dir, '%s_{epoch:04d}.h5' % self.name)
    checkpoint_cb = ModelCheckpoint( # Save weights, every 10-epochs. 
        filepath, #save_weights_only=True,period=period,
        monitor='loss', verbose=1, 
        save_best_only=True) 
    last_epochs = 0
    for epoch in range(self.max_epochs,0,-1):
      #print(filepath)
      value = {"epoch": epoch} 
      if os.path.isfile(filepath.format(**value)):
        print("Load saved weights from %s" % filepath.format(**value))
        model.load_weights(filepath.format(**value))
        last_epochs = epoch # the last epoch
        break

    # training process in a for loop with learning rate drop every 25 epochs.
    #def lr_scheduler(epoch):
    #    return learning_rate * (0.5 ** (epoch // lr_drop))
    #reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)
    
    # Continue training
    H = model.fit_generator(
      train_gen,
      steps_per_epoch=len(train_files) / self.max_epochs, 
      epochs=self.max_epochs,
      initial_epoch=last_epochs,
      validation_data=validation_data,
      callbacks=[checkpoint_cb],
      verbose=1)
    
    # Save the final weights
    # model.save_weights(os.path.join(self.model_dir, self.final_weights_file))
    # Plot the training history
    if last_epochs==0: # only plot the training history when training from 1st epoch
      print(H.history)
      fig, axs = plt.subplots(1, 2, figsize = (15, 4))
      fig.tight_layout()
      training_loss = H.history['loss']
      validation_loss = H.history['val_loss']      
      training_accuracy = H.history['binary_accuracy']
      validation_accuracy = H.history['val_binary_accuracy']      
      epoch_count = range(1, len(training_loss) + 1)      
      axs[0].plot(epoch_count, training_loss, 'r--')
      axs[0].plot(epoch_count, validation_loss, 'b-')
      axs[0].legend(['Training Loss', 'Validation Loss'])      
      axs[1].plot(epoch_count, training_accuracy, 'r--')
      axs[1].plot(epoch_count, validation_accuracy, 'b-')
      axs[1].legend(['Training Accuracy', 'Validation Accuracy'])
      plot_path = os.path.join(
        os.getcwd(), "%s_plot_%04d.png" % (self.name,self.max_epochs))
      plt.savefig(plot_path)
    else: # 
      print("Train from 1st epoch if needing to plot H.history!")
    return model

if __name__ == '__main__':
  # 1. Construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument("-in", "--input_dir", type=str, required=True,
    help="the directory of input is required")
  ap.add_argument("-t", "--test_dir", type=str, default=None,
    help="the directory of covid-19 images")
  ap.add_argument("-b", "--batch_size", type=int, default=2,
    help="batch size, default is 2")
  ap.add_argument("-e", "--max_epochs", type=int, default=56,
    help="max epoches, default is 56")
  args = vars(ap.parse_args())
  
  # 2. Create and train the model
  model = SegUNetLungs(args, x_shape=(512,512,1))

  # 3. Perform segmentation
  model.predict()

  # 4. Perform segmentation on COVID-19
  if args["test_dir"]:
    model.predict(args["test_dir"])

