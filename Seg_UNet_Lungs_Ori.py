# !mkdir ../input/segmentation
# !mkdir ../input/segmentation/test
# !mkdir ../input/segmentation/train
# !mkdir ../input/segmentation/train/augmentation
# !mkdir ../input/segmentation/train/image
# !mkdir ../input/segmentation/train/mask
# !mkdir ../input/segmentation/train/dilate

import os

import numpy as np
import cv2
import matplotlib.pyplot as plt

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

from glob import glob
from tqdm import tqdm

INPUT_DIR = os.path.join("/home/snzeng", "Lab0_Data_Keras")

SEGMENTATION_DIR = os.path.join(INPUT_DIR, "segmentation")
SEGMENTATION_TEST_DIR = os.path.join(SEGMENTATION_DIR, "test")
SEGMENTATION_TRAIN_DIR = os.path.join(SEGMENTATION_DIR, "train")
SEGMENTATION_AUG_DIR = os.path.join(SEGMENTATION_TRAIN_DIR, "augmentation")
SEGMENTATION_IMAGE_DIR = os.path.join(SEGMENTATION_TRAIN_DIR, "image")
SEGMENTATION_MASK_DIR = os.path.join(SEGMENTATION_TRAIN_DIR, "mask")
SEGMENTATION_DILATE_DIR = os.path.join(SEGMENTATION_TRAIN_DIR, "dilate")

SEGMENTATION_SOURCE_DIR = os.path.join(INPUT_DIR, "pulmonary-chest-xray-abnormalities")
print("%s exists: %s" % (SEGMENTATION_SOURCE_DIR, str(os.path.exists(SEGMENTATION_SOURCE_DIR))))

SHENZHEN_TRAIN_DIR = os.path.join(SEGMENTATION_SOURCE_DIR, "ChinaSet_AllFiles")
SHENZHEN_IMAGE_DIR = os.path.join(SHENZHEN_TRAIN_DIR, "CXR_png")
SHENZHEN_MASK_DIR = os.path.join(INPUT_DIR, "shcxr-lung-mask", "mask")

MONTGOMERY_TRAIN_DIR = os.path.join(SEGMENTATION_SOURCE_DIR, "MontgomerySet")
MONTGOMERY_IMAGE_DIR = os.path.join(MONTGOMERY_TRAIN_DIR, "CXR_png")
MONTGOMERY_LEFT_MASK_DIR = os.path.join(MONTGOMERY_TRAIN_DIR, "ManualMask", "leftMask")
MONTGOMERY_RIGHT_MASK_DIR = os.path.join(MONTGOMERY_TRAIN_DIR, "ManualMask", "rightMask")

DILATE_KERNEL = np.ones((15, 15), np.uint8)

BATCH_SIZE=2

#Prod
EPOCHS=56

#Desv
#EPOCHS=16

IN_SIZE = 128 # 256, 512

if not os.path.exists('unet_lung_seg.hdf5'): # if not trained
  montgomery_left_mask_dir = glob(os.path.join(MONTGOMERY_LEFT_MASK_DIR, '*.png'))
  montgomery_test = montgomery_left_mask_dir[0:50]
  montgomery_train= montgomery_left_mask_dir[50:]
  
  for left_image_file in tqdm(montgomery_left_mask_dir):
      base_file = os.path.basename(left_image_file)
      image_file = os.path.join(MONTGOMERY_IMAGE_DIR, base_file)
      right_image_file = os.path.join(MONTGOMERY_RIGHT_MASK_DIR, base_file)
  
      image = cv2.imread(image_file)
      left_mask = cv2.imread(left_image_file, cv2.IMREAD_GRAYSCALE)
      right_mask = cv2.imread(right_image_file, cv2.IMREAD_GRAYSCALE)
      
      image = cv2.resize(image, (IN_SIZE, IN_SIZE))
      left_mask = cv2.resize(left_mask, (IN_SIZE, IN_SIZE))
      right_mask = cv2.resize(right_mask, (IN_SIZE, IN_SIZE))
      
      mask = np.maximum(left_mask, right_mask)
      mask_dilate = cv2.dilate(mask, DILATE_KERNEL, iterations=1)
      
      if (left_image_file in montgomery_train):
          cv2.imwrite(os.path.join(SEGMENTATION_IMAGE_DIR, base_file), \
                      image)
          cv2.imwrite(os.path.join(SEGMENTATION_MASK_DIR, base_file), \
                      mask)
          cv2.imwrite(os.path.join(SEGMENTATION_DILATE_DIR, base_file), \
                      mask_dilate)
      else:
          filename, fileext = os.path.splitext(base_file)
          cv2.imwrite(os.path.join(SEGMENTATION_TEST_DIR, base_file), \
                      image)
          cv2.imwrite(os.path.join(SEGMENTATION_TEST_DIR, \
                                   "%s_mask%s" % (filename, fileext)), mask)
          cv2.imwrite(os.path.join(SEGMENTATION_TEST_DIR, \
                                   "%s_dilate%s" % (filename, fileext)), mask_dilate)

def add_colored_dilate(image, mask_image, dilate_image):
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

def add_colored_mask(image, mask_image):
    mask_image_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    
    mask = cv2.bitwise_and(mask_image, mask_image, mask=mask_image_gray)
    
    mask_coord = np.where(mask!=[0,0,0])

    mask[mask_coord[0],mask_coord[1],:]=[255,0,0]

    ret = cv2.addWeighted(image, 0.7, mask, 0.3, 0)

    return ret

def diff_mask(ref_image, mask_image):
    mask_image_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    
    mask = cv2.bitwise_and(mask_image, mask_image, mask=mask_image_gray)
    
    mask_coord = np.where(mask!=[0,0,0])

    mask[mask_coord[0],mask_coord[1],:]=[255,0,0]

    ret = cv2.addWeighted(ref_image, 0.7, mask, 0.3, 0)
    return ret

if not os.path.exists('unet_lung_seg.hdf5'): # if not trained
  shenzhen_mask_dir = glob(os.path.join(SHENZHEN_MASK_DIR, '*.png'))
  shenzhen_test = shenzhen_mask_dir[0:50]
  shenzhen_train= shenzhen_mask_dir[50:]
  
  for mask_file in tqdm(shenzhen_mask_dir):
      base_file = os.path.basename(mask_file).replace("_mask", "")
      image_file = os.path.join(SHENZHEN_IMAGE_DIR, base_file)
  
      image = cv2.imread(image_file)
      mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
          
      image = cv2.resize(image, (IN_SIZE, IN_SIZE))
      mask = cv2.resize(mask, (IN_SIZE, IN_SIZE))
      mask_dilate = cv2.dilate(mask, DILATE_KERNEL, iterations=1)
      
      if (mask_file in shenzhen_train):
          cv2.imwrite(os.path.join(SEGMENTATION_IMAGE_DIR, base_file), \
                      image)
          cv2.imwrite(os.path.join(SEGMENTATION_MASK_DIR, base_file), \
                      mask)
          cv2.imwrite(os.path.join(SEGMENTATION_DILATE_DIR, base_file), \
                      mask_dilate)
      else:
          filename, fileext = os.path.splitext(base_file)
  
          cv2.imwrite(os.path.join(SEGMENTATION_TEST_DIR, base_file), \
                      image)
          cv2.imwrite(os.path.join(SEGMENTATION_TEST_DIR, \
                                   "%s_mask%s" % (filename, fileext)), mask)
          cv2.imwrite(os.path.join(SEGMENTATION_TEST_DIR, \
                                   "%s_dilate%s" % (filename, fileext)), mask_dilate)
        
train_files = glob(os.path.join(SEGMENTATION_IMAGE_DIR, "*.png"))
test_files = glob(os.path.join(SEGMENTATION_TEST_DIR, "*.png"))
mask_files = glob(os.path.join(SEGMENTATION_MASK_DIR, "*.png"))
dilate_files = glob(os.path.join(SEGMENTATION_DILATE_DIR, "*.png"))

(len(train_files), \
 len(test_files), \
 len(mask_files), \
 len(dilate_files))

# From: https://github.com/zhixuhao/unet/blob/master/data.py
def train_generator(batch_size, train_path, image_folder, mask_folder, aug_dict,
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
    
    for (img, mask) in train_gen:
        img, mask = adjust_data(img, mask)
        yield (img,mask)

def adjust_data(img,mask):
    img = img / 255
    mask = mask / 255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    
    return (img, mask)
  
# From: https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py
def dice_coef(y_true, y_pred):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def unet(input_size=(256,256,1)):
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

# From: https://github.com/zhixuhao/unet/blob/master/data.py
def test_load_image(test_file, target_size=(256,256)):
    img = cv2.imread(test_file, cv2.IMREAD_GRAYSCALE)
    img = img / 255
    img = cv2.resize(img, target_size)
    img = np.reshape(img, img.shape + (1,))
    img = np.reshape(img,(1,) + img.shape)
    return img

def test_generator(test_files, target_size=(256,256)):
    for test_file in test_files:
        yield test_load_image(test_file, target_size)
        
def save_result(save_path, npyfile, test_files):
    for i, item in enumerate(npyfile):
        result_file = test_files[i]
        img = (item[:, :, 0] * 255.).astype(np.uint8)

        filename, fileext = os.path.splitext(os.path.basename(result_file))

        result_file = os.path.join(save_path, "%s_predict%s" % (filename, fileext))

        cv2.imwrite(result_file, img)

def add_suffix(base_file, suffix):
    filename, fileext = os.path.splitext(base_file)
    return "%s_%s%s" % (filename, suffix, fileext)

test_files = [test_file for test_file in glob(os.path.join(SEGMENTATION_TEST_DIR, "*.png")) \
              if ("_mask" not in test_file \
                  and "_dilate" not in test_file \
                  and "_predict" not in test_file)]

validation_data = (test_load_image(test_files[0], target_size=(IN_SIZE, IN_SIZE)),
                    test_load_image(add_suffix(test_files[0], "dilate"), target_size=(IN_SIZE, IN_SIZE)))

len(test_files), len(validation_data)

train_generator_args = dict(rotation_range=0.2,
                            width_shift_range=0.05,
                            height_shift_range=0.05,
                            shear_range=0.05,
                            zoom_range=0.05,
                            horizontal_flip=True,
                            fill_mode='nearest')

train_gen = train_generator(BATCH_SIZE,
                            SEGMENTATION_TRAIN_DIR,
                            'image',
                            'dilate', 
                            train_generator_args,
                            target_size=(IN_SIZE,IN_SIZE),
                            save_to_dir=os.path.abspath(SEGMENTATION_AUG_DIR))

model = unet(input_size=(IN_SIZE,IN_SIZE,1))
model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, \
                  metrics=[dice_coef, 'binary_accuracy'])
model.summary()

model_checkpoint = ModelCheckpoint('unet_lung_seg.hdf5', 
                                   monitor='loss', 
                                   verbose=1, 
                                   save_best_only=True)

if not os.path.exists('unet_lung_seg.hdf5'): 
  history = model.fit_generator(train_gen,
                              steps_per_epoch=len(train_files) / BATCH_SIZE, 
                              epochs=EPOCHS, 
                              callbacks=[model_checkpoint],
                              validation_data = validation_data)
else: # load the trained model
  history = None
  model.load_weights('unet_lung_seg.hdf5')
  print("Load pre-trained unet_lung_seg.hdf5")

# Test original test samples
#test_gen = test_generator(test_files, target_size=(IN_SIZE,IN_SIZE))
#results = model.predict_generator(test_gen, len(test_files), verbose=1)
#save_result(SEGMENTATION_TEST_DIR, results, test_files)

def test_customized_dir(model, test_dir, test_ext):
  test_files = [test_file for test_file in glob(os.path.join(test_dir,test_ext)) \
              if ("_mask" not in test_file \
                  and "_dilate" not in test_file \
                  and "_predict" not in test_file)]
  len(test_files)
  test_gen = test_generator(test_files, target_size=(IN_SIZE,IN_SIZE))
  results = model.predict_generator(test_gen, len(test_files), verbose=1)
  save_result(test_dir, results, test_files)

# Test COVID-19 samples
SEGMENTATION_COVID19_DIR = os.path.join(INPUT_DIR, "covid-19")
# dataset100
SEGMENTATION_COVID19_DATA100_DIR = os.path.join(SEGMENTATION_COVID19_DIR, "dataset100")
# - covid
SEGMENTATION_COVID19_DATA100_C_DIR = os.path.join(SEGMENTATION_COVID19_DATA100_DIR, "covid")
test_customized_dir(model, SEGMENTATION_COVID19_DATA100_C_DIR, ".png")
test_customized_dir(model, SEGMENTATION_COVID19_DATA100_C_DIR, ".jpg")
test_customized_dir(model, SEGMENTATION_COVID19_DATA100_C_DIR, ".jpeg")
# - normal
SEGMENTATION_COVID19_DATA100_N_DIR = os.path.join(SEGMENTATION_COVID19_DATA100_DIR, "normal")
test_customized_dir(model, SEGMENTATION_COVID19_DATA100_N_DIR, ".png")
test_customized_dir(model, SEGMENTATION_COVID19_DATA100_N_DIR, ".jpg")
test_customized_dir(model, SEGMENTATION_COVID19_DATA100_N_DIR, ".jpeg")
# dataset100
SEGMENTATION_COVID19_DATA300_DIR = os.path.join(SEGMENTATION_COVID19_DIR, "dataset300")
# - covid
SEGMENTATION_COVID19_DATA300_C_DIR = os.path.join(SEGMENTATION_COVID19_DATA300_DIR, "covid")
test_customized_dir(model, SEGMENTATION_COVID19_DATA300_C_DIR, ".png")
test_customized_dir(model, SEGMENTATION_COVID19_DATA300_C_DIR, ".jpg")
test_customized_dir(model, SEGMENTATION_COVID19_DATA300_C_DIR, ".jpeg")
# - normal
SEGMENTATION_COVID19_DATA300_N_DIR = os.path.join(SEGMENTATION_COVID19_DATA300_DIR, "normal")
test_customized_dir(model, SEGMENTATION_COVID19_DATA300_N_DIR, ".png")
test_customized_dir(model, SEGMENTATION_COVID19_DATA300_N_DIR, ".jpg")
test_customized_dir(model, SEGMENTATION_COVID19_DATA300_N_DIR, ".jpeg")
# - pneumnia
SEGMENTATION_COVID19_DATA300_P_DIR = os.path.join(SEGMENTATION_COVID19_DATA300_DIR, "pneumnia")
test_customized_dir(model, SEGMENTATION_COVID19_DATA300_P_DIR, ".png")
test_customized_dir(model, SEGMENTATION_COVID19_DATA300_P_DIR, ".jpg")
test_customized_dir(model, SEGMENTATION_COVID19_DATA300_P_DIR, ".jpeg")

if history is not None: # Plot training history
  fig, axs = plt.subplots(1, 2, figsize = (15, 4))
  
  training_loss = history.history['loss']
  validation_loss = history.history['val_loss']
  
  training_accuracy = history.history['binary_accuracy']
  validation_accuracy = history.history['val_binary_accuracy']
  
  epoch_count = range(1, len(training_loss) + 1)
  
  axs[0].plot(epoch_count, training_loss, 'r--')
  axs[0].plot(epoch_count, validation_loss, 'b-')
  axs[0].legend(['Training Loss', 'Validation Loss'])
  
  axs[1].plot(epoch_count, training_accuracy, 'r--')
  axs[1].plot(epoch_count, validation_accuracy, 'b-')
  axs[1].legend(['Training Accuracy', 'Validation Accuracy'])
  
  plt.savefig('seg_unet_lungs_results.jpg', bbox_inches='tight')