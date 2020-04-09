# coding=utf-8
# Copyright 2020 Shaoning Zeng
#
# Discriminative Wavelet Tuning Deep Neural Networks for Detecting COVID-19
# 
# Cite our paper: 

# import the necessary packages
from __future__ import print_function
import tensorflow as tf 
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Reshape, merge
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv2D
from keras.layers.core import Lambda
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling2D, MaxPooling2D,\
  AveragePooling2D
from keras.models import Model
from keras.optimizers import SGD
#from tf.keras import backend as K
#from tf.keras import regularizers
from keras.utils import to_categorical

# resize input to 48x48 at least
import cv2  

#from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

from WaveletDeconvolution import WaveletDeconvolution
from Bias_Off_Crop import crop as bias_off_crop
from custom_layers.scale_layer import Scale

class DWTDenseNetCOVID19:
  def __init__(self, hps, train=True): 
    self.dataset      = hps["dataset"]
    self.network      = hps["network"]
    self.weights_dir  = hps["weights_dir"]
    self.num_classes  = hps["num_classes"] #10
    self.learning_rate= hps["learning_rate"] #1e-3
    self.weight_decay = hps["weight_decay"] #0.0005
    self.wavelet      = hps["wavelet"]
    self.bias_off_crop= hps["bias_off_crop"]
    self.batch_size   = hps["batch_size"]
    self.max_epochs  = hps["max_epochs"]
    self.x_train = hps["x_train"]
    self.x_test  = hps["x_test"]
    self.x_val   = hps["x_val"]
    self.y_train = hps["y_train"]
    self.y_test  = hps["y_test"]
    self.y_val   = hps["y_val"]
    self.x_shape = self.x_train.shape[1:]
    self.hps = hps # other hp
    if self.bias_off_crop: # append _boc to data set name
      self.dataset = "%s_boc" % self.dataset   
    if self.wavelet:
      self.name = "dwt_densenet169_%s_wavelet_%d" % (
        self.dataset, self.batch_size)
    else:
      self.name = "dwt_densenet169_%s_%d" % (
        self.dataset, self.batch_size)

    self.model_dir = os.path.join(os.getcwd(), "%s_model" % self.name)
    self.final_weights_file = "%s_%d.h5" % (self.name, self.max_epochs)
    self.model = self.build_model()
    if train:
      self.model = self.train(self.model)
    else: # load the saved model
      self.model.load_weights(os.path.join(
        self.model_dir, self.final_weights_file))
  
  def conv_block(self, x, stage, branch, nb_filter, dropout_rate=None, weight_decay=1e-4):
    '''Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
        # Arguments
            x: input tensor 
            stage: index for dense block
            branch: layer index within each dense block
            nb_filter: number of filters
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''
    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_' + str(branch)
    relu_name_base = 'relu' + str(stage) + '_' + str(branch)

    # 1x1 Convolution (Bottleneck layer)
    inter_channel = nb_filter * 4  
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x1_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_x1_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x1')(x)
    x = Conv2D(inter_channel, (1, 1), name=conv_name_base+'_x1', bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    # 3x3 Convolution
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x2_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_x2_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x2')(x)
    x = ZeroPadding2D((1, 1), name=conv_name_base+'_x2_zeropadding')(x)
    x = Conv2D(nb_filter, (3, 3), name=conv_name_base+'_x2', bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


  def transition_block(self, x, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout 
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_filter: number of filters
            compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''

    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_blk'
    relu_name_base = 'relu' + str(stage) + '_blk'
    pool_name_base = 'pool' + str(stage) 

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_scale')(x)
    x = Activation('relu', name=relu_name_base)(x)
    x = Conv2D(int(nb_filter * compression), (1, 1), name=conv_name_base, bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(x)

    return x

  def dense_block(self, x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_layers: the number of layers of conv_block to append to the model.
            nb_filter: number of filters
            growth_rate: growth rate
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            grow_nb_filters: flag to decide to allow number of filters to grow
    '''

    eps = 1.1e-5
    concat_feat = x

    for i in range(nb_layers):
        branch = i+1
        x = self.conv_block(concat_feat, stage, branch, growth_rate, dropout_rate, weight_decay)
        concat_feat = merge([concat_feat, x], mode='concat', concat_axis=concat_axis, name='concat_'+str(stage)+'_'+str(branch))

        if grow_nb_filters:
            nb_filter += growth_rate

    return concat_feat, nb_filter
      
  def build_densenet_model(self, inputs,                            
      nb_dense_block=4, growth_rate=32, 
      nb_filter=64, reduction=0.5, 
      dropout_rate=0.0, weight_decay=1e-4):  
    '''
    DenseNet 169 Model for Keras
    Model Schema is based on 
    https://github.com/flyyufelix/DenseNet-Keras
    ImageNet Pretrained Weights 
    Theano: https://drive.google.com/open?id=0Byy2AcGyEVxfN0d3T1F1MXg0NlU
    TensorFlow: https://drive.google.com/open?id=0Byy2AcGyEVxfSEc5UC1ROUFJdmM
    # Arguments
        nb_dense_block: number of dense blocks to add to end
        growth_rate: number of filters to add per dense block
        nb_filter: initial number of filters
        reduction: reduction factor of transition blocks.
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        classes: optional number of classes to classify images
        weights_path: path to pre-trained weights
    # Returns
        A Keras model instance.
    '''
    eps = 1.1e-5
    # img_rows = inputs.shape[1].value
    # img_cols = inputs.shape[2].value
    # color_type = inputs.shape[3].value
    # num_classes=None  

    # compute compression factor
    compression = 1.0 - reduction

    # Handle Dimension Ordering for different backends
    global concat_axis
    concat_axis = 3
    #img_input = Input(shape=(224, 224, 3), name='data')
    img_input = inputs 

    # From architecture for ImageNet (Table 1 in the paper)
    nb_filter = 64
    nb_layers = [6,12,32,32] # For DenseNet-169

    # Initial convolution
    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Conv2D(nb_filter, (7, 7), subsample=(2, 2), name='conv1', bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv1_bn')(x)
    x = Scale(axis=concat_axis, name='conv1_scale')(x)
    x = Activation('relu', name='relu1')(x)
    x = ZeroPadding2D((1, 1), name='pool1_zeropadding')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        stage = block_idx+2
        x, nb_filter = self.dense_block(x, stage, nb_layers[block_idx], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

        # Add transition_block
        x = self.transition_block(x, stage, nb_filter, compression=compression, dropout_rate=dropout_rate, weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)

    final_stage = stage + 1
    x, nb_filter = self.dense_block(x, final_stage, nb_layers[-1], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv'+str(final_stage)+'_blk_bn')(x)
    x = Scale(axis=concat_axis, name='conv'+str(final_stage)+'_blk_scale')(x)
    x = Activation('relu', name='relu'+str(final_stage)+'_blk')(x)

    x_fc = GlobalAveragePooling2D(name='pool'+str(final_stage))(x)
    x_fc = Dense(1000, name='fc6')(x_fc)
    x_fc = Activation('softmax', name='prob')(x_fc)

    model = Model(img_input, x_fc, name='densenet')
    return [model, x, final_stage]

  def build_model(self):
    inputs = Input(shape=self.x_shape, name='data')
    base_model, base_output, final_stage = self.build_densenet_model(inputs)
    # Use pre-trained weights for Tensorflow backend
    weights_path = os.path.join(self.weights_dir, #'densennet-imagenet-models'
                                'densenet169_weights_tf.h5')
    base_model.load_weights(weights_path, by_name=True)
    # loop over all layers in the base model and freeze them so they will
    # *not* be updated during the first training process
    #for layer in base_model.layers[-3]:
    #  layer.trainable = False
      
    if self.hps["wavelet"]: # add WaveletDeconv       
      print(base_output.shape) 
      w = base_output.shape[1].value 
      h = base_output.shape[2].value 
      z = base_output.shape[3].value 
      # TypeError: float() argument must be a string or a number, not 'Dimension' keras
      # Solution: https://blog.csdn.net/qq_40774175/article/details/105196387
      # When using Reshape, make sure the shape is using int values. 
      wd_layers = Reshape([w*h,z])(base_output)
      # Error: Variable is unhashable if Tensor equality is enabled
      #headModel = Lambda(lambda x: tf.reshape(x, [-1,7*7,512]))(headModel) 
      wd_layers = WaveletDeconvolution(
        5, kernel_length=500, padding='same', 
        input_shape=[w*h,z], data_format='channels_first')(wd_layers)
      wd_layers = Activation('tanh')(wd_layers)
      wd_layers = Conv2D(5, (3, 3), padding='same')(wd_layers)
      wd_layers = Activation('relu')(wd_layers)
      print(wd_layers.shape) 
      wd_layers = Lambda(lambda x: tf.reduce_min(x, 3))(wd_layers) 
      wd_layers = Reshape([w,h,z])(wd_layers)
      print(wd_layers.shape) 
      base_output = wd_layers
    # Truncate and replace softmax layer for transfer learning
    # Cannot use model.layers.pop() since model is not of Sequential() type
    # The method below works since pre-trained weights are stored in layers but not in the model
    head_model = GlobalAveragePooling2D(name='pool'+str(final_stage))(base_output)
    head_model = Dense(num_classes, name='fc6')(head_model)
    head_model = Activation('softmax', name='prob')(head_model)

    model = Model(inputs, head_model)   
    return model 

  def predict(self):
    x = self.x_test
    y = self.y_test
    predIdxs = self.model.predict(x, self.batch_size)
    #score = log_loss(y, predIdxs)

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
    #x_test  = self.x_test
    #x_train, x_test = self.normalize(x_train, x_test)
    x_val   = self.x_val

    #y_train = keras.utils.to_categorical(y_train, self.num_classes)
    #y_test = keras.utils.to_categorical(y_test, self.num_classes)
    y_train = self.y_train
    #y_test  = self.y_test
    y_val   = self.y_val

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
    #opt = Adam(lr=learning_rate, decay=lr_decay)
    opt = SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
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
      validation_data=(x_val, y_val),
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
      if "val_accuracy" in H.history:
        plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
      else:
        plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
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
  ap.add_argument("-net", "--network", type=int, default=201,
    help="DenseNet version, 121, 169, or 201")
  ap.add_argument("-ws", "--weights_dir", type=str, default="densenet-imagenet-models",                  
    help="directory containing pre-trained models")  
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
  ap.add_argument('--boc', dest='bias_off_crop', action='store_true')
  ap.add_argument('--no-boc', dest='bias_off_crop', action='store_false')
  ap.set_defaults(bias_off_crop=False)
  ap.add_argument('--wavelet', dest='wavelet', action='store_true')
  ap.add_argument('--no-wavelet', dest='wavelet', action='store_false')
  ap.set_defaults(wavelet=False)
  args = vars(ap.parse_args())
  
  in_size = 224
  
  # 2. Load images
  # grab the list of images in our dataset directory, then initialize
  # the list of data (i.e., images) and class images
  print("[INFO] loading images...")
  dataset_dir = args["dataset_dir"]
  images_dir  = os.path.join(dataset_dir, "Images-processed")
  #imagePaths = list(paths.list_images(dataset_dir))
  xray_images = [xray_image for xray_image in list(paths.list_images(images_dir)) \
              if ("_mask" not in xray_image \
                  and "_crop" not in xray_image \
                  and "_dilate" not in xray_image \
                  and "_predict" not in xray_image)]
  print("Found %d x-ray images in: %s " % (len(xray_images), dataset_dir))
    
  data = []
  labels = []
  # load train, validation and test samples
  data_split_dir = os.path.join(dataset_dir, "Data-split")
  
  def get_split_filenames(label, split_name):
    split_name_file = open(os.path.join(data_split_dir, label, "%sCT_%s.txt" % (split_name, label)))  
    split_filenames  = split_name_file.readlines()
    split_name_file.close()
    return [filename.replace("\n","") for filename in split_filenames]
  
  def get_split_data(label, split_name):
    split_images = []
    split_labels = []
    split_filenames = get_split_filenames(label, split_name)
    for xray_image_path in xray_images: # filename and label are matched
      if (os.path.basename(xray_image_path) in split_filenames \
          and label in os.path.dirname(xray_image_path)):
        split_images.append(read_and_crop_image(xray_image_path))
        split_labels.append(label)
      else: # 
        print("%s contains no %s" % (split_name, os.path.basename(xray_image_path)))
    return [split_images, split_labels]
  
  def read_and_crop_image(xray_image_path): 
    # load the image, swap color channels, and resize it to be a fixed
    # 224x224 pixels while ignoring aspect ratio
    if args["bias_off_crop"]:
      image = bias_off_crop(xray_image_path)
    else: # no crop
      image = cv2.imread(xray_image_path)
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (in_size, in_size))
    return image
  
  x_train = []
  y_train = []
  x_test  = []
  y_test  = []
  x_val   = []
  y_val   = []
  labels  = [os.path.basename(label_dir) for label_dir in os.listdir(data_split_dir) \
             if os.path.isdir(os.path.join(data_split_dir,label_dir))]
  print("Labels: %s " % str(labels))
  for i, label in enumerate(labels): 
    train_filenames = get_split_filenames(label, "train")
    test_filenames = get_split_filenames(label, "test")
    val_filenames = get_split_filenames(label, "val")
    #sub_xray_images = [xray_image_path for xray_image_path in xray_images \
    #                   if os.path.basename(os.path.dirname(xray_image_path)).startswith(label)]
    for xray_image_path in xray_images: # 
      sub_dir_name = os.path.basename(os.path.dirname(xray_image_path))
      #print(sub_dir_name)
      if sub_dir_name.startswith("CT_%s" % label):
        if os.path.basename(xray_image_path) in train_filenames:        
          x_train.append(read_and_crop_image(xray_image_path))
          y_train.append(label)
        elif os.path.basename(xray_image_path) in test_filenames:       
          x_test.append(read_and_crop_image(xray_image_path))
          y_test.append(label)
        elif os.path.basename(xray_image_path) in val_filenames:       
          x_val.append(read_and_crop_image(xray_image_path))
          y_val.append(label)
        else:
          print("!!! %s is nothing matched!!!" % os.path.basename(xray_image_path))
    print("Train: %d in %s" % (len(x_train),str(labels[:i+1])))
    print("Test: %d in %s" % (len(x_test),str(labels[:i+1])))
    print("Val: %d in %s" % (len(x_val),str(labels[:i+1])))

  # convert the data and labels to NumPy arrays while scaling the pixel
  # intensities to the range [0, 255]
  x_train = np.array(x_train) / 255.0
  x_test = np.array(x_test) / 255.0
  x_val = np.array(x_val) / 255.0
  #print("Train: %d in total" % len(x_train))
  #print("Test: %d in total" % len(x_test))
  #print("Val: %d in total" % len(x_val))
  #print(data)    
  # perform one-hot encoding on the labels
  #lb = LabelBinarizer()
  #labels = lb.fit_transform(labels)
  le = LabelEncoder()
  def labels_to_categorical(labels):
    labels = np.array(labels)
    labels = le.fit_transform(labels)
    labels = to_categorical(labels)  
    return labels
  
  y_train = labels_to_categorical(y_train) 
  y_test = labels_to_categorical(y_test) 
  y_val = labels_to_categorical(y_val) 
  #print(labels)
  #print(le.classes_)
  num_classes = len(le.classes_) # 2 or 3
  args["num_classes"] = num_classes

  # partition the data into training and testing splits using 80% of
  # the data for training and the remaining 20% for testing
  #(x_train, x_test, y_train, y_test) = train_test_split(data, labels,
  #  test_size=0.20, stratify=labels, random_state=42)
  #print(y_train)
  #print(y_test)
  args['x_train'] = x_train
  args['x_test']  = x_test
  args['x_val']   = x_val
  args['y_train'] = y_train
  args['y_test']  = y_test
  args['y_val']   = y_val
  print(y_train[:10,:])
  print(y_test[:10,:])

  # 3. Create and train the model
  model = DWTDenseNetCOVID19(args, True)

  # 4. Predict the result
  predicted_x = model.predict()

  # 5. Evaluate the result
  model.evaluate()

