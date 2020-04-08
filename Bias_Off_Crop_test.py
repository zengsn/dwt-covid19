# coding=utf-8
# Copyright 2020 Shaoning Zeng
#
# Bias-Off Cropping for the COVID-19 X-Ray Images
# 
# Cite our paper: 

import os
import numpy as np
import Bias_Off_Crop

def test_get_smallest_size(): 
  A = np.zeros((10,20,3))
  B = np.zeros((10,40,3))
  C = np.zeros((32,20,3))
  D = np.zeros((20,10,3))
  h, w = Bias_Off_Crop.get_smallest_size(A, B, C, D)
  assert(h==10)
  assert(w==10)
  print("test_get_smallest_size(): passed.")
  
def test_crop(in_image_path):
  Bias_Off_Crop.crop(in_image_path,save_crop=True,save_progress=True)
  filename, ext_name = os.path.splitext(in_image_path)
  crop_image_path = os.path.join(
    os.path.dirname(in_image_path), "%s_crop%s" % (filename, ext_name))
  print("test_crop(): %s" % crop_image_path)
  assert os.path.exists(crop_image_path)
  print("test_crop(): passed.")
  
if __name__ == '__main__':
  """
  Test the function.
  """
  
  data_dir = "/Volumes/SanDisk256B/Lab0_Data_Keras/covid-19-predict/dataset300/"
  #test_image_path = os.path.join(data_dir, "covid", "covid-19-pneumonia-2.jpg")
  #test_image_path = os.path.join(data_dir, "covid", "1-s2.0-S0140673620303706-fx1_lrg.jpg")
  #test_image_path = os.path.join(data_dir, "covid", "covid-19-infection-exclusive-gastrointestinal-symptoms-l.png")
  #test_image_path = os.path.join(data_dir, "covid", "covid-19-pneumonia-7-PA.jpg")
  #test_image_path = os.path.join(data_dir, "normal", "NORMAL2-IM-0753-0001.jpeg")
  #test_image_path = os.path.join(data_dir, "pneumnia", "person1306_bacteria_3272.jpeg")
  #test_image_path = os.path.join(data_dir, "pneumnia", "person1343_bacteria_3411.jpeg")
  #test_image_path = os.path.join(data_dir, "pneumnia", "person1619_bacteria_4269.jpeg")
  test_image_path = os.path.join(data_dir, "pneumnia", "person253_bacteria_1156.jpeg")
  data_dir = "/Volumes/SanDisk256B/Lab0_Data_Keras/COVID-CT/Images-processed/"
  test_image_path = os.path.join(data_dir, "CT_COVID_unet_seg", \
                                 "2020.02.23.20026856-p17-115%2.png")
  test_image_path = os.path.join(data_dir, "CT_COVID_unet_seg", \
                                 "2020.02.24.20027052-p8-73%0.png")
  test_image_path = os.path.join(data_dir, "CT_NonCOVID_unet_seg", \
                                 "103.png")
  assert os.path.isfile(test_image_path), "File not found!"
  
  # Test test_get_smallest_size()
  test_get_smallest_size()
  
  # 
  test_crop(test_image_path)
  
  