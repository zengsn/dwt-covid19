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
  
def test_crop(in_image_path, crop_image_path):
  Bias_Off_Crop.crop(in_image_path)
  assert(os.path.isfile(crop_image_path))
  print("test_crop(): passed.")
  
if __name__ == '__main__':
  """
  Test the function.
  """
  
  data_dir = "/Volumes/SanDisk256B/Lab0_Data_Keras/covid-19-predict/dataset300/"
  #test_image_path = os.path.join(data_dir, "covid", "covid-19-pneumonia-2.jpg")
  test_image_path = os.path.join(data_dir, "covid", "1-s2.0-S0140673620303706-fx1_lrg.jpg")
  #test_image_path = os.path.join(data_dir, "covid", "covid-19-pneumonia-7-PA.jpg")
  #test_image_path = os.path.join(data_dir, "normal", "NORMAL2-IM-0753-0001.jpeg")
  assert os.path.isfile(test_image_path), "File not found!"
  
  # Test test_get_smallest_size()
  test_get_smallest_size()
  
  # 
  crop_image_path = os.path.join(data_dir, "covid", "covid-19-pneumonia-2_crop.jpg")
  test_crop(test_image_path, crop_image_path)
  
  