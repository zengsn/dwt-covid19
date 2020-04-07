# coding=utf-8
# Copyright 2020 Shaoning Zeng
#
# Bias-Off Cropping for the COVID-19 X-Ray Images
# 
# Cite our paper: 

import os
import cv2 
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import patches 

N_SKIP = 3
SAVE_PROGRESS = False

def crop(in_image_path, out_shape=(224,224,3), predict_suff="_predict"):
  """
  Crop the input image using the predicted label / mask.
  
  # Arguments:
    in_image_path: the input image file path
    out_shape: the shape of the cropped image
    predict_suff: the suffix of predicted label or mask, 
      the default is _predict, e.g., sample01_predict.jpg.
  """
  assert out_shape[0] % 4 == 0, "The output width should be 4*n pixels!"
  assert out_shape[1] % 4 == 0, "The output height should be 4*n pixels!"
  
  filename = os.path.basename(in_image_path)
  print("Cropping the image: %s" % filename)
  filename, ext_name = os.path.splitext(filename) # filename.split(".")[-1]
  mask_filename = "%s%s%s" % (filename, predict_suff, ext_name)
  mask_image_path = os.path.join(os.path.dirname(in_image_path), mask_filename)
  assert os.path.isfile(mask_image_path), "Mask image %s not found!" % mask_filename
  print("Found the mask image: %s" % mask_filename)
  crop_image_filename = "%s_crop.jpg" % filename
  
  # Read the images
  in_image = cv2.imread(in_image_path)
  in_image = cv2.cvtColor(in_image, cv2.COLOR_BGR2RGB)
  #print("Input image shape is %s" % str(in_image.shape))
  min_size = in_image.shape[0]
  if min_size > in_image.shape[1]:
    min_size = in_image.shape[1]
  in_image = cv2.resize(in_image, (min_size,min_size))
  in_image_ori = in_image
  #print("Input image is resized to %s" % str(in_image.shape))
  mask_image = cv2.imread(mask_image_path)
  #mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
  mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
  #mask_image = cv2.resize(mask_image, in_image.shape[:-1])
  ret, mask_image=cv2.threshold(mask_image,127,255,cv2.THRESH_BINARY)
  print("Mask image shape is %s, %s" % (str(mask_image.shape), str(ret))) 
  
  # Scale rate between input and mask
  scale_h = int(in_image.shape[0] / mask_image.shape[0])
  scale_w = int(in_image.shape[1] / mask_image.shape[1])
  #print("Scale rate (h, w): (%d, %d)" % (scale_h, scale_w))
  scale_rate = scale_h # = scale_w
  
  if SAVE_PROGRESS:
    plt.subplot(1,2,1)
    plt.imshow(in_image,'gray')
    plt.title("Lungs")
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1,2,2)
    mask_image_s = cv2.resize(mask_image, None, fx=scale_h, fy=scale_w)
    plt.imshow(mask_image_s,'gray')
    plt.title("Mask")
    plt.xticks([])
    plt.yticks([])
    plt.savefig('fig1_Lungs_and_Mask.jpg', bbox_inches='tight')
    plt.show()
  
  # Crop the input image using the mask 
  n_skip = N_SKIP
  left = n_skip 
  for i in range(n_skip,mask_image.shape[1]): # left to right
    if left > n_skip:
      break
    for j in range(n_skip,mask_image.shape[0]): # top to bottom
      if mask_image[j,i] > 0:
        left = i
        break
  right = mask_image.shape[1]-n_skip  
  for i in range(mask_image.shape[1]-n_skip,0,-1): # right to left
    if right < mask_image.shape[1]-n_skip:
      break
    for j in range(n_skip,mask_image.shape[0]): # top to bottom
      if mask_image[j,i] > 0:
        right = i
        break
  top = n_skip
  for i in range(n_skip,mask_image.shape[0]): # top to bottom
    if top > n_skip:
      break
    for j in range(n_skip,mask_image.shape[1]): # left to right
      if mask_image[i,j] > 0:
        top = i
        break
  bottom = mask_image.shape[0]-n_skip 
  for i in range(mask_image.shape[0]-n_skip,0,-1): # bottom to top
    if bottom < mask_image.shape[0]-n_skip:
      break
    for j in range(n_skip,mask_image.shape[1]): # left to right
      if mask_image[i,j] > 0:
        bottom = i
        break 
  
  # Locate the center point
  start_row = 10
  # - skip some rows
  num_good_rows = 0
  for i in range(start_row,bottom): # from top to bottom
    # go to 00011111110000 area
    num_div = 1
    first_255 = 0
    last_255 = 0
    last_p = -1    
    for j in range(left, right): # from left to right
      if mask_image[i,j]==255:
        last_255 = j
        if first_255 == 0:
          first_255 = j 
      if last_p != mask_image[i,j]: # change
        if last_p > -1: # new divide
          num_div += 1
        last_p = mask_image[i,j]
    # check if match 3 divisions
    if num_div==3 and last_255>left+0.5*(right-left):
      start_row = i
      num_good_rows += 1
      if num_good_rows == n_skip: 
        break # make sure 
    else: #
      num_good_rows = 0
  #print("start_row = %d" % start_row)
  # - start to match the center point
#   center_x = 0
#   center_y = start_row  
#   for i in range(start_row, bottom): # from top to bottom
#     if center_x>0 and center_y>start_row: # found
#       break
#     # find the center point 0001111110111111100000
#     met_255 = False 
#     met_255_0 = False
#     for j in range(left, right): # from left to right
#       if mask_image[i,j]==255:
#         if not met_255: # not met 255 before,
#           met_255 = True # mark we got 1 already
#         elif met_255_0: # met both 255 and 0
#           # restrict center_x around center of image
#           if abs(j-(left+0.5*(right-left)))<0.1*mask_image.shape[1]:
#             # otherwise, segmentation is not good 
#             center_x = j # this is the center we want
#             center_y = i # save it 
#             break # this is where we want
#       if mask_image[i,j]==0 and met_255 and not met_255_0: # meet 0 after 255
#         met_255_0 = True   
  # Find the column containing the fewest 255
  min_total_255 = mask_image.shape[0]
  start_x = int(left+0.5*(right-left)-0.2*mask_image.shape[1])
  end_x   = int(left+0.5*(right-left)+0.2*mask_image.shape[1])
  min_total_255_x = start_x 
  for i in range(start_x,end_x):
    unique, counts = np.unique(mask_image[:,i], return_counts=True)
    assert len(unique) == 2 # only 0 and 255
    # dict(zip(unique, counts))
    total_255 = counts[1]
    if total_255 < min_total_255: 
      min_total_255 = total_255
      min_total_255_x = i
  center_x = min_total_255_x
  center_y = bottom
  for i in range(bottom,top,-1):
    if mask_image[i,min_total_255_x]==255:
      center_y = i
      break
  
  # Print the results
  print("top:%d, bottom:%d, left:%d, right:%d, center_x:%d, center_y:%d"  % \
        (top, bottom, left, right, center_x, center_y))      
  assert (center_x>0 and center_y>start_row)
  # x - columns, y - rows
  
  # Split image to left and right based on the center (x,y)
  mask_left  = mask_image[:,left:center_x] 
  mask_right = mask_image[:,center_x:right]
  image_left  = in_image[:,left*scale_w:center_x*scale_w,:] 
  image_right = in_image[:,center_x*scale_w:right*scale_w,:]
  
  # Resize left and right to the same size
  h_mask = mask_left.shape[0]
  w_mask = mask_left.shape[1]
  if w_mask < mask_right.shape[1]: # left is smaller
    mask_right = cv2.resize(mask_right, (w_mask, h_mask))
    image_left = cv2.resize(image_left, (w_mask*scale_w, h_mask*scale_h))
    image_right = cv2.resize(image_right, (w_mask*scale_w, h_mask*scale_h))
  else: # right is smaller
    w_mask = mask_right.shape[1]
    mask_left = cv2.resize(mask_left, (w_mask, h_mask))
    image_left = cv2.resize(image_left, (w_mask*scale_w, h_mask*scale_h))
    image_right = cv2.resize(image_right, (w_mask*scale_w, h_mask*scale_h))
  
  if SAVE_PROGRESS:
    plt.subplot(1,4,1)
    mask_left_s = cv2.resize(mask_left, None, fx=scale_w, fy=scale_h)
    plt.imshow(mask_left_s,'gray')
    plt.title("Left Mask")
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1,4,2)
    mask_right_s = cv2.resize(mask_right, None, fx=scale_w, fy=scale_h)
    plt.imshow(mask_right_s,'gray')
    plt.title("Right Mask")
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1,4,3)
    plt.imshow(image_left,'gray')
    plt.title("Left Lung")
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1,4,4)
    plt.imshow(image_right,'gray')
    plt.title("Right Lung")
    plt.xticks([])
    plt.yticks([])
    plt.savefig('fig2_Lungs_Left_Right_Split.jpg', bbox_inches='tight')
    plt.show()
  
  # Split image to A, B, C, D
  center_v = top + int(np.ceil(0.5*(bottom-top)))
  A_mask  = mask_left[top:center_v,:]
  B_mask = mask_right[top:center_v,:]
  C_mask  = mask_left[center_v:bottom,:]
  D_mask = mask_right[center_v:bottom,:]
  
  A_in  = image_left[top*scale_h:center_v*scale_h,:,:]
  B_in = image_right[top*scale_h:center_v*scale_h,:,:]
  C_in  = image_left[center_v*scale_h:bottom*scale_h,:,:]
  D_in = image_right[center_v*scale_h:bottom*scale_h,:,:]
  
  if SAVE_PROGRESS:
    h_in = A_in.shape[0]
    w_in = A_in.shape[1]
    plt.subplot(2,4,1)
    A_mask_s = cv2.resize(A_mask, (w_in, h_in))
    plt.imshow(A_mask_s,'gray')
    plt.title("A Mask")
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2,4,2)
    B_mask_s = cv2.resize(B_mask, (w_in, h_in));
    plt.imshow(B_mask_s,'gray')
    plt.title("B Mask")
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2,4,3)
    plt.imshow(A_in,'gray')
    plt.title("A Area")
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2,4,4)
    plt.imshow(B_in,'gray')
    plt.title("B Area")
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2,4,5)
    C_mask_s = cv2.resize(C_mask, (w_in, h_in));
    plt.imshow(C_mask_s,'gray')
    plt.title("C Mask")
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2,4,6)
    D_mask_s = cv2.resize(D_mask, (w_in, h_in));
    plt.imshow(D_mask_s,'gray')
    plt.title("D Mask")
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2,4,7)
    plt.imshow(C_in,'gray')
    plt.title("C Area")
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2,4,8)
    plt.imshow(D_in,'gray')
    plt.title("D Area")
    plt.xticks([])
    plt.yticks([])
    plt.savefig('fig3_Lungs_ABCD_Areas_Ori.jpg', bbox_inches='tight')
    plt.show()    
  
  # Trim them
  s = scale_rate
  A_tl, A_br = trim_rectangle(A_mask)
  A          = A_in[A_tl[0]*s:A_br[0]*s,A_tl[1]*s:A_br[1]*s,:]
  B_tl, B_br = trim_rectangle(B_mask, False)# mirror it
  row_s      = B_in.shape[0]-B_br[0]*s # start row
  row_e      = B_in.shape[0]-B_tl[0]*s # end row
  col_s      = B_in.shape[1]-B_br[1]*s # start column
  col_e      = B_in.shape[1]-B_tl[1]*s # end column
  B          = B_in[row_s:row_e,col_s:col_e,:]
  C_tl, C_br = trim_rectangle(C_mask)
  C          = C_in[C_tl[0]*s:C_br[0]*s,C_tl[1]*s:C_br[1]*s,:]
  D_tl, D_br = trim_rectangle(D_mask, False) # mirror it
  row_s      = D_in.shape[0]-D_br[0]*s # start row
  row_e      = D_in.shape[0]-D_tl[0]*s # end row
  col_s      = D_in.shape[1]-D_br[1]*s # start column
  col_e      = D_in.shape[1]-D_tl[1]*s # end column
  D          = D_in[row_s:row_e,col_s:col_e,:]
  
  if SAVE_PROGRESS:
    h_in = A.shape[0]
    w_in = A.shape[1]
    # Mask A
    ax = plt.subplot(2,4,1)
    A_mask_s = cv2.resize(A_mask, None, fx=scale_w, fy=scale_h)
    plt.imshow(A_mask_s,'gray')
    rect = patches.Rectangle(
      (A_tl[0]*scale_w,A_tl[1]*scale_h),(A_br[1]-A_tl[1])*scale_w,(A_br[0]-A_tl[0])*scale_h,
      linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    plt.title("A Mask")
    plt.xticks([])
    plt.yticks([])
    # Mask B
    ax = plt.subplot(2,4,2)
    B_mask_s = cv2.resize(B_mask, None, fx=scale_w, fy=scale_h);
    plt.imshow(B_mask_s,'gray')
    rect = patches.Rectangle(
      (B_tl[0]*scale_w,B_tl[1]*scale_h),(B_br[1]-B_tl[1])*scale_w,(B_br[0]-B_tl[0])*scale_h,
      linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    plt.title("B Mask")
    plt.xticks([])
    plt.yticks([])
    # Area A
    plt.subplot(2,4,3)
    plt.imshow(A,'gray')
    plt.title("A Area")
    plt.xticks([])
    plt.yticks([])
    # Area B
    plt.subplot(2,4,4)
    plt.imshow(B,'gray')
    plt.title("B Area")
    plt.xticks([])
    plt.yticks([])
    # Mask C
    ax = plt.subplot(2,4,5)
    C_mask_s = cv2.resize(C_mask, None, fx=scale_w, fy=scale_h);
    plt.imshow(C_mask_s,'gray')
    rect = patches.Rectangle(
      (C_tl[0]*scale_w,C_tl[1]*scale_h),(C_br[1]-C_tl[1])*scale_w,(C_br[0]-C_tl[0])*scale_h,
      linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    plt.title("C Mask")
    plt.xticks([])
    plt.yticks([])
    # Mask D
    ax = plt.subplot(2,4,6)
    D_mask_s = cv2.resize(D_mask, None, fx=scale_w, fy=scale_h);
    plt.imshow(D_mask_s,'gray')
    rect = patches.Rectangle(
      (D_tl[0]*scale_w,D_tl[1]*scale_h),(D_br[1]-D_tl[1])*scale_w,(D_br[0]-D_tl[0])*scale_h,
      linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    plt.title("D Mask")
    plt.xticks([])
    plt.yticks([])
    # Area C
    plt.subplot(2,4,7)
    plt.imshow(C,'gray')
    plt.title("C Area")
    plt.xticks([])
    plt.yticks([])
    # Area D
    plt.subplot(2,4,8)
    plt.imshow(D,'gray')
    plt.title("D Area")
    plt.xticks([])
    plt.yticks([])
    plt.savefig('fig4_Lungs_ABCD_Areas_Trim.jpg', bbox_inches='tight')
    plt.show()   
  
  # Resize them to the smallest rectangle
  h, w = get_smallest_size(A, B, C, D)
  #print("Resized to smallest size (%d, %d)" % (h, w))
  A = cv2.resize(A, (w,h))
  B = cv2.resize(B, (w,h))
  C = cv2.resize(C, (w,h))
  D = cv2.resize(D, (w,h))
  
  # Compose them to one image
  crop_image = np.zeros([h*2,w*2,3])
  #print("Composed image shape: %s" % str(crop_image.shape))
  crop_image[0:1*h,0:1*w,:] = A
  crop_image[0:1*h,w:2*w,:] = B
  crop_image[h:2*h,0:1*w,:] = C
  crop_image[h:2*h,w:2*w,:] = D
  crop_image = cv2.resize(crop_image, (in_image_ori.shape[1],in_image_ori.shape[0]))
  
  if SAVE_PROGRESS: 
    #plt.subplot(1,2,1)
    #plt.imshow(in_image_ori)
    #plt.title("Original Image")
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1,1,1)
    plt.imshow(crop_image/255)
    plt.title("Cropped Image")
    plt.xticks([])
    plt.yticks([])
    plt.savefig('fig5_Lungs_Crop_Image.jpg', bbox_inches='tight')
    plt.show()    
  
  # Resize to output shape
  crop_image = cv2.resize(crop_image, out_shape[:-1])
  cv2.imwrite(os.path.join(os.path.dirname(in_image_path), crop_image_filename), crop_image)
  print("Cropped image saved to: %s" % crop_image_filename)
  
  return crop_image
  
def trim_rectangle(mask, left_right=True):
  h = mask.shape[0]
  w = mask.shape[1]
  top_left = [N_SKIP, N_SKIP]
  bottom_right = [h-N_SKIP, w-N_SKIP]
  if left_right:
    for i in range(N_SKIP,h-N_SKIP):
      if mask[i,i]==255:
        top_left = [i,i]
        break
    for i in range(N_SKIP,h-N_SKIP):
      if mask[h-N_SKIP-i,w-N_SKIP-i]==255:
        bottom_right = [h-N_SKIP-i,w-N_SKIP-i]
        break
  else: # right to left
    top_right = [h-N_SKIP,N_SKIP]
    for i in range(N_SKIP,h-N_SKIP):
      if mask[i,w-N_SKIP-i]==255:
        top_right = [h-N_SKIP-i,i]
        break
    bottom_left = [N_SKIP, w-N_SKIP]
    for i in range(N_SKIP,h-N_SKIP):
      if mask[h-N_SKIP-i,i]==255:
        bottom_left = [i,w-N_SKIP-i]
        break
    top_left = [bottom_left[0],top_right[1]]
    bottom_right = [top_right[0],bottom_left[1]]
  
  tl = top_left
  br = bottom_right
  
  return [tl, br]

def get_smallest_size(A, B, C, D):
  h = np.array([A.shape[0],B.shape[0],C.shape[0],D.shape[0]])
  w = np.array([A.shape[1],B.shape[1],C.shape[1],D.shape[1]])
  return [h.min(), w.min()]
  
  