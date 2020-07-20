'''
# USAGE
COVID-CT-old
python sample_random.py --input COVID-CT-old/CT_NonCOVID --output Dataset/ct/old/dataset_264_2/ncovid --sample 264
python sample_random.py --input COVID-CT-old/COVID-CT-Images/positive --output Dataset/ct/old/dataset_264_2/covid

COVID-CT-new
python sample_random.py --input COVID-CT-new/Images-processed/CT_NonCOVID --output Dataset/ct/new/dataset_349_2/ncovid --sample 349
python sample_random.py --input COVID-CT-new/Images-processed/CT_COVID --output Dataset/ct/new/dataset_349_2/covid 
'''

# import the necessary packages
from imutils import paths
import argparse
import random
import shutil
import os

from PIL import Image
import filetype
import imageio
import pydicom

def pic_2_png(imagePath, outputPath):
  def dcm_2_png(dcm, png):
    ds = pydicom.read_file(dcm)
    img = ds.pixel_array
    imageio.imwrite(png, img)
  def jpg_2_png(jpg, png):
    img = Image.open(jpg)
    img.save(png)
  def type_guess(file_path):
    kind = filetype.guess(file_path)
    if kind == None:
      return None
    return kind.extension
  outputPath = os.path.splitext(outputPath)[0] + '.png'
  ext = type_guess(imagePath)
  if ext == 'dcm':
    dcm_2_png(imagePath, outputPath)
  elif ext == 'jpg':
    jpg_2_png(imagePath, outputPath)
  elif ext == 'png':
    shutil.copy2(imagePath, outputPath)
  return None

if __name__ == '__main__':
  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument("-i", "--input", required=True,
      help="path to directory where input images will be selected")
  ap.add_argument("-o", "--output", required=True,
      help="path to directory where output images will be stored")
  ap.add_argument("-s", "--sample", type=int, default=0,
      help="# of samples to pull from input dataset")
  args = vars(ap.parse_args())

  # grab all image paths from the input CT dataset
  inputPaths = list(paths.list_images(args["input"]))
  outputBase = args["output"]
  if not os.path.exists(outputBase):
    os.makedirs(outputBase)

  # randomly sample the image paths, presumely enough.
  random.seed(44)
  random.shuffle(inputPaths)
  sample = args["sample"] if args["sample"] > 0 and args["sample"] <= len(inputPaths) else len(inputPaths)
  imagePaths = inputPaths[:sample]

  # loop over the image paths
  for imagePath in imagePaths:
	  # extract the filename from the image path and then construct the
	  # path to the copied image file
	  filename = imagePath.split(os.path.sep)[-1]
	  outputPath = os.path.sep.join([outputBase, filename])
	  # copy the image
	  pic_2_png(imagePath, outputPath)
  print('%d sample(s) has copied from %s/ to %s/' % (sample, args['input'], args['output']))
