'''
# USAGE
# old: covid_pa only 99, we copy and rename 1 add it, so it becomes 100
# dataset145-->dataset_145_2
python build_chest_xray.py --csv covid-chestxray-dataset-old/metadata.csv --input  covid-chestxray-dataset-old/images --output Dataset/xray/old/dataset_145_2/covid --type covid
python build_chest_xray.py --csv rsna-pneumonia-detection-challenge/stage_2_detailed_class_info.csv --input  rsna-pneumonia-detection-challenge/stage_2_train_images --output Dataset/xray/old/dataset_145_2/ncovid --type ncovid --sample 145
# dataset200-->dataset_100_2
python build_chest_xray.py --csv covid-chestxray-dataset-old/metadata.csv --input  covid-chestxray-dataset-old/images --output Dataset/xray/old/dataset_100_2/covid --type covid_pa
python build_chest_xray.py --csv rsna-pneumonia-detection-challenge/stage_2_detailed_class_info.csv --input  rsna-pneumonia-detection-challenge/stage_2_train_images --output Dataset/xray/old/dataset_100_2/ncovid --type ncovid --sample 100
# dataset_3class-->dataset_100_3
python build_chest_xray.py --csv covid-chestxray-dataset-old/metadata.csv --input  covid-chestxray-dataset-old/images --output Dataset/xray/old/dataset_100_3/covid --type covid_pa
python build_chest_xray.py --csv rsna-pneumonia-detection-challenge/stage_2_detailed_class_info.csv --input  rsna-pneumonia-detection-challenge/stage_2_train_images --output Dataset/xray/old/dataset_100_3/normal --type normal --sample 100
python build_chest_xray.py --csv rsna-pneumonia-detection-challenge/stage_2_detailed_class_info.csv --input  rsna-pneumonia-detection-challenge/stage_2_train_images --output Dataset/xray/old/dataset_100_3/pneumnia --type lung_opacity --sample 100
# dataset400-->dataset_100_4
python build_chest_xray.py --csv covid-chestxray-dataset-old/metadata.csv --input  covid-chestxray-dataset-old/images --output Dataset/xray/old/dataset_100_4/covid --type covid_pa
python build_chest_xray.py --csv rsna-pneumonia-detection-challenge/stage_2_detailed_class_info.csv --input  rsna-pneumonia-detection-challenge/stage_2_train_images --output Dataset/xray/old/dataset_100_4/normal --type normal --sample 100
python build_chest_xray.py --csv rsna-pneumonia-detection-challenge/stage_2_detailed_class_info.csv --input  rsna-pneumonia-detection-challenge/stage_2_train_images --output Dataset/xray/old/dataset_100_4/pneumnia --type lung_opacity --sample 100
python build_chest_xray.py --csv rsna-pneumonia-detection-challenge/stage_2_detailed_class_info.csv --input  rsna-pneumonia-detection-challenge/stage_2_train_images --output Dataset/xray/old/dataset_100_4/other --type other --sample 100

# new:
# dataset_204_2
python build_chest_xray.py --csv covid-chestxray-dataset-new/metadata.csv --input  covid-chestxray-dataset-new/images --output Dataset/xray/new/dataset_204_2/covid --type covid_pa
python build_chest_xray.py --csv rsna-pneumonia-detection-challenge/stage_2_detailed_class_info.csv --input  rsna-pneumonia-detection-challenge/stage_2_train_images --output Dataset/xray/new/dataset_204_2/ncovid --type ncovid --sample 204

# dataset_464_2
python build_chest_xray.py --csv covid-chestxray-dataset-new/metadata.csv --input  covid-chestxray-dataset-new/images --output Dataset/xray/new/dataset_464_2/covid --type covid
python build_chest_xray.py --csv rsna-pneumonia-detection-challenge/stage_2_detailed_class_info.csv --input  rsna-pneumonia-detection-challenge/stage_2_train_images --output Dataset/xray/new/dataset_464_2/ncovid --type ncovid --sample 464

# dataset_204_3
python build_chest_xray.py --csv covid-chestxray-dataset-new/metadata.csv --input  covid-chestxray-dataset-new/images --output Dataset/xray/new/dataset_204_3/covid --type covid_pa
python build_chest_xray.py --csv rsna-pneumonia-detection-challenge/stage_2_detailed_class_info.csv --input  rsna-pneumonia-detection-challenge/stage_2_train_images --output Dataset/xray/new/dataset_204_3/normal --type normal --sample 204
python build_chest_xray.py --csv rsna-pneumonia-detection-challenge/stage_2_detailed_class_info.csv --input  rsna-pneumonia-detection-challenge/stage_2_train_images --output Dataset/xray/new/dataset_204_3/pneumnia --type lung_opacity --sample 204

# dataset_464_3
python build_chest_xray.py --csv covid-chestxray-dataset-new/metadata.csv --input  covid-chestxray-dataset-new/images --output Dataset/xray/new/dataset_464_3/covid --type covid
python build_chest_xray.py --csv rsna-pneumonia-detection-challenge/stage_2_detailed_class_info.csv --input  rsna-pneumonia-detection-challenge/stage_2_train_images --output Dataset/xray/new/dataset_464_3/normal --type normal --sample 464
python build_chest_xray.py --csv rsna-pneumonia-detection-challenge/stage_2_detailed_class_info.csv --input  rsna-pneumonia-detection-challenge/stage_2_train_images --output Dataset/xray/new/dataset_464_3/pneumnia --type lung_opacity --sample 464

'''
import pandas as pd
import argparse
import shutil
import random
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

if __name__ == "__main__":
  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument("-c", "--csv", required=True,
      help="path to stage_2_detailed_class_info.csv file")
  ap.add_argument("-in", "--input", required=True,
      help="path to base directory for input dataset")
  ap.add_argument("-o", "--output", required=True,
      help="path to directory where noamal images will be stored")
  ap.add_argument("-t", "--type", required=True,
      help="type to the image file")
  ap.add_argument("-s", "--sample", type=int, default=0,
      help="# of samples to pull from input dataset")
  args = vars(ap.parse_args())

  if not os.path.exists(args['output']):
    os.makedirs(args['output'])
  # constract a map from args["type"] to the original type
  original_type = {
			"normal" 	: "Normal",\
			"lung_opacity"	: "Lung Opacity",\
			"other"		: "No Lung Opacity / Not Normal"}
  # construct the path to csv file and load it
  csvPath = args["csv"]
  df = pd.read_csv(csvPath)

  # create list L for suffling
  pairPaths = []

  # loop over the rows of the input data frame
  for (i, row) in df.iterrows():
    # choose the image type
    if args["type"] == "covid_pa":
      if row["finding"] != "COVID-19" or row["view"] != "PA":
        continue
      filename = row["filename"]
    elif args["type"] == "covid":
      if row["finding"] != "COVID-19" or row["modality"] != "X-ray":
        continue
      filename = row["filename"]
    else:
      if args["type"] != "ncovid" and row["class"] != original_type[args["type"]]:
        continue
      suffix = ".dcm"
      filename = row["patientId"] + suffix

    # build the path to the input image file
    imagePath = os.path.sep.join([args["input"], filename])

    # if the input image file does not exist (there are some errors in
    # the csv file or downloading uncomplete), ignore the row
    if not os.path.exists(imagePath):
      print(imagePath, "not exists.")
      continue

    # construct the path to the copied image file
    outputPath = os.path.sep.join([args["output"], filename])

    # add the pairPaths
    pairPaths.append((imagePath, outputPath))

  # shuffle the list pairPaths, and store the images
  pairPaths = list(set(pairPaths))
  pairPaths.sort()
  random.seed(44)
  random.shuffle(pairPaths)
  sample = args["sample"] if args["sample"] > 0 and args["sample"] <= len(pairPaths) else len(pairPaths)
  pairPaths = pairPaths[:sample]
  for (imagePath, outputPath) in pairPaths:
    pic_2_png(imagePath, outputPath)
  print('%d sample(s) has copied from %s/ to %s/' % (sample, args['input'], args['output']))

