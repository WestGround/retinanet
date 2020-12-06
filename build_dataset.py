# import the necessary packages
from config import custom_dataset_config as config
from bs4 import BeautifulSoup
from imutils import paths
import argparse
import random
import os

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-A", "--trannotations", default=config.TRAIN_ANNOT_PATH,
    help="path to train annotations")
ap.add_argument("-I", "--trimages", default=config.TRAIN_IMAGES_PATH,
	help="path to train images")
ap.add_argument("-a", "--testannotations", default=config.TEST_ANNOT_PATH,
  help="path to test annotations")
ap.add_argument("-i", "--testimages", default=config.TEST_IMAGES_PATH,
	help="path to test images")
ap.add_argument("-t", "--train", default=config.TRAIN_CSV,
	help="path to output training CSV file")
ap.add_argument("-e", "--test", default=config.TEST_CSV,
	help="path to output test CSV file")
ap.add_argument("-c", "--classes", default=config.CLASSES_CSV,
	help="path to output classes CSV file")
args = vars(ap.parse_args())

# Create easy variable names for all the arguments
train_annot_path = args["trannotations"]
train_images_path = args["trimages"]
test_annot_path = args["testannotations"]
test_images_path = args["testimages"]
train_csv = args["train"]
test_csv = args["test"]
classes_csv = args["classes"]

trainImagePaths = list(paths.list_files(train_images_path))
testImagePaths = list(paths.list_files(test_images_path))

# create the list of datasets to build
dataset = [ ("train", trainImagePaths, train_annot_path, train_csv),
            ("test", testImagePaths, test_annot_path, test_csv)]

# initialize the set of classes we have
CLASSES = set()

# loop over the datasets
for (dType, imagePaths, annot_path, outputCSV) in dataset:
    # load the contents
    print ("[INFO] creating '{}' set...".format(dType))
    print ("[INFO] {} total images in '{}' set".format(len(imagePaths), dType))

    # open the output CSV file
    csv = open(outputCSV, "w")

    # loop over the image paths
    for imagePath in imagePaths:
        # build the corresponding annotation path
        fname = imagePath.split(os.path.sep)[-1]
        fname = "{}.xml".format(fname[:fname.rfind(".")])
        annotPath = os.path.sep.join([annot_path, fname])

        # load the contents of the annotation file and buid the soup
        contents = open(annotPath).read()
        soup = BeautifulSoup(contents, "html.parser")

        # extract the image dimensions
        w = int(soup.find("width").string)
        h = int(soup.find("height").string)

        # loop over all object elements
        for o in soup.find_all("object"):
            #extract the label and bounding box coordinates
            label = o.find("name").string
            xMin = int(float(o.find("xmin").string))
            yMin = int(float(o.find("ymin").string))
            xMax = int(float(o.find("xmax").string))
            yMax = int(float(o.find("ymax").string))

            # truncate any bounding box coordinates that fall outside
            # the boundaries of the image
            xMin = max(0, xMin)
            yMin = max(0, yMin)
            xMax = min(w, xMax)
            yMax = min(h, yMax)

            # ignore the bounding boxes where the minimum values are larger
            # than the maximum values and vice-versa due to annotation errors
            if xMin >= xMax or yMin >= yMax:
                continue
            elif xMax <= xMin or yMax <= yMin:
                continue

            # write the image path, bb coordinates, label to the output CSV
            row = [os.path.abspath(imagePath),str(xMin), str(yMin), str(xMax),
                    str(yMax), str(label)]
            csv.write("{}\n".format(",".join(row)))

            # update the set of unique class labels
            CLASSES.add(label)

    # close the CSV file
    csv.close()

# write the classes to file
print("[INFO] writing classes...")
csv = open(classes_csv, "w")
rows = [",".join([c, str(i)]) for (i,c) in enumerate(CLASSES)]
csv.write("\n".join(rows))
csv.close()
