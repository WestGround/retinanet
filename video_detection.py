import sys
sys.path.insert(0, '../')


# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

# adjust this to point to your downloaded/trained model
model_path = "/content/retinanet/prediction/resnet50_csv_1.h5"
model = models.load_model(model_path, backbone_name='resnet50')
labels_to_names = {0: 'warship', 1: 'boat', 2: 'wood boat', 3: 'ferry', 4: 'cargo ship'}

video_path = '/mydrive/images/cargo.mp4'  ## replace with input video path
output_path = '/mydrive/images/cargo_res_retina.mp4' ## replace with path where you want to save the output
fps = 15

vcapture = cv2.VideoCapture(video_path)
width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))  # uses given video width and height
height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
vwriter = cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc(*'mp4v'),fps, (width, height)) #
num_frames = int(vcapture.get(cv2.CAP_PROP_FRAME_COUNT))

def run_detection_video(video_path):
    count = 0
    success = True
    start = time.time()
    while success:
        if count % 100 == 0:
            print("frame: ", count)
        count += 1
        # Read next image
        success, image = vcapture.read()

        if success:
            draw = image.copy()
            draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

            image = preprocess_image(image)
            image, scale = resize_image(image)

            boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
            boxes /= scale
            for box, score, label in zip(boxes[0], scores[0], labels[0]):
                # scores are sorted so we can break
                if score < 0.4:
                    break

                color = label_color(label)

                b = box.astype(int)
                draw_box(draw, b, color=color)

                caption = "{} {:.3f}".format(labels_to_names[label], score)
                draw_caption(draw, b, caption)
            detected_frame = cv2.cvtColor(draw, cv2.COLOR_RGB2BGR)
            vwriter.write(detected_frame)  # overwrites video slice

    vcapture.release()
    vwriter.release()  #
    end = time.time()

    print("Total Time: ", end - start)

run_detection_video(video_path)           
