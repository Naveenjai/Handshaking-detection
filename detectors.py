import os
import cv2
import numpy as np
#import face_recognition
# https://www.cvlib.net/
#import cvlib as cv
from utils import detector_utils as detector_utils
# pip3 install --upgrade https://storage.googleapis.com/tensorflow/windows/cpu/tensorflow_cpu-2.1.0-cp37-cp37m-win_amd64.whl
#import tensorflow as tf
from shapely.geometry import Polygon

############################################################################
class Detector:
    detector_params = {}
    detector = None

    def __init__(self):
        pass

    def set_detector_params(self, params):
        self.detector_params = params

    def detect(self):
        pass


############################################################################
class TSDetector(Detector):
    def __init__(self):
        self.detection_graph, self.sess = detector_utils.load_inference_graph()

    def detect(self, rgb_image):
        # returns (top [0], left [1], bottom [2], right [3])
        boxes, confidences = detector_utils.detect_objects(rgb_image, self.detection_graph, self.sess)

        im_height, im_width = rgb_image.shape[:2]

        detection_th = self.detector_params.get('detection_th', 0.5)
        objects = [(box[0] * im_height, box[3] * im_width, box[2] * im_height, box[1] * im_width) for box, score in zip(boxes, confidences) if score >= detection_th]
        # change to an array of (x, y, w, h)
        return [(int(left), int(top), int(right - left), int(bottom - top)) for (top, right, bottom, left) in objects]


############################################################################
def add_objects_to_image(img_, objects, color=(255, 0, 0)):
    img = np.copy(img_)
    for (x, y, w, h) in objects:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    return img

def obj_to_poly(obj):
    x, y, w, h = obj
    return Polygon([(x, y), (x+w, y), (x+w, y+h), (x, y+h)])

def objects_touch(handchec, hands):
      handchec_poly = obj_to_poly(handchec)
      for hand in hands:
        hand_poly = obj_to_poly(hand)
          if handchec_poly.intersects(hand_poly):
                return True
    return False

