import cv2 as cv
import numpy as np
import tensorflow as tf

def partition(img, target_size=(64,64)):
  """
  Partitions the date such that each character in the date becomes a separate image.
  This preprocessing turns the problem into a simple digit classification problem.
  Partitioning is done by thresholding, dilation, then finding the bounding box for each character
  """
  _,thresh = cv.threshold(img,100,255,cv.THRESH_BINARY_INV)
  kernel = np.ones((2, 2), np.uint8)
  dilated = cv.dilate(thresh, kernel, iterations=1)
  contours, _ = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

  boxes = []
  for i in range(len(contours)):
    x,y,w,h = cv.boundingRect(contours[i])
    boxes.append([x,y,w,h])

  boxes.sort(key=lambda b: b[0])
  partitions = []
  for box in boxes:
      x,y,w,h = box
      bb = thresh[y:y+h, x:x+w]
      partitions.append(cv.resize(bb, target_size, interpolation = cv.INTER_LINEAR))

  return tf.expand_dims(tf.constant(partitions, dtype=tf.float32) / 255.0, -1)