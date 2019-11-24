---
title: "YOLO: A state-of-the-art Object Detection Algorithm"
date: 2019-09-28T14:41:01-04:00
draft: true
---

Object detection is an important computer vision technique that aims to classify multiple real-world objects as well as find bounding boxes for these objects in a single image (whether static or a single frame from a video feed). 

An important application of this technique is in self-driving technology, which has prompted various research teams to come up with numerous competing object detection models. One of the most successful out of all these models is called <mark>YOLO</mark> (You Only Look Once).

Unlike other deep neural network-based object detection methods which apply different localizer and classifier models to the image of interest at multiple locations and scales, YOLO uses only 1 large neural network and applies it to the whole image. In other words, the image of interest is fed to the trained YOLO model in only 1 forward pass and the model outputs all the bounding boxes containing what it thinks are objects in the image and the predicted classes for each box (as well as their confidence scores)

[A single frame from front facing video feed from an autonomous car with bounding boxes and predicted object classes]()

# 