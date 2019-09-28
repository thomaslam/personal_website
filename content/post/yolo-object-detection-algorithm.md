---
title: "YOLO: A state-of-the-art Object Detection Algorithm"
date: 2019-09-28T14:41:01-04:00
draft: true
---

Object detection is an important computer vision technique that aims to classify multiple real-world objects as well as find bounding boxes for these objects in a single image (whether static or a single frame from a video feed). Applications of this technique are wide-ranging: self-driving cars

An important algorithm allowing autonomous cars to detect different objects based on real-time image outputs from their front-facing cameras is called YOLO (or You Only Look Once)

Unlike other deep neural network-based object detection methods which apply different localizer and classifier models to the image of interest at multiple locations and scales, YOLO uses only 1 large neural network and applies it to the whole image. In other words, 