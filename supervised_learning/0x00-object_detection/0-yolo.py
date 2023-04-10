#!/usr/bin/env python3
"""
Implementation of Yolo V3 Algorithm
"""
import numpy as np
import tensorflow.keras as K


class Yolo:
    """
    Initialize Yolo class.
    """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Class Constructor
        model_path: path to where Darnket Keras model is stored
        classes_path: path to list of class names used for Darknet model
        class_t: float representing box score threshold for initial filtering
        nms_t: float representing the IOU threshold for non-max suppression
        anchors: numpy.ndarray of shape (outputs, anchor_boxes, 2) containing
        the anchor boxes
            outputs: number of outputs made by the Darknet model
            anchor_boxes: number of anchor boxes used for each prediction
            2: [anchor_box_width, anchor_box_height]

        Public Instance Attributes
            model: Darknet Keras Model
            class_names: list of the class names for the model
            class_t: box score threshold for initial filtering
            nms_t: IOU threshold for non-max suppression
            anchors: the anchor boxes
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path) as file:
            class_names = file.read()
        self.class_names = class_names.replace("\n", "|").split("|")[:-1]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
