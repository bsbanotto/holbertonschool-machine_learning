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

    # Define sigmoid activation function
    def sigmoid(self, z):
        """
        sigmoid activation function
        """
        return (1 / (1 + np.exp(-z)))

    def process_outputs(self, outputs, image_size):
        """
        Process outputs of Darknet model
        outputs: numpy.ndarray containing predictions for a single image
            Each output will have the shape (grid_height,
                                             grid_width,
                                             anchor_boxes,
                                             4 + 1 + classes)
                grid_height: height of grid used for output
                grid_width: width of grid used for output
                anchor_boxes: number of anchor boxes used
                4: t_x, t_y, t_w, t_h
                1: box confidence
                classes: class probabilities for all classes
        image_size: numpy.ndarray containing the original image size
            [image_height, image_width]

        Returns a tuple of (boxes, box_confidences, box_class_probs)
            boxes: list of numpy.ndarrays of shape (grid_height,
                                                    grid_width,
                                                    anchor_boxes,
                                                    4)
                4: x1, y1, x2, y2
                    (x1, y1, x2, y2) should represent the boundary box relative
                    to original image
            box_confidences: a list of numpy.ndarrays of shape (grid_height,
                                                                grid_width,
                                                                anchor_boxes,
                                                                classes)
                containing the confidences for each output, respectively
            box_class_probs: list of numpy.ndarrays of shape (grid_height,
                                                              grid_width,
                                                              anchor_boxes,
                                                              classes)
                containing the box's class probabilities for each output
        """
        # Create lists for return
        box_confidences, box_class_probs = [], []
        boxes = [output[..., :4] for output in outputs]

        # Create lists for bounding box corner coordinates
        x_corners, y_corners = [], []

        # Creat all of the grid cells to overlay image
        # Calculate box_confidences and box_class_probs
        for output in outputs:
            grid_height = output.shape[0]
            grid_width = output.shape[1]
            anchors = output.shape[2]

            cx = np.arange(grid_width).reshape(1, grid_width)
            cx = np.repeat(cx, grid_height, axis=0)
            x_corners.append(np.repeat(cx[..., np.newaxis], anchors, axis=2))

            cy = np.arange(grid_width).reshape(1, grid_width)
            cy = np.repeat(cy, grid_height, axis=0).T
            y_corners.append(np.repeat(cy[..., np.newaxis], anchors, axis=2))

            box_confidences.append(self.sigmoid(output[..., 4:5]))
            box_class_probs.append(self.sigmoid(output[..., 5:]))

        input_width = self.model.input.shape[1].value
        input_height = self.model.input.shape[2].value

        for x, box in enumerate(boxes):
            # Activate bounding boxes
            bx = (self.sigmoid(box[..., 0]) + x_corners[x])/outputs[x].shape[1]
            by = (self.sigmoid(box[..., 1]) + y_corners[x])/outputs[x].shape[0]
            bw = (np.exp(box[..., 2]) * self.anchors[x, :, 0]) / input_width
            bh = (np.exp(box[..., 3]) * self.anchors[x, :, 1]) / input_height

            # Move bounding box coordinates from corner to center
            box[..., 0] = (bx - (bw * .5)) * image_size[1]
            box[..., 1] = (by - (bh * .5)) * image_size[0]
            box[..., 2] = (bx + (bw * .5)) * image_size[1]
            box[..., 3] = (by + (bh * .5)) * image_size[0]


        return (boxes, box_confidences, box_class_probs)
