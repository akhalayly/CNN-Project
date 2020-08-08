import sys

from vgg16 import VGG16
from keras.preprocessing import image
from imagenet_utils import preprocess_input, decode_predictions
import edgeB as eb
from keras.models import Model
import numpy as np
from sklearn import random_projection
import torchvision as tv
import tensorflow as tf


def ImageBoxesVggGaussian(model, img, transformer, inter_layer):
    boxes = eb.edgeBoxes(model, img)
    transformed_boxes = []
    for bx in boxes:
        trans1 = tv.transforms.ToTensor()
        x = trans1(bx[0]).reshape(1, 3, 231, 231)
        # x = np.expand_dims(x, axis=0)
        # x = preprocess_input(x)
        intermediate_output = inter_layer.forward(x)
        intermediate_output = np.reshape(intermediate_output.detach(), (1, 13 * 13 * 384))
        intermediate_output = transformer.transform(intermediate_output)
        transformed_boxes.append((intermediate_output, bx[1], bx[2]))
    return transformed_boxes
        # preds = model.predict(x)
        # print('Predicted:', decode_predictions(preds))
    # print: [[u'n02504458', u'African_elephant']]
