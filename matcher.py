import sys
from modelo import ImageBoxesVggGaussian
from sklearn import random_projection
import numpy as np
import math
import Alexnet as AN
from scipy import spatial
from os import listdir


def FindMatchingBoxes(img1, img2):
    img1_len, img2_len = (img1.__len__(), img2.__len__())
    list1 = list()
    list2 = list()
    listF = list()
    for bx1 in range(img1_len):
        min_index = np.argmin([spatial.distance.cosine(img1[bx1][0], img2[bx2][0]) for bx2 in range(img2_len)])
        list1.append(min_index)
    for bx2 in range(img2_len):
        min_index = np.argmin([spatial.distance.cosine(img1[bx1][0], img2[bx2][0]) for bx1 in range(img1_len)])
        list2.append(min_index)
    for bx1 in range(img1_len):
        if list2[list1[bx1]] == bx1:
            listF.append((bx1, list1[bx1]))
    return listF


def SimilarityBetweenImgs(img1, img2):
    rows, cols = (img1.__len__(), img2.__len__())
    dis = 0
    simi = 0
    listF = FindMatchingBoxes(img1, img2)
    for (bx1, bx2) in listF:
        w1, w2, h1, h2 = (img1[bx1][1], img2[bx2][1], img1[bx1][2], img2[bx2][2])
        distances = math.exp((0.5 * ((np.abs(w1 - w2) / max(w1, w2)) + (np.abs(h1 - h2) / max(h1, h2)))))
        distances = spatial.distance.cosine(img1[bx1][0], img2[bx2][0]) * distances
        simi = simi + (1 - distances)
        dis = dis + distances
    dis = dis / math.sqrt(rows * cols)
    simi = simi / math.sqrt(rows * cols)
    return simi, dis


if __name__ == '__main__':
    transformer = random_projection.GaussianRandomProjection(1024, random_state=14)
    intermediate_layer_model = AN.AlexNetConv3()
    images_list = list()
    filesd = listdir("GardenPointWalking/day_right")
    filesn = listdir("GardenPointWalking/day_left")
    projection_vec = np.reshape(np.asarray([0] * (13 * 13 * 384)), (1, 13 * 13 * 384))
    transformer.fit(projection_vec)
    sample_images = list()
    results = list()
    for file in filesd:
        sample_images.append(
            ImageBoxesVggGaussian(sys.argv[1], "GardenPointWalking/day_right/" + file, transformer,
                                  intermediate_layer_model))
    for file in filesn:
        images_list.append(ImageBoxesVggGaussian(sys.argv[1], "GardenPointWalking/day_left/" + file, transformer,
                                                 intermediate_layer_model))
    for i in range(0, sample_images.__len__()):
        maxS = (-np.inf, -1)
        maxS2 = maxS
        minD = (np.inf, -1)
        minD2 = minD
        for j in range(0, images_list.__len__()):
            # if j == 0:
            # print("started ")
            # print(i)
            # print("\n")
            sim, dis = SimilarityBetweenImgs(sample_images[i], images_list[j])
            if maxS[0] < sim:
                maxS2 = maxS
                maxS = (sim, j)
                minD2 = minD
                minD = (dis, j)
            elif maxS2[0] < sim:
                maxS2 = (sim, j)
                minD2 = (dis, j)
        print("SImage " + filesd[i] + " ==> RImage " + filesn[maxS[1]] + ", 2/1 s d" + str(maxS2[0] / maxS[0])
              + "," + str(minD2[0] / minD[0]) + "\n")
        # print("finished ")
        # print(i)
        # print("\n")
