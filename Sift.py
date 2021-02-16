import cv2
from PIL import Image
import matplotlib.pyplot as plt


def circleToBox(x, y, radius, image):
    new_x = max(0, x - radius)
    new_y = max(0, y - radius)
    width = 2 * radius if (new_x + 2 * radius) < image.shape[1] else (image.shape[1] - new_x)
    height = 2 * radius if (new_y + 2 * radius) < image.shape[0] else (image.shape[0] - new_y)
    return new_x, new_y, width, height


def filtering(param, index):
    if param[1] in index:
        return True
    return False


def Sift(image):
    # reading image
    # _img1 = cv2.imread('GardenPointWalking/day_left/0.jpg')
    _img1 = cv2.imread(image)
    img1 = cv2.cvtColor(_img1, cv2.COLOR_BGR2GRAY)
    # keypoints
    sift = cv2.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
    boxes = []
    scores = []
    keypoints_1.sort(key=lambda k: k.response, reverse=True)
    for kp in keypoints_1:
        if kp.size < 15:
            boxes.append(circleToBox(kp.pt[0], kp.pt[1], kp.size, _img1))
            scores.append(kp.response)
    # print(len(keypoints_1), len(keypoints_2))
    highest_kp = max(keypoints_1, key=lambda x: x.response)
    lowest_radius = min(keypoints_1, key=lambda x: x.size)
    highest_radius = max(keypoints_1, key=lambda x: x.size)
    # bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    # matches = bf.match(descriptors_1, descriptors_2)
    # matches = sorted(matches, key=lambda x: x.distance)
    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.015, 0.3)
    for i in range(boxes.__len__()):
        boxes[i] = (boxes[i], i)
        scores[i] = (scores[i], i)
    boxes = list(filter(lambda param: filtering(param, indices), boxes))
    scores = list(filter(lambda param: filtering(param, indices), scores))
    for i in range(boxes.__len__()):
        boxes[i] = boxes[i][0]
        scores[i] = scores[i][0]
    factor = min(150,boxes.__len__())
    boxes = boxes[:factor]
    scores = scores[:factor]
    boxes_to_return = []
    to_crop = Image.open(image)
    to_crop = to_crop.resize((231, 231))
    for box in boxes:
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        cropped = to_crop.crop((x, y, x + w, y + h))
        cropped = cropped.resize((231, 231))
        boxes_to_return.append((cropped, w, h))
        cv2.rectangle(_img1, (round(x), round(y)), (round(x + w), round(y + h)), (0, 255, 0), 1, cv2.LINE_AA)
    # cv2.imshow("edgeboxes", _img1)
    # cv2.waitKey(0)
    return boxes_to_return

# if __name__ == "__main__":
#      Sift('GardenPointWalking/day_left/0.jpg')