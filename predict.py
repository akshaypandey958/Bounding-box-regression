from pyimagesearch import config
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model
import numpy as np
import imutils
import cv2
import argparse
import os
import math

def compute_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (box1[2] - box1[0]) * (box1[3] - box1[1])
    boxBArea = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return interArea / float(boxAArea + boxBArea - interArea)

def compute_mpdiou(box1, box2):
    iou = compute_iou(box1, box2)
    p = [
        (box1[0], box1[1]),
        (box1[2], box1[1]),
        (box1[0], box1[3]),
        (box1[2], box1[3])
    ]

    g = [
        (box2[0], box2[1]),
        (box2[2], box2[1]),
        (box2[0], box2[3]),
        (box2[2], box2[3])
    ]

    min_dist = float("inf")
    for px, py in p:
        for gx, gy in g:
            d = math.sqrt((px - gx)**2 + (py - gy)**2)
            if d < min_dist:
                min_dist = d

    xC_min = min(box1[0], box2[0])
    yC_min = min(box1[1], box2[1])
    xC_max = max(box1[2], box2[2])
    yC_max = max(box1[3], box2[3])
    C = math.sqrt((xC_max - xC_min)**2 + (yC_max - yC_min)**2)

    mpdiou = iou - (min_dist / (C))
    return mpdiou

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
    help="path to input image")
ap.add_argument("-g", "--gt", required=False, nargs=4, type=int,
    metavar=('startX', 'startY', 'endX', 'endY'),
    help="ground truth bounding box (in pixel coordinates)")
ap.add_argument("-o", "--output", required=False,
    help="path to save output image with IoU score")
args = vars(ap.parse_args())

imagePath = args["input"]

if not os.path.exists(imagePath):
    raise FileNotFoundError(f"Input image '{imagePath}' not found.")

print("[INFO] loading object detector...")
model = load_model(config.MODEL_PATH)

image_224 = load_img(imagePath, target_size=(224, 224))
image_224 = img_to_array(image_224) / 255.0
image_224 = np.expand_dims(image_224, axis=0)

preds = model.predict(image_224)[0]
(startX, startY, endX, endY) = preds

image = cv2.imread(imagePath)
(h, w) = image.shape[:2]

# normalization
pred_box = [
    int(startX * w),
    int(startY * h),
    int(endX * w),
    int(endY * h)
]

# Drawing bounding box
cv2.rectangle(image, (pred_box[0], pred_box[1]), (pred_box[2], pred_box[3]), (0, 255, 0), 2)
cv2.putText(image, "Predicted", (pred_box[0], pred_box[1]-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

if args["gt"] is not None:
    gt_box = args["gt"]

    cv2.rectangle(image, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), (0, 0, 255), 2)
    cv2.putText(image, "Ground Truth", (gt_box[0], gt_box[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    iou = compute_iou(pred_box, gt_box)
    mpdiou = compute_mpdiou(pred_box, gt_box)

    print(f"[INFO] IoU: {iou:.4f}")
    print(f"[INFO] MPDIoU: {mpdiou:.4f}")

    cv2.putText(image, f"IoU: {iou:.4f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.putText(image, f"MPDIoU: {mpdiou:.4f}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

cv2.imshow("Output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


