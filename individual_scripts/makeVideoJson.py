import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
import json

def drawBoundingBox(img, objects, color, confThresh, putText=False):
    # [frame index, class name, confidence, bounding box as xyxy)
    for obj in objects:
        if obj[2] >= confThresh:
            bbox = obj[3].astype(int)
            cv2.rectangle(img, tuple(bbox[:2]), tuple(bbox[2:]), color, 1)
            if putText:
                cv2.putText(img, '{}: {}'.format(obj[1], round(obj[2], 2)), tuple(bbox[:2]-10), 0, 0.3, color)

    return img

def makeVideo():
    assert os.path.exists(args.json), "JSON {} does not exist".format(args.json)
    os.makedirs(args.folder, exist_ok=True)
    outPath = os.path.join(args.folder, args.name)

    with open(args.json, "r") as f:
        detList = json.load(f)
    
    # check first image size for video size
    firstImgPath = detList[0]["filename"]
    img = cv2.imread(firstImgPath, cv2.IMREAD_UNCHANGED)
    vidOut = cv2.VideoWriter(outPath, cv2.VideoWriter_fourcc('M','J','P','G'), 5, (img.shape[1], img.shape[0]))

    for detDict in tqdm(detList, desc="Drawing Detections"):
        detections = []
        frameId = detDict['frame_id'] - 1
        dets = detDict['objects']
        filename = detDict['filename']

        img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        img = cv2.normalize(img[:,:,0], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        img = np.dstack([img, img, img])
        imgH = img.shape[0]
        imgW = img.shape[1]

        for object in dets:
            conf = object["confidence"]
            clsId = object["class_id"]
            name = object["name"]
            coords = object["relative_coordinates"]
            xcen = coords["center_x"] * imgW
            ycen = coords["center_y"] * imgH
            xcen = np.clip(xcen, 0, imgW)
            ycen = np.clip(ycen, 0, imgH)
            width = coords["width"] * imgW
            height = coords["height"] * imgH
            xmin = xcen - (width / 2)
            xmax = xcen + (width / 2)
            ymin = ycen - (height / 2)
            ymax = ycen + (height / 2)
            bbox = [xmin, ymin, xmax, ymax]
            bbox = np.array([int(round(x, 0)) for x in bbox])
            detections.append([frameId, name, conf, bbox])

        img = drawBoundingBox(img, detections, (0, 255, 0), 0, putText=True)
        vidOut.write(img)

    vidOut.release()
    return
        

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--json', type=str, help='Path to darknet json containing detections', required=True)
    parser.add_argument('-f', '--folder', type=str, help="Path to output folder", required=True)
    parser.add_argument('-n', '--name', type=str, default="drawnDets_video")
    parser.add_argument('--fps', type=int, default=5, help="Frames per second")
    args = parser.parse_args()
    makeVideo()