import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from general import plot_labels, plot_nms_limits
import torch
import pickle

import sys
sys.path.append("PyTorch_YOLOv4")
from utils.general import non_max_suppression, xywh2xyxy

CACHE_LOCATION=os.path.join(os.path.dirname(os.path.abspath(__file__)), "cached_datasets")
os.makedirs(CACHE_LOCATION, exist_ok=True)

# TODO: add nms type
def get_nms_bounds(labels, classes, netShape):
    print("Checking for NMS Limitations...", end="\r")
    # set up IOU thresholds for NMS
    threshs = [round(x, 2) for x in np.linspace(0.05, 1.0, num=19, endpoint=False)]
    nmsDict = dict((th, 0) for th in threshs)

    # convert to pytorch tensor so we can utilize a gpu
    labels = torch.from_numpy(labels).cuda()

    # change to relative
    labels[:, [2, 4]] *=  netShape[1]
    labels[:, [3, 5]] *= netShape[0]

    # add objectness conf
    fakeObjectness = torch.ones((labels.shape[0], 1)).float().cuda()
    detLabels = torch.cat((labels, fakeObjectness), dim=1)

    # nx7 from (imgIdx, cls, xcen, ycen, width, height, conf) -> (imgIdx, xcen, ycen, width, height, conf, cls)
    permuteIdxs = torch.LongTensor([0, 2, 3, 4, 5, 6, 1])
    detLabels = detLabels[:, permuteIdxs]

    # replace cls with one hot encoded vector
    labelCls = detLabels[:, -1:].long()
    fakeCls = torch.zeros((labels.shape[0], len(classes))).float().cuda()
    fakeCls[torch.arange(fakeCls.size(0)).unsqueeze(1), labelCls] = 1.
    detLabels = torch.cat((detLabels[:, :-1], fakeCls), dim=1)

    # do nms per img
    imgNums = torch.unique(detLabels[:, 0], dim=0)
    for imgIdx in imgNums:
        currDetIdxs = detLabels[:, 0] == imgIdx
        currDets = detLabels[currDetIdxs]
        numDets = currDets.shape[0]
        for th in threshs:
            afterNms = non_max_suppression(currDets[:, 1:].unsqueeze(0), iou_thres=th)[0]
            nmsDict[th] += (numDets - afterNms.shape[0])

    print("Checking for NMS Limitations - Done.")
    return nmsDict

def verifyDataHelper(logger, outputPath, txtFile, netShape, classNames, clearCache=False):
    classes = set()
    badImages = []
    badText = []
    shapes = []
    labels = []

    txtName = txtFile.split("/")[-1]
    cachedDir = os.path.join(CACHE_LOCATION, txtName.replace(".txt", ".p"))
    if os.path.exists(cachedDir) and not clearCache:
        with open(cachedDir, "rb") as f:
            vars = pickle.load(f)

        badImages = vars["badImages"]
        badText = vars["badText"]
        shapes = vars["shapes"]
        labels = vars["labels"]
        classes = vars["classes"]
    else:
        # read file
        with open(txtFile, "r") as f:
            imgPaths = f.readlines()

        # iterate through image paths
        for idx, imgPath in enumerate(tqdm(imgPaths, desc="Verifying Data from {}".format(txtName))):
            imgPath = imgPath.strip()

            # check if the image exists
            if not os.path.exists(imgPath):
                badImages.append("MISSING: {}".format(imgPath))
                continue

            img = Image.open(imgPath)  # this reads the headers of the file without actually loading the image
            imgShape = img.size
            del img
            shapes.append([imgShape[1], imgShape[0]])

            # get text file
            fn, ext = os.path.splitext(imgPath)
            txtTruth = imgPath.replace(ext, ".txt")

            # if the text file exists, check the truths
            # otherwise, mark as missing
            if os.path.exists(txtTruth):
                with open(txtTruth, "r") as f:
                    truthLines = f.readlines()

                for truth in truthLines:
                    line = truth.strip().split(" ")
                    line = [float(x) for x in line]
                    currLbl = np.asarray([idx] + line)
                    labels.append(currLbl)
                    box = line[1:]
                    classes.add(line[0])
                    if any(x for x in box if x > 1.0 or x < 0):
                        badText.append("BBOX ATTRIBUTE > 1 or < 0: {}".format(txtTruth))
            else:
                badText.append("MISSING: {}".format(txtTruth))

        # save information so we don't need to iterate through the entire dataset again
        vars = dict()
        vars["badImages"] = badImages
        vars["badText"] = badText
        vars["shapes"] = shapes
        vars["labels"] = labels
        vars["classes"] = classes
        with open(cachedDir, "wb") as f:
            pickle.dump(vars, f)

    txtName = txtFile.split("/")[-1]
    badImgTxt = os.path.join(outputPath, txtName.replace(".txt", ".badImg"))
    badTextTxt = os.path.join(outputPath, txtName.replace(".txt", ".badTxt"))
    numBadImg = len(badImages)
    numBadTxt = len(badText)

    # check we have the same number of classes
    if len(classes) != len(classNames):
        logger.warn("Mismatched number of classes, {} were given but {} were found in {}".format(
            len(classNames), len(classes), txtName
        ))

    # make sure we got some labels
    assert len(labels) > 0, "Missing ground truth for all images in {}".format(txtName)\

    # plot label information
    labels = np.stack(labels)
    saveName = txtName.replace(".txt", "") + "_labels_{}.png"
    plot_labels(labels[:, 1:].copy(), names=classes, save_dir=os.path.join(outputPath, saveName))

    # check if nms causes a performance limitation
    nmsDict = get_nms_bounds(labels.copy(), classes, netShape)
    plot_nms_limits(outputPath, nmsDict, labels.shape[0])

    # check truth sizes
    boxes = labels[:, 2:]
    areas = (boxes[:, 2] * netShape[0]) * (boxes[:, 3] * netShape[1])

    if numBadImg > 0:
        logger.warn("Found {} bad images in {}. Recording to {}".format(len(badImages), txtFile, badImgTxt))
        with open(badImgTxt, "w") as f:
            f.writelines("{}\n".format(x) for x in badImages)

    if numBadTxt > 0:
        logger.warn("Found {} bad text files in {}. Recording to {}".format(len(badText), txtFile, badTextTxt))
        with open(badTextTxt, "w") as f:
            f.writelines("{}\n".format(x) for x in badText)

    logger.info("Found in {}:".format(txtFile))
    logger.print("\tTotal Images: {}".format(len(imgPaths)))
    logger.print("\tTotal Classes: {}".format(len(classes)))
    logger.print("\tMinimum BBox Area (by Network Input): {}".format(areas.min()))
    logger.print("\tMaximum BBox Area (by Network Input): {}".format(areas.max()))
    logger.print("\tMean BBox Area (by Network Input): {}".format(areas.mean()))
    logger.print()

    infoDict = {"NumImages": len(imgPaths),
                "NumClasses": len(classes),
                "MinArea": areas.min(),
                "MaxArea": areas.max(),
                "Boxes": boxes,
                "Shapes": np.asarray(shapes)}

    return infoDict