import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
import pandas as pd
import copy
from collections import Counter
import sys
from tqdm import tqdm
import os

def drawPlots(results, totalGts, outputPath, outputName, numImages):
    """
    Draw ROC plots using Probability of Detection vs False Alarm Rate
    :param results: dictionary containing true positives and false positives for each threshold
    :param totalGts: number of unique detections or ground truths
    :param outputPath: path for ROC png output
    :param outputName: name for ROC png output
    :param numImages: total number of images/frames
    :return:
    """
    x_far = []
    y_pdet = []
    y_precision = []
    tps, fps = [], []
    for r in sorted(results.keys()):
        tp = results[r]['TP']
        fp = results[r]['FP']
        for r_2 in sorted(results.keys()):
            if r >= r_2: continue
            tp += results[r_2]['TP']
            fp += results[r_2]['FP']
        pdet = tp / totalGts
        far = (fp / numImages)
        den = (tp + fp)
        prec = tp / den if den > 0 else 0
        x_far.append(far)
        y_pdet.append(pdet)
        y_precision.append(prec)
        tps.append(tp)
        fps.append(fp)

    # pdet is the same as recall so copy the list
    x_recall = y_pdet.copy()
    resultDict = {"Thresholds": sorted(results.keys()),
                  "PDet": y_pdet,
                  "FAR": x_far,
                  "Precision": y_precision,
                  "Recall": x_recall,
                  'TP':tps,
                  'FP':fps,
                  'Total GT':totalGts,
                  'Number of Images': numImages}

    # save results as csv
    resultDF = pd.DataFrame(resultDict)
    fn_csv = os.path.join(outputPath, outputName + "_Results.csv")
    resultDF.to_csv(fn_csv, index=False)

    title_pdetfar = "Probability of Detection vs. False Alarm Rate"
    fn_pdetfar = os.path.join(outputPath, outputName + "_PDet_FAR.png")

    # add zero zero for nice graph
    if 0 not in x_far:
        x_far.append(0)
        y_pdet.append(0)

    plt.grid()
    plt.plot(x_far, y_pdet, "b-o")
    plt.yticks(np.arange(0,1.1,0.1))

    plt.title(title_pdetfar)
    plt.xlabel("False Alarm Rate (per Frame)")
    plt.ylabel("Probability of Detection")
    plt.savefig(fn_pdetfar)
    plt.close()

    title_precrec = "Precision vs. Recall"
    fn_precrec = os.path.join(outputPath, outputName + "_Precision_Recall.png")
    plt.grid()
    plt.plot(x_recall, y_precision, "b-o")
    plt.title(title_precrec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig(fn_precrec)
    plt.close()

    return

def calc_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1
        box2
    Returns:
        iou: the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    inter = (np.min((box1[2], box2[2])) - np.max((box1[0], box2[0]))).clip(0) * \
            (np.min((box1[3], box2[3])) - np.max((box1[1], box2[1]))).clip(0)

    return inter / (area1 + area2 - inter)  # iou = inter / (area1 + area2 - inter)

def setupThreshDict():
    """
    Sets up a dictionary containing different confidence thresholds
    :return: dictionary with score thresholds
    """
    ranges = [round(x, 2) for x in np.linspace(0.05, 1.0, num=20, endpoint=True)]
    ranges += [round(x, 3) for x in np.linspace(0.955, 1.0, num=9, endpoint=False)]
    ranges = sorted(ranges)

    threshDict = dict()
    for r in ranges:
        threshDict[r] = dict()
        threshDict[r]['FP'] = 0
        threshDict[r]['TP'] = 0

    return threshDict

def evalMetrics(groundtruths, detections, numImages, iou_thresh=0.5, noDualEval=False):
    """Get the metrics used by the VOC Pascal 2012 challenge.
    Get
    Args:
        groundtruths: list of ground truth detections
        detections: list of model detections
        IOUThreshold: IOU threshold indicating which detections will be considered TP or FP
        (default value = 0.5);
        method (default =  0 for EveryPointInterpolation): It can be calculated as the implementation
        in the official PASCAL VOC toolkit (EveryPointInterpolation), or applying the 11-point
        interpolation as described in the paper "The PASCAL Visual Object Classes(VOC) Challenge"
        or EveryPointInterpolation"  (1 for ElevenPointInterpolation);
    Returns:
        A list of dictionaries. Each dictionary contains information and metrics of each class.
        The keys of each dictionary are:
            dict['precision']: array with the precision values;
            dict['recall']: array with the recall values;
            dict['AP']: average precision;
            dict['interpolated precision']: interpolated precision values;
            dict['interpolated recall']: interpolated recall values;
            dict['total positives']: total number of ground truth positives;
            dict['total TP']: total number of True Positive detections;
            dict['total FP']: total number of False Negative detections;
            dict['ThreshDict']: dictionary of dictionaries containing TP and FP at thresholds 0.5:0.95
            dict['DetectedGTClasses']: list containing index of groundtruth class for groundtruths that were detected
            dict['GoodPredClasses']: list containing index of predicted class for true positive detections
        pDet: Probability of detection
        far: False Alarm Rate
    """
    pDet = 0
    far = 0
    total_gt = 0

    # sequester results by confidence
    threshDict = setupThreshDict()

    # copy lists
    gts = copy.deepcopy(groundtruths)
    dects = copy.deepcopy(detections)

    # number of groundtruths
    npos = len(gts)

    # sort detections by decreasing confidence
    ogDectIdxs = sorted(list(range(len(dects))), key=lambda conf: dects[conf][2], reverse=True)
    dects = sorted(dects, key=lambda conf: conf[2], reverse=True)
    TP = np.zeros(len(dects))
    FP = np.zeros(len(dects))

    # create dictionary with amount of gts for each image
    det = Counter([cc[0] for cc in gts])
    for key, val in det.items():
        det[key] = np.zeros(val)

    # Loop through detections
    for d in range(len(dects)):
        thresholds = sorted(threshDict.keys())
        matchingThresh = [x for idx, x in enumerate(thresholds) if x <= dects[d][2] < thresholds[idx+1]] # TODO: might need to use numpy.isclose for the = 
        matchingThresh = matchingThresh[0] if len(matchingThresh) != 0 else None

        # Find ground truth image
        gt = []
        for idx, g in enumerate(gts):
            if g[0] == dects[d][0]:
                gt.append(g)
        iouMax = sys.float_info.min
        jmax = None
        for j in range(len(gt)):
            iou = calc_iou(dects[d][3], gt[j][3])
            if iou > iouMax:
                iouMax = iou
                jmax = j

        # Assign detection as true positive/don't care/false positive
        if iouMax > iou_thresh:
            if det[dects[d][0]][jmax] == 0:
                TP[d] = 1  # count as true positive
                det[dects[d][0]][jmax] = 1  # flag as already 'seen'

                # mark tp for thresh
                if matchingThresh is not None:
                    threshDict[matchingThresh]['TP'] += 1
            else:
                if not noDualEval:
                    FP[d] = 1
                    # mark fp for thresh
                    if matchingThresh is not None:
                        threshDict[matchingThresh]['FP'] += 1
        # - A detected "cat" is overlaped with a GT "cat" with IOU >= IOUThreshold.
        else:
            FP[d] = 1  # count as false positive
            # mark fp for thresh
            if matchingThresh is not None:
                threshDict[matchingThresh]['FP'] += 1

    # compute precision, recall and average precision
    acc_FP = np.cumsum(FP)
    acc_TP = np.cumsum(TP)
    rec = acc_TP / npos
    prec = np.divide(acc_TP, (acc_FP + acc_TP))
    
    #TODO: do this using numpy so its faster?
    ogIdxTp = [None] * len(TP)
    ogIdxFp = [None] * len(FP)
    for ogIdx, tpVal, fpVal in zip(ogDectIdxs, TP, FP):
        ogIdxTp[ogIdx] = tpVal
        ogIdxFp[ogIdx] = fpVal
    
    # add result in the dictionary to be returned
    r = {
        'Precision': prec,
        'Recall': rec,
        'Total Detections': npos,
        'Total TP': np.sum(TP),
        'Total FP': np.sum(FP),
        'ThreshDict': threshDict,
        'TPByIndex':ogIdxTp,
        'FPByIndex':ogIdxFp
    }
    total_gt += npos
    pDet += np.sum(TP)
    far += np.sum(FP)

    pDet /= total_gt
    far /= numImages
    return r, pDet, far

def drawBoundingBox(img, objects, color, confThresh, putText=False):
    """
    Draws bounding box(s) on image
    :param img: image to draw boxes on
    :param objects: list of object(s) bounding box
    :param color: color to draw bounding box
    :param putText: add text to bounding box
    :return: image with bounding box drawn
    """

    # [frame index, class id, confidence, bounding box as xyxy)
    for obj in objects:
        if obj[2] >= confThresh:
            bbox = obj[3].astype(int)
            cv2.rectangle(img, tuple(bbox[:2]), tuple(bbox[2:]), color, 1)
            if putText:
                cv2.putText(img, 'Target: {}'.format(round(obj[2], 2)), tuple(bbox[:2]-10), 0, 0.3, color)

    return img

def drawDetections(outputPath, imageList, groundtruths, predictions, tpByIndex, fpByIndex):
    tpColor = (0, 255, 0) # green
    fpColor = (0, 0, 255) # red
    gtColor = (255,255,255) # white
    commonPath = os.path.commonpath(imageList) + "/"
    drawnPath = os.path.join(outputPath, "drawnDets")
    os.makedirs(drawnPath, exist_ok=True)

    # split gts by frame
    gtByFrame = dict()
    for gt in groundtruths:
        frameIdx = gt[0]
        if frameIdx not in gtByFrame:
            gtByFrame[frameIdx] = list()
        gtByFrame[frameIdx].append(gt)

    # split tps and fps by frame
    tpByFrame = dict()
    fpByFrame = dict()
    for idx, pred in enumerate(predictions):
        frameIdx = pred[0]
        if tpByIndex[idx]:
            if frameIdx not in tpByFrame:
                tpByFrame[frameIdx] = list()
            tpByFrame[frameIdx].append(pred)
        elif fpByIndex[idx]:
            if frameIdx not in fpByFrame:
                fpByFrame[frameIdx] = list()
            fpByFrame[frameIdx].append(pred)
        else:
            pass
            # this is blank cause sometimes a prediction can be a duplicate (so not wrong, but also not another TP)

    for idx, imgPath in enumerate(imageList):
        imgDir, imgName = os.path.split(imgPath)
        img = cv2.imread(imgPath, cv2.IMREAD_UNCHANGED)
        img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        if img.ndim != 3:
            img = np.dstack([img, img, img])

        # get groundtruths for this frame
        currGts = gtByFrame[idx] if idx in gtByFrame else []

        # get predictions for this frame
        tps = tpByFrame[idx] if idx in tpByFrame else []
        fps = fpByFrame[idx] if idx in fpByFrame else []

        # draw bounding boxes
        img = drawBoundingBox(img, currGts, gtColor, confThresh=0)
        img = drawBoundingBox(img, tps, tpColor, confThresh=0.5, putText=True)
        img = drawBoundingBox(img, fps, fpColor, confThresh=0.5, putText=True)

        # check for sub directories in case file names are the same
        subDir = imgPath.replace(commonPath, "").replace(imgName, "")
        if len(subDir) > 0:
            tempPath = os.path.join(drawnPath, subDir)
            os.makedirs(tempPath, exist_ok=True)
            currOutPath = os.path.join(tempPath, imgName)
        else:
            currOutPath = os.path.join(drawnPath, imgName)

        cv2.imwrite(currOutPath, img)

    return