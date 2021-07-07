import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

import sys
sys.path.append("PyTorch_YOLOv4")
from utils.model import Darknet
from utils.general import parse_cfg, parse_names, clip_coords, build_targets
from utils.torch_utils import select_device
from utils.datasets import LoadData

from general import plot_labels
from tqdm import tqdm

def plot_histograms(data, height, width, outPath, name):
    x,y = data.T
    ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)[1].ravel()

    # plot 2d histogram
    ax[0].hist2d(x, y, bins=[width, height])
    ax[0].set_title("XY 2D Histogram")

    # plot 1d histogram for x and y
    ax[1].hist(x, bins=width)
    ax[1].set_title("X Center 1D Histogram")
    ax[2].hist(y, bins=height)
    ax[2].set_title("Y Center 1D Histogram")

    file = os.path.join(outPath, "{}_xyCen_histograms.png".format(name))
    plt.savefig(file, dpi=200)
    plt.close()
    return

def get_coco_object_scale(area):
    """ Checks whether the area of an object is considered small, medium, or large based off:
        https://cocodataset.org/#detection-eval
    """
    if area < 32 ** 2:
        return "small"
    elif 32 ** 2 <= area < 96 ** 2:
        return "medium"
    else:
        return "large"

def study_predictions(outPath, outName, modelCfg, modelWeights, dataFile, names, imgShape=None, device='cpu', imgType=np.uint8):

    # get device name
    device = select_device(device, batch_size=1)

    # make the out directory
    os.makedirs(outPath, exist_ok=True)

    # instantiate model
    modelDefs = parse_cfg(modelCfg)
    model = Darknet(modelDefs, img_size=imgShape)
    netParams = model.netParams

    # Load weights
    if modelWeights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(modelWeights, map_location=device)['model'])
    else:  # darknet format
        model.load_darknet_weights(modelWeights)

    # Fuse Conv2d + BatchNorm2d layers
    model.fuse()

    # Eval mode
    model.to(device).eval()

    # instantiate dataloader
    dataloader = LoadData(dataFile, netParams, imgShape, valid=True)

    allPredictions = []
    truthXY, predXY = dict((i, []) for i in range(len(model.yolo_layers))), dict((i, []) for i in range(len(model.yolo_layers)))
    truthXY2 = []
    for img, truth, pth, shape in tqdm(dataloader, desc="Studying predictions"):
        img = img.to(device)
        tmpTruth = truth.clone()
        tmpTruth[:, [2, 4]] *= img.shape[2]
        tmpTruth[:, [3, 5]] *= img.shape[1]
        truthXY2.append(tmpTruth[:, 2:4])
        truth = truth.to(device)
        img = img.float()
        img /= np.iinfo(imgType).max # 0- 1.0
        if img.ndimension() == 3: # make it a batch
            img = img.unsqueeze(0)

        with torch.no_grad():
            infOut, rawPred = model(img, study_preds=True)
            tcls, tbox, tIdxs, anchs = build_targets(rawPred, truth, model)

        # TODO: process detections (might need to do nms first, which means need to implement darknet nms methods)
        for headNum in range(len(infOut)):
            # get predictions for this head
            yoloHead = model.module_list[model.yolo_layers[headNum]]
            headPred = infOut[headNum].squeeze(0)
            headRaw = rawPred[headNum].squeeze(0)

            # get truths for this head and scale to prediction height/width
            headTruthBox = tbox[headNum].clone()
            headTruthBox[:, [0, 2]] *= img.shape[3]
            headTruthBox[:, [1, 3]] *= img.shape[2]

            # add xcens and ycens to dictionary
            truthXY[headNum].append(headTruthBox[:, :2].cpu())
            currPredXY = headPred.clone().reshape(-1, headPred.shape[-1])[:, :2]
            predXY[headNum].append(currPredXY.cpu())

    # plot histograms of xy cens
    truth = torch.cat(truthXY2, 0).numpy()
    plot_histograms(truth, width=netParams["width"], height=netParams["height"],
                        outPath=outPath, name="train_all_raw")

    for i in truthXY.keys():
        # plot truths
        truth = torch.cat(truthXY[i], 0).numpy()
        plot_histograms(truth, width=netParams["width"], height=netParams["height"],
                        outPath=outPath, name="train_head_{}".format(i))

        # plot preds
        pred = torch.cat(predXY[i], 0).numpy()
        plot_histograms(pred, width=netParams["width"], height=netParams["height"],
                        outPath=outPath, name="preds_head_{}".format(i))

    # plot all
    truth = [y for x in truthXY.values() for y in x]
    truth = torch.cat(truth, 0).numpy()
    plot_histograms(truth, width=netParams["width"], height=netParams["height"],
                    outPath=outPath, name="train_all")

    pred = [y for x in predXY.values() for y in x]
    pred = torch.cat(pred, 0).numpy()
    plot_histograms(pred, width=netParams["width"], height=netParams["height"],
                    outPath=outPath, name="pred_all")

    # TODO: plot statistics of all predictions
    # allPredictions = np.stack(allPredictions)
    # saveName = outName + "_predictions_{}.png"
    # plot_labels(allPredictions, names, os.path.join(outPath, saveName))

    return


def test_script():
    cocoFile = "/home/adam/PycharmProjects/darknetController/PyTorch_YOLOv4/data/testdev2017.txt"
    cocoNames = parse_names("/home/adam/PycharmProjects/darknetController/PyTorch_YOLOv4/data/coco.names")

    study_predictions(outPath="testingStudy",
                      outName="coco_val_2017",
                      modelCfg="/home/adam/PycharmProjects/PyTorch_YOLOv4_Darknet/cfgs/yolov4.cfg",
                      modelWeights="/home/adam/PycharmProjects/PyTorch_YOLOv4_Darknet/weights/yolov4.weights",
                      dataFile="/home/adam/PycharmProjects/darknetController/PyTorch_YOLOv4/data/coco2017_val.data",
                      imgShape=[512, 512],
                      names=cocoNames,
                      device='0',
                      imgType=np.uint8)

    return

# TODO: make this individual script with args
if __name__ == "__main__":
    test_script()