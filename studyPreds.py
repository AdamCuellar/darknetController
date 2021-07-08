import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

import sys
sys.path.append("PyTorch_YOLOv4")
from utils.model import Darknet
from utils.general import parse_cfg, parse_names, clip_coords, build_targets, non_max_suppression, xyxy2xywh
from utils.torch_utils import select_device
from utils.datasets import LoadData

from general import plot_labels
from tqdm import tqdm

def get_grid_edges(strides, scope):
    edges = []
    for st in strides:
        tmp = np.arange(0, int(scope/st) + 1) * st
        edges.append(tmp)

    return np.concatenate(edges)

def plot_objectness_xy(data, outPath, name, width, height, strides):
    # threshs = [round(x, 2) for x in np.linspace(0.05, 1.0, num=19, endpoint=False)] + [1.0]

    ax = plt.subplots(2, 1, figsize=(32, 16), tight_layout=True)[1].ravel()
    # for idx, th in enumerate(threshs):
    #     rgb = np.random.rand(3,)
    #     lastTh = threshs[idx-1] if idx > 0 else 0
    #     lbl = "{} <= x < {}".format(lastTh, th)
    #     currData = data[data[:, 4] < th]
    #     currData = currData[currData[:, 4] >= lastTh]
    #     ax[0].plot(currData[:, 0], currData[:, 1], 'o', color=rgb, label=lbl)

    xticks = get_grid_edges(strides, width)
    yticks = [0] + [round(x, 2) for x in np.linspace(0.05, 1.0, num=19, endpoint=False)] + [1.0]
    ax[0].plot(data[:, 0], data[:, 4], 'bo', markersize=1)
    ax[0].set_xlabel("X Center")
    ax[0].set_ylabel("Objectness")
    ax[0].set_xticks(xticks)
    ax[0].set_yticks(yticks)
    ax[0].set_xticklabels(labels=xticks, rotation=(45), fontsize=10, ha='right')

    xticks = get_grid_edges(strides, height)
    ax[1].plot(data[:, 1], data[:, 4], 'bo', markersize=1)
    ax[1].set_ylabel("Y Center")
    ax[1].set_ylabel("Objectness")
    ax[1].set_xticks(xticks)
    ax[1].set_yticks(yticks)
    ax[1].set_xticklabels(labels=xticks, rotation=(45), fontsize=10, ha='right')
    # ax.legend()
    file = os.path.join(outPath, "{}_xyCen_objectness.png".format(name))
    plt.savefig(file, dpi=200)
    plt.close()
    return

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

    # get nms options from last yolo layer
    last_yolo_idx = model.yolo_layers[-1]
    nmsType = model.module_defs[last_yolo_idx].get('nms_kind', "normal")
    beta1 = model.module_defs[last_yolo_idx].get('beta_nms', 0.6)

    # instantiate dataloader
    dataloader = LoadData(dataFile, netParams, imgShape, valid=True)

    truthXY, filteredPredXY = [], []
    for img, truth, pth, shape in tqdm(dataloader, desc="Studying predictions"):
        img = img.to(device)

        # record truth
        tmpTruth = truth.clone()
        tmpTruth[:, [2, 4]] *= img.shape[2]
        tmpTruth[:, [3, 5]] *= img.shape[1]
        truthXY.append(tmpTruth[:, 2:4])

        truth = truth.to(device)
        img = img.float()
        img /= np.iinfo(imgType).max # 0- 1.0
        if img.ndimension() == 3: # make it a batch
            img = img.unsqueeze(0)

        with torch.no_grad():
            infOut, rawPred = model(img, study_preds=True)

        # flatten for plotting and nms
        flatPreds = torch.cat([x.view(1, -1, x.shape[-1]) for x in infOut], 1)

        # filter preds via nms
        filteredPreds = non_max_suppression(flatPreds, conf_thres=0.0001, iou_thres=0.45, nmsType=nmsType, beta1=beta1, max_det=1000)[0]
        if filteredPreds is None: continue

        # convert to xywh and record
        filteredPreds[:, :4] = xyxy2xywh(filteredPreds[:, :4])
        filteredPredXY.append(filteredPreds.clone().cpu())

        # TODO: add mAP, check how many small, medium, large objects we missed


    # plot histograms of xy cens
    truthXY = torch.cat(truthXY, 0).numpy()
    plot_histograms(truthXY, width=netParams["width"], height=netParams["height"],
                        outPath=outPath, name="truths")

    filteredPredXY = torch.cat(filteredPredXY, 0).numpy()
    strides = [model.module_list[idx].stride for idx in model.yolo_layers]
    plot_objectness_xy(filteredPredXY, outPath, name="preds_afterNMS", width=netParams["width"],
                       height=netParams["height"], strides=strides)

    plot_histograms(filteredPredXY[:, :2], width=netParams["width"], height=netParams["height"],
                    outPath=outPath, name="preds_afterNMS")


    # TODO: plot statistics of all predictions
    # allPredictions = np.stack(allPredictions)
    # saveName = outName + "_predictions_{}.png"
    # plot_labels(allPredictions, names, os.path.join(outPath, saveName))

    return


def test_script():
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