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

def check_lands_on_grid_edge(gridEdges, xy, name):
    xEdges, yEdges = [], []
    for idx, edges in gridEdges.items():
        xEdges.extend(edges[1].tolist())
        yEdges.extend(edges[0].tolist())

    xEdges, yEdges = set(xEdges), set(yEdges)
    numX = [np.where(xy[:, 0] == x, 1, 0).sum() for x in xEdges]
    numY = [np.where(xy[:, 1] == x, 1, 0).sum() for x in yEdges]
    numBoth = [np.where(np.logical_and(xy[:, 0] == x, xy[:, 1] == y), 1, 0).sum() for x in xEdges
               for y in yEdges]
    print("{} {} land on X edges".format(sum(numX), name))
    print("{} {} land on Y edges".format(sum(numY), name))
    print("{} {} land on both X&Y edges".format(sum(numBoth), name))
    return

def get_grid_edges(strides, scope):
    edges = []
    if not isinstance(strides, list):
        strides = [strides]

    for st in strides:
        tmp = np.arange(0, int(scope/st) + 1) * st
        edges.append(tmp)

    return np.concatenate(edges)

def plot_objectness_xy(data, outPath, name, width, height, strides):
    yticks = [0] + [round(x, 2) for x in np.linspace(0.05, 1.0, num=19, endpoint=False)] + [1.0]

    ax = plt.subplots(len(strides), 2, figsize=(40, 16), tight_layout=True)[1].ravel()
    for idx in range(len(strides)):
        xticks = get_grid_edges(strides[idx], width)
        plotIdx = idx+1*idx
        ax[plotIdx].plot(data[idx][:, 0], data[idx][:, 4], 'bo', markersize=1)
        ax[plotIdx].set_xlabel("X Center")
        ax[plotIdx].set_ylabel("Objectness")
        ax[plotIdx].set_xticks(xticks)
        ax[plotIdx].set_yticks(yticks)
        ax[plotIdx].set_xticklabels(labels=xticks, rotation=(45), fontsize=10, ha='right')
        ax[plotIdx].set_title("Head {} w/ Stride {}".format(idx, strides[idx]))

        xticks = get_grid_edges(strides[idx], height)
        ax[plotIdx+1].plot(data[idx][:, 1], data[idx][:, 4], 'bo', markersize=1)
        ax[plotIdx+1].set_xlabel("Y Center")
        ax[plotIdx+1].set_ylabel("Objectness")
        ax[plotIdx+1].set_xticks(xticks)
        ax[plotIdx+1].set_yticks(yticks)
        ax[plotIdx+1].set_xticklabels(labels=xticks, rotation=(45), fontsize=10, ha='right')
        ax[plotIdx+1].set_title("Head {} w/ Stride {}".format(idx, strides[idx]))

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

    # fake pass to initialize model variables
    _ = model(torch.randn(1,3,netParams["height"], netParams["width"]).to(device))

    # get nms options from last yolo layer
    last_yolo_idx = model.yolo_layers[-1]
    nmsType = model.module_defs[last_yolo_idx].get('nms_kind', "normal")
    beta1 = model.module_defs[last_yolo_idx].get('beta_nms', 0.6)

    # get model strides and edges
    strides = [model.module_list[idx].stride for idx in model.yolo_layers]
    gridEdges = dict((idx, [get_grid_edges(s, netParams["height"]), get_grid_edges(s, netParams["width"])])
                     for idx, s in enumerate(strides))

    # instantiate dataloader
    dataloader = LoadData(dataFile, netParams, imgShape, valid=True, imgType=imgType)

    truthXY, filteredPredXY = [], dict((i, []) for i in range(len(model.yolo_layers)))
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

        # go through the predicitions from each head, flatten them, and run NMS (this might show duplicates that would normally be reduced via nms on ALL heads)
        for idx, x in enumerate(infOut):
            flatPreds = x.view(1, -1, x.shape[-1])

            # filter preds via nms
            filteredPreds = non_max_suppression(flatPreds.cpu(), conf_thres=0.0001, iou_thres=0.45, nmsType=nmsType, beta1=beta1, max_det=300)[0]
            if filteredPreds is None: continue

            # convert to xywh and record
            filteredPreds[:, :4] = xyxy2xywh(filteredPreds[:, :4])
            filteredPredXY[idx].append(filteredPreds.clone().cpu())

        # TODO: add mAP, check how many small, medium, large objects we missed

    truthXY = torch.cat(truthXY, 0).numpy()

    # check how many times we have a truth on a grid edge
    check_lands_on_grid_edge(gridEdges, truthXY, "truths")

    # plot histograms of xy cens
    plot_histograms(truthXY, width=netParams["width"], height=netParams["height"],
                        outPath=outPath, name="truths")

    # create tensors of each head
    for idx, vals in filteredPredXY.items():
        filteredPredXY[idx] = torch.cat(vals, 0).numpy()

    plot_objectness_xy(filteredPredXY, outPath, name="preds_afterNMS", width=netParams["width"],
                       height=netParams["height"], strides=strides)

    # flatten dictionary for histogram plots
    filteredPredXY = np.concatenate(list(filteredPredXY.values()), 0)
    plot_histograms(filteredPredXY[:, :2], width=netParams["width"], height=netParams["height"],
                    outPath=outPath, name="preds_afterNMS")

    # check how many predictions land on a grid edge
    check_lands_on_grid_edge(gridEdges, filteredPredXY, "predictions")


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
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--cfg", type=str, help="Path to model cfg file", required=True)
    # parser.add_argument("--weights", type=str, help="Path to model weights", required=True)
    # parser.add_argument("--data", type=str, help="Path to *.data file", required=True)
    # parser.add_argument("--names", type=str, help="Path to *.names file", required=True)
    # parser.add_argument("--width", type=int, default=None, help="Width of network for inference. Defaults to width in cfg if not given")
    # parser.add_argument("--height", type=int, default=None, help="Height of network for inference. Defaults to height in cfg if not given")
    # parser.add_argument("--device", type=str, default=0, help="Device to use for inference, 'cpu' or gpu number (0,1,...)")
    # parser.add_argument("--use16bit", action="store_true", help="Load the images as 16 bit images")
    # parser.add_argument("--output", type=str, help="Path to save output")
    # args = parser.parse_args()
    #
    # imgShape = None if None in [args.width, args.height] else [args.height, args.width]
    # imgType = np.uint16 if args.use16bit else np.uint8
    # names = parse_names(args.names)
    # study_predictions(outPath=args.output,
    #                   outName="", # TODO: change this to be the name of the data file
    #                   modelCfg=args.cfg,
    #                   modelWeights=args.weights,
    #                   dataFile=args.data,
    #                   names=names,
    #                   imgShape=imgShape,
    #                   device=args.device,
    #                   imgType=imgType)