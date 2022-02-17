import sys
import os

try:
    from utils.model import *
    from utils.general import parse_cfg
except:
    sys.path.append(os.path.join(os.getcwd(), "PyTorch_YOLOv4"))
    from utils.model import *
    from utils.general import parse_cfg

import numpy as np
import torch
from tqdm import tqdm
from scipy.cluster.vq import kmeans
import math
import argparse

import DarknetController

def isAscending(A):
    A = A.tolist()
    if A == sorted(A,reverse=False):
        return True
    elif A == sorted(A,reverse=True):
        return False
    else:
        return None # this should never happen


def autoAnchors_darknet(cfg, shapes, boxes, img_size, thr=4.0):
    model, layers, strides = runModel(cfg)
    currAnchors = np.array([layers[l].anchors.numpy() for l in model.yolo_layers]).flatten()
    n = len(currAnchors) // 2
    strides = np.array(strides)
    theoreticalMinAreas = (strides*2)**2
    scaledBoxes = boxes.copy()
    scaledBoxes[:,2] *= img_size[1]
    scaledBoxes[:,3] *= img_size[0]
    datasetMinArea = np.min(scaledBoxes[:,2] * scaledBoxes[:,3])
    yoloAscending = isAscending(strides)

    if all(x > datasetMinArea for x in theoreticalMinAreas):
        print("WARNING: The current model *.cfg has a theoretical minimum detection area of {}, the smallest object "
              "in the training set has an area of {}. Modify the number of subsampling layers to increase "
              "the theoretical minimum detection area...".format(np.min(theoreticalMinAreas), datasetMinArea))

    # copied from pytorch version as well, but we only use it for k-means once
    thr = 1.0/thr

    def metric(k, wh):  # compute metrics
        r = wh[:, None] / k[None]
        x = torch.min(r, 1. / r).min(2)[0]  # ratio metric
        # x = wh_iou(wh, torch.tensor(k))  # iou metric
        return x, x.max(1)[0]  # x, best_x

    def print_results(k):
        k = k[np.argsort(k.prod(1))]  # sort small to large
        x, best = metric(k, wh0)
        bpr, aat = (best > thr).float().mean(), (x > thr).float().mean() * n  # best possible recall, anch > thr
        print('thr=%.2f: %.4f best possible recall, %.2f anchors past thr' % (thr, bpr, aat))
        print('n=%g, img_size=%s, metric_all=%.3f/%.3f-mean/best, past_thr=%.3f-mean: ' %
              (n, img_size, x.mean(), best.mean(), x[x > thr].mean()), end='')
        for i, x in enumerate(k):
            print('%i,%i' % (round(x[0]), round(x[1])), end=',  ' if i < len(k) - 1 else '\n')  # use in *.cfg
        return k

    wh0 = np.concatenate([np.expand_dims(l, axis=0)[:, 2:4] * s for s, l in zip(shapes, boxes)])  # wh
    # Filter
    i = (wh0 < 3.0).any(1).sum()
    if i:
        print('WARNING: Extremely small objects found. '
              '%g of %g labels are < 3 pixels in width or height.' % (i, len(wh0)))
    wh = wh0[(wh0 >= 2.0).any(1)]  # filter > 2 pixels

    # Kmeans calculation
    print('Running kmeans for %g anchors on %g points...' % (n, len(wh)))
    s = wh.std()  # sigmas for whitening
    k, dist = kmeans(wh / s, n, iter=1000)  # points, mean distance
    k *= s
    wh = torch.tensor(wh, dtype=torch.float32)  # filtered
    wh0 = torch.tensor(wh0, dtype=torch.float32)  # unfiltered
    k = print_results(k)

    # no longer copied from pytorch, here we move the anchors to the proper yolo layer
    oldAnchors = currAnchors.copy().reshape(-1, 2)
    copiedK = k.copy()
    newAnchors = dict()
    for idx, minArea in enumerate(theoreticalMinAreas):
        newAnchors[idx] = []

        if yoloAscending:
            if idx < len(theoreticalMinAreas) - 1:
                currMin, currMax = minArea, theoreticalMinAreas[idx+1]
            else:
                currMin, currMax = minArea, sys.float_info.max
        else:
            if idx == 0:
                currMin, currMax = minArea, sys.float_info.max
            elif idx == len(theoreticalMinAreas) - 1:
                currMin, currMax = sys.float_info.min, theoreticalMinAreas[idx-1]
            else:
                currMin, currMax = minArea, theoreticalMinAreas[idx-1]

        copiedCopy = copiedK.copy()
        removeIdxs = []
        for idx2, anchor in enumerate(copiedCopy):
            anchor = np.rint(anchor)
            currArea = anchor[0]*anchor[1]
            if yoloAscending:
                metCondition = currArea < currMax
            else:
                metCondition = currArea > currMin

            if metCondition:
                newAnchors[idx].append(anchor)
                removeIdxs.append(idx2)

        copiedK = np.delete(copiedK, removeIdxs, axis=0)

    # TODO: Clean up
    # fill empty anchors with minimum area for the respective yolo head
    masks = dict()
    lastLength = 0
    for yoloLayer, anchorList in newAnchors.items():
        if len(anchorList) == 0:
            currMin = int(math.floor(np.sqrt(theoreticalMinAreas[yoloLayer])/10.0)) * 10
            anchorList.append(np.array([currMin, currMin]))

        masks[yoloLayer] = ",".join([str(x+lastLength) for x in range(len(anchorList))])
        lastLength += len(anchorList)

    tempList = [str(int(x)) for anchorList in newAnchors.values() for anchor in anchorList for x in anchor]
    anchors = ",".join(tempList)
    return anchors, masks, len(tempList)//2

def runModel(cfg):
    s = 256
    modelDefs = parse_cfg(cfg)
    model = Darknet(modelDefs)
    layers = model.module_list
    x = torch.randn((1,3,s,s))
    out = model(x)
    strides = [s / x.shape[-2] for x in out]
    return model, layers, strides

# Stolen from: https://github.com/WongKinYiu/PyTorch_YOLOv4
# adapted for DarknetController

def kmean_anchors(boundingBoxes, shapes, n=9, img_size=640, thr=4.0, gen=1000, verbose=True):
    """ Creates kmeans-evolved anchors from training dataset

        Arguments:
            path: path to dataset *.yaml, or a loaded dataset
            n: number of anchors
            img_size: image size used for training
            thr: anchor-label wh ratio threshold hyperparameter hyp['anchor_t'] used for training, default=4.0
            gen: generations to evolve anchors using genetic algorithm

        Return:
            k: kmeans evolved anchors

        Usage:
            from utils.utils import *; _ = kmean_anchors()
    """
    thr = 1. / thr

    def metric(k, wh):  # compute metrics
        r = wh[:, None] / k[None]
        x = torch.min(r, 1. / r).min(2)[0]  # ratio metric
        # x = wh_iou(wh, torch.tensor(k))  # iou metric
        return x, x.max(1)[0]  # x, best_x

    def fitness(k):  # mutation fitness
        _, best = metric(torch.tensor(k, dtype=torch.float32), wh)
        return (best * (best > thr).float()).mean()  # fitness

    def print_results(k):
        k = k[np.argsort(k.prod(1))]  # sort small to large
        x, best = metric(k, wh0)
        bpr, aat = (best > thr).float().mean(), (x > thr).float().mean() * n  # best possible recall, anch > thr
        print('thr=%.2f: %.4f best possible recall, %.2f anchors past thr' % (thr, bpr, aat))
        print('n=%g, img_size=%s, metric_all=%.3f/%.3f-mean/best, past_thr=%.3f-mean: ' %
              (n, img_size, x.mean(), best.mean(), x[x > thr].mean()), end='')
        for i, x in enumerate(k):
            print('%i,%i' % (round(x[0]), round(x[1])), end=',  ' if i < len(k) - 1 else '\n')  # use in *.cfg
        return k


    # Get label wh
    shapes = img_size * shapes / shapes.max(1, keepdims=True)
    wh0 = np.concatenate([np.expand_dims(l, axis=0)[:,2:4] * s for s, l in zip(shapes, boundingBoxes)])  # wh

    # Filter
    i = (wh0 < 3.0).any(1).sum()
    if i:
        print('WARNING: Extremely small objects found. '
              '%g of %g labels are < 3 pixels in width or height.' % (i, len(wh0)))
    wh = wh0[(wh0 >= 2.0).any(1)]  # filter > 2 pixels

    # Kmeans calculation
    print('Running kmeans for %g anchors on %g points...' % (n, len(wh)))
    s = wh.std(0)  # sigmas for whitening
    k, dist = kmeans(wh / s, n, iter=30)  # points, mean distance
    k *= s
    wh = torch.tensor(wh, dtype=torch.float32)  # filtered
    wh0 = torch.tensor(wh0, dtype=torch.float32)  # unflitered
    k = print_results(k)

    # Plot
    # k, d = [None] * 20, [None] * 20
    # for i in tqdm(range(1, 21)):
    #     k[i-1], d[i-1] = kmeans(wh / s, i)  # points, mean distance
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    # ax = ax.ravel()
    # ax[0].plot(np.arange(1, 21), np.array(d) ** 2, marker='.')
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7))  # plot wh
    # ax[0].hist(wh[wh[:, 0]<100, 0],400)
    # ax[1].hist(wh[wh[:, 1]<100, 1],400)
    # fig.tight_layout()
    # fig.savefig('wh.png', dpi=200)

    # Evolve
    npr = np.random
    f, sh, mp, s = fitness(k), k.shape, 0.9, 0.1  # fitness, generations, mutation prob, sigma
    pbar = tqdm(range(gen), desc='Evolving anchors with Genetic Algorithm')  # progress bar
    for _ in pbar:
        v = np.ones(sh)
        while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
            v = ((npr.random(sh) < mp) * npr.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)
        fg = fitness(kg)
        if fg > f:
            f, k = fg, kg.copy()
            pbar.desc = 'Evolving anchors with Genetic Algorithm: fitness = %.4f' % f
            if verbose:
                print_results(k)

    return print_results(k)

def check_anchor_order(anchors, strides):
    # Check anchor order against stride order for YOLO Detect() module m, and correct if necessary
    a = anchors.prod(-1).view(-1)  # anchor area
    da = a[-1] - a[0]  # delta a
    ds = strides[-1] - strides[0]  # delta s
    if da.sign() != ds.sign():  # same order
        print('Reversing anchor order')
        anchors[:] = anchors.flip(0)
    return anchors

def autoAnchors_pytorch(cfg, ogShapes, boxes, img_size, thr=4.0):
    model, layers, strides = runModel(cfg)
    currAnchors = np.array([layers[l].anchors.numpy() for l in model.yolo_layers]).flatten()
    shapes = img_size[0] * ogShapes[:,::-1] / ogShapes.max(1, keepdims=True)
    scale = np.random.uniform(0.9, 1.1, size=(shapes.shape[0], 1))
    wh = torch.tensor(np.concatenate([np.expand_dims(l, axis=0)[:,2:4] * s for s, l in zip(shapes * scale, boxes)])).float()

    def metric(k):  # compute metric
        r = wh[:, None] / k[None]
        x = torch.min(r, 1. / r).min(2)[0]  # ratio metric
        best = x.max(1)[0]  # best_x
        aat = (x > 1. / thr).float().sum(1).mean()  # anchors above threshold
        bpr = (best > 1. / thr).float().mean()  # best possible recall
        return bpr, aat

    newAnchors = None
    currAnchorsT = currAnchors.copy().reshape(-1,2)
    bpr, aat = metric(currAnchorsT)
    print('anchors/target = %.2f, Best Possible Recall (BPR) = %.4f' % (aat, bpr), end='')
    if bpr < 0.98:  # threshold to recompute
        print('. Attempting to generate improved anchors, please wait...' % bpr)
        na = newAnchors.numel() // 2 if newAnchors is not None else len(currAnchors) # number of anchors
        temp_anchors = kmean_anchors(boxes, ogShapes, n=na, img_size=img_size[0], thr=thr, gen=1000, verbose=False)
        new_bpr = metric(temp_anchors.reshape(-1, 2))[0]
        if new_bpr > bpr:  # replace anchors
            temp_anchors = torch.tensor(temp_anchors).type_as(newAnchors)
            newAnchors = temp_anchors.clone().view_as(newAnchors)
            newAnchors = check_anchor_order(newAnchors, strides)
            print('New anchors saved to model. Update model *.cfg to use these anchors in the future.')
        else:
            print('Original anchors better than new anchors. Proceeding with original anchors.')
    print('')  # newline

    return newAnchors

# call as individual script
def calc_anchors():
    trainTxt = args.trainTxt
    classes = args.classes
    datasetName = args.experimentName
    ogCfg = args.cfg

    outPath = os.path.join(os.getcwd(), datasetName + "_Anchors")
    os.makedirs(outPath, exist_ok=True)
    dc = DarknetController.DarknetController(darknetPath="")
    trainInfo, testInfo = dc.verifyDataset(outPath, classes=classes, trainTxt=trainTxt, testTxt=trainTxt,
                                           netShape=[args.trainHeight, args.trainWidth], clearCache=True)
    namesFile = dc.createNamesFile(outPath, datasetName=datasetName, classes=classes)
    dataFile, weightsPath = dc.createDataFile(outPath, datasetName, trainTxt, trainTxt, namesFile, len(classes))
    cfgFile = dc.createCfgFile(outPath, datasetName, ogCfg, numClasses=len(classes), trainInfo=trainInfo,
                               trainHeight=args.trainHeight,
                               trainWidth=args.trainWidth, channels=args.channels, subdivisions=64,
                               maxBatches=None, auto_anchors=args.autoAnchors)
    print("Done! CFG with adjusted anchors is located in {}".format(os.path.abspath(outPath)))
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-name', '--experimentName', help="The name of the experiment/dataset you're training on", required=True)
    parser.add_argument('-train', '--trainTxt', type=str, help="Path to training text file", required=True)
    parser.add_argument('--cfg', type=str, help="Path to cfg file to utilize", required=True)
    parser.add_argument('--trainWidth', type=int, help="Width of network input", required=True)
    parser.add_argument('--trainHeight', type=int, help="Height of network input", required=True)
    parser.add_argument('--channels', type=int, help="Channels of network input", required=True)
    parser.add_argument('--classes', nargs='+', type=str, default=["target"], help="Names of classes (must be in order of class index)")
    parser.add_argument('--autoAnchors', default=2, type=int, help="Use 1 for auto anchors calculated like PyTorch Implementation,"
                                                                   " use 2 for how Alexey recommends on GitHub")
    args = parser.parse_args()
    calc_anchors()