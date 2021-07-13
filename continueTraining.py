import os
from DarknetController import DarknetController
from logger import Logger
from general import setupGPU
import eval_utils

import sys
sys.path.append("PyTorch_YOLOv4")
from utils.general import parse_cfg
from utils.datasets import parseDataFile

import argparse
import time
import pickle

def continue_training():
    # get information from darknetController experiment
    varsFile = os.path.join(args.folder, "allVars.p")
    assert os.path.exists(varsFile), "allVars.p file missing from {}." \
                                     " Can't continue training without it :(".format(args.folder)

    localVars = pickle.load(open(varsFile, "rb"))

    # get weights file to start training from
    weights = [x for x in localVars["weightFiles"] if args.weights in os.path.basename(x).split("_")[-1]]
    assert len(weights) > 0, "Weights '{}' not found in {}".format(args.weights, localVars["weightsPath"])
    weights = weights[0]
    assert os.path.exists(weights), "{} file not found :(".format(weights)

    # setup darknet controller
    outPath = os.path.dirname(localVars["dataFile"])
    expName = "continued"
    logger = Logger(outputPath=outPath)
    dc = DarknetController(darknetPath=args.darknetPath, logger=logger)

    # parse the cfg and data file
    modelParams = parse_cfg(localVars["cfgFile"])[0]
    numIterations = modelParams["max_batches"] + args.numIterations
    dataInfo = parseDataFile(localVars["dataFile"])
    testTxt = dataInfo["valid"]

    # adjust lr unless directed otherwise
    if args.adjustLR:
        lr = modelParams["learning_rate"]
        currIterations = int(args.weights) if args.weights.isnumeric() else modelParams["max_batches"]
        for numSteps in modelParams["steps"].split(","):
            if currIterations > int(numSteps):
                lr *= 0.1

    # setup new cfg
    trainInfo = localVars["trainInfo"]
    newCfg = dc.createCfgFile(outPath, expName, localVars["cfgFile"], numClasses=trainInfo["NumClasses"],
                              trainInfo=trainInfo, trainHeight=modelParams["height"], trainWidth=modelParams["width"],
                              channels=modelParams["channels"], subdivisions=modelParams["subdivisions"], maxBatches=numIterations, lr=lr)

    # make the selected gpu(s) the only one we can see, that way we can use it for all steps
    multiGPU = setupGPU(args.gpus)

    if multiGPU:
        gpus = [str(i) for i in range(len(args.gpus))]  # need to start from zero since we set the CUDA_VISIBLE_DEVICES flag
    else:
        gpus = 0

    dc.train(outPath, localVars["dataFile"], newCfg, weights, gpu=gpus, doMap=True, dontShow=args.dont_show, clear=False)
    resultsPath, weightFiles = dc.validate(outPath, localVars["weightsPath"], localVars["dataFile"], newCfg, skipExisting=True)
    predJsons = dc.test(resultsPath, weightFiles, localVars["dataFile"], newCfg, testTxt)
    imageList, groundtruths, extractedPreds = dc.evalDarknetJsons(predJsons, testTxt, drawDets=args.drawDets, noDualEval=args.atdTypeEval)

    # TODO: explainable AI stuff automatically
    localVars = {"trainInfo":trainInfo,
                 "testInfo":localVars["testInfo"],
                 "namesFile":localVars["namesFile"],
                 "dataFile":localVars["dataFile"],
                 "weightsPath":localVars["weightsPath"],
                 "cfgFile":newCfg,
                 "resultsPath":resultsPath,
                 "weightFiles":weightFiles,
                 "predJsons":predJsons,
                 "imageList":imageList,
                 "groundtruths":groundtruths,
                 "extractedPreds":extractedPreds}

    with open(os.path.join(outPath, "allVars.p"), "wb") as f:
        pickle.dump(localVars, f)

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--darknetPath', type=str, help='Path to darknet executable', required=True)
    parser.add_argument("-f", "--folder", type=str, help="Path to experiment folder", required=True)
    parser.add_argument("-w", "--weights", type=str, default="last", help="Number or name of weight file to start training from ex. last or best or 1000. Defaults to last")
    parser.add_argument("-n", "--numIterations", type=int, help="Number of iterations to train (not total, just how many more)")
    parser.add_argument('--adjustLR', action='store_true', help="Adjust LR to value last seen at training.")
    parser.add_argument('--dont_show', action='store_true', help="Don't show training plot (necessary for domino)")
    parser.add_argument('--gpus', default=[0], type=int, nargs='+', help="GPUs available")
    parser.add_argument('--atdTypeEval', action='store_true', help="Run evaluation without marking multiple TP dets as incorrect.")
    parser.add_argument('--drawDets', action='store_true', help="Draw detections for each weights file")
    args = parser.parse_args()

    assert os.path.exists(args.folder), "{} does not exist!".format(args.folder)
    continue_training()