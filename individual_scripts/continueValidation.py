import os
import sys
sys.path.append(os.getcwd())

from DarknetController import DarknetController
from logger import Logger
from general import setupGPU
import eval_utils

sys.path.append(os.path.join(os.getcwd(), "PyTorch_YOLOv4"))
from utils.datasets import parseDataFile

import argparse
import time
import pickle
import glob

def getVars():
    # only gets necessary variables
    localVars = dict()
    localVars["dataFile"] = glob.glob(args.folder + "*.data")[0]
    localVars["weightsPath"] = os.path.join(args.folder, "weights")
    localVars["cfgFile"] = glob.glob(args.folder + "*.cfg")[0]
    localVars["namesFile"] = glob.glob(args.folder + "*.names")[0]
    return localVars

def continue_validation():
    # get information from darknetController experiment
    varsFile = os.path.join(args.folder, "allVars.p")

    localVars = pickle.load(open(varsFile, "rb")) if os.path.exists(varsFile) else getVars()

    # setup darknet controller
    outPath = os.path.dirname(localVars["dataFile"])
    logger = Logger(outputPath=outPath)
    dc = DarknetController(darknetPath=args.darknetPath, logger=logger)

    # parse the data file
    dataInfo = parseDataFile(localVars["dataFile"])
    testTxt = dataInfo["valid"]

    # make the selected gpu(s) the only one we can see, that way we can use it for all steps
    multiGPU = setupGPU(args.gpus)

    if multiGPU:
        gpus = [str(i) for i in range(len(args.gpus))]  # need to start from zero since we set the CUDA_VISIBLE_DEVICES flag
    else:
        gpus = 0

    resultsPath, weightFiles = dc.validate(outPath, localVars["weightsPath"], localVars["dataFile"], localVars["cfgFile"], skipExisting=True)
    predJsons = dc.test(resultsPath, weightFiles, localVars["dataFile"], localVars["cfgFile"], testTxt)
    imageList, groundtruths, extractedPreds = dc.evalDarknetJsons(predJsons, testTxt, drawDets=args.drawDets, noDualEval=args.atdTypeEval)

    # TODO: explainable AI stuff automatically
    
    if os.path.exists(varsFile):
        localVars = {"trainInfo":localVars['trainInfo'],
                     "testInfo":localVars["testInfo"],
                     "namesFile":localVars["namesFile"],
                     "dataFile":localVars["dataFile"],
                     "weightsPath":localVars["weightsPath"],
                     "cfgFile":localVars["cfgFile"],
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
    parser.add_argument('--gpus', default=[0], type=int, nargs='+', help="GPUs available")
    parser.add_argument('--atdTypeEval', action='store_true', help="Run evaluation without marking multiple TP dets as incorrect.")
    parser.add_argument('--drawDets', action='store_true', help="Draw detections for each weights file")
    args = parser.parse_args()

    assert os.path.exists(args.folder), "{} does not exist!".format(args.folder)
    continue_validation()