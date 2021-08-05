import os
import sys
sys.path.append("../")
from DarknetController import DarknetController
from logger import Logger
from general import setupGPU
import argparse

def checkForTruth(txtFile):
    with open(txtFile, "r") as f:
        lines = f.readlines()

    truthFiles = []
    for l in lines:
        imgPath = l.strip()
        name, ext = os.path.splitext(imgPath.split("/")[-1])
        txtPath = imgPath.replace(ext, ".txt")
        if os.path.exists(txtPath):
            truthFiles.append(txtPath)

    return len(truthFiles) > 0


def test():
    # make sure we have everything we need
    assert os.path.exists(args.cfg), "Cfg not found at {}".format(args.cfg)
    assert os.path.exists(args.weights), "Weights not found at {}".format(args.weights)
    assert os.path.exists(args.txtFile), "Text file not found at {}".format(args.txtFile)

    expName = args.txtFile.split("/")[-1].replace(".txt", "")
    # make output folder if it doesn't exist
    outPath = os.path.abspath(args.folder)
    os.makedirs(outPath, exist_ok=True)

    # set up darknet
    logger = Logger(outputPath=outPath)
    dc = DarknetController(darknetPath=args.darknetPath, logger=logger)

    # make necessary files
    namesFile = dc.createNamesFile(outPath, datasetName=expName, classes=args.classes)
    dataFile, _ = dc.createDataFile(outPath, expName, args.txtFile, args.txtFile,
                                    namesFile, len(args.classes), makeWeightsFolder=False)

    # run test
    predJsons = dc.test(outPath, [args.weights], dataFile, args.cfg, args.txtFile)

    # if we have truth, run evaluation
    if checkForTruth(args.txtFile):
        imageList, groundtruths, extractedPreds = dc.evalDarknetJsons(predJsons, args.txtFile, drawDets=args.drawDets,
                                                                      noDualEval=args.atdTypeEval)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--darknetPath', type=str, help='Path to darknet executable', required=True)
    parser.add_argument("-f", "--folder", type=str, help="Path to output folder", required=True)
    parser.add_argument("-w", "--weights", type=str, help="Path to weights", required=True)
    parser.add_argument("-t", "--txtFile", type=str, help="Path to text file containing list of test images", required=True)
    parser.add_argument("-c", "--cfg", type=str, help="Path to CFG file", required=True)
    parser.add_argument('--classes', nargs='+', type=str, default=["target"], help="Names of classes (must be in order of class index)")
    parser.add_argument('--gpus', default=[0], type=int, nargs='+', help="GPUs available")
    parser.add_argument('--atdTypeEval', action='store_true', help="Run evaluation without marking multiple TP dets as incorrect.")
    parser.add_argument('--drawDets', action='store_true', help="Draw detections for each weights file")
    args = parser.parse_args()

    test()