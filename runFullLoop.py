import os
from DarknetController import DarknetController
from logger import Logger
from general import setupGPU
import eval_utils
import argparse
import time
import pickle

def main():
    trainTxt = args.trainTxt
    testTxt = args.testTxt
    classes = args.classes
    datasetName = args.experimentName
    preWeights = args.pretrainWeights
    ogCfg = args.cfg
    outPath = os.path.join(os.getcwd(), datasetName + "_{}".format(time.strftime("%Y%m%d-%H%M%S")))
    os.makedirs(outPath, exist_ok=True)
    logger = Logger(outPath)

    # make the selected gpu(s) the only one we can see, that way we can use it for all steps
    multiGPU = setupGPU(args.gpus)

    dc = DarknetController(darknetPath=args.darknetPath, logger=logger)
    trainInfo, testInfo = dc.verifyDataset(outPath, classes=classes, trainTxt=trainTxt, testTxt=testTxt, netShape=[args.trainHeight, args.trainWidth], clearCache=args.ignoreCache)
    namesFile = dc.createNamesFile(outPath, datasetName=datasetName, classes=classes)
    dataFile, weightsPath = dc.createDataFile(outPath, datasetName, trainTxt, testTxt, namesFile, len(classes))

    cfgFile = None
    if multiGPU:
        tempMaxBatches = len(args.gpus) * 1000
        gpus = [str(i) for i in range(len(args.gpus))] # need to start from zero since we set the CUDA_VISIBLE_DEVICES flag
        cfgFile1 = dc.createCfgFile(outPath, datasetName + "_burn_in", ogCfg, numClasses=len(classes), trainInfo=trainInfo, trainHeight=args.trainHeight,
                                   trainWidth=args.trainWidth, channels=args.channels, subdivisions=args.subdivisions, maxBatches=tempMaxBatches, burn_in=True, auto_anchors=args.autoAnchors)
        cfgFile2 = dc.createCfgFile(outPath, datasetName, ogCfg, numClasses=len(classes), trainInfo=trainInfo, trainHeight=args.trainHeight,
                                   trainWidth=args.trainWidth, channels=args.channels, subdivisions=args.subdivisions, maxBatches=args.maxBatches, auto_anchors=args.autoAnchors)
        dc.train_multiGPU(outPath, dataFile, cfgFile1, cfgFile2, preWeights, gpus=gpus, burnAmount=tempMaxBatches, doMap=True, dontShow=args.dont_show, clear=args.clear)
        cfgFile = cfgFile2
    else:
        gpu = 0 # since we set the CUDA_VISIBLE_DEVICES variable above, we can call any gpu 0
        cfgFile = dc.createCfgFile(outPath, datasetName, ogCfg, numClasses=len(classes), trainInfo=trainInfo, trainHeight=args.trainHeight,
                                   trainWidth=args.trainWidth, channels=args.channels, subdivisions=args.subdivisions, maxBatches=args.maxBatches, auto_anchors=args.autoAnchors)
        dc.train(outPath, dataFile, cfgFile, preWeights, gpu=gpu, doMap=True, dontShow=args.dont_show, clear=args.clear)

    resultsPath, weightFiles = dc.validate(outPath, weightsPath, dataFile, cfgFile)
    predJsons = dc.test(resultsPath, weightFiles, dataFile, cfgFile, testTxt)
    imageList, groundtruths, extractedPreds = dc.evalDarknetJsons(predJsons, testTxt, drawDets=args.drawDets, noDualEval=args.atdTypeEval)

    # TODO: explainable AI stuff automatically
    localVars = {"trainInfo":trainInfo,
                 "testInfo":testInfo,
                 "namesFile":namesFile,
                 "dataFile":dataFile,
                 "weightsPath":weightsPath,
                 "cfgFile":cfgFile,
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
    parser.add_argument('-name', '--experimentName', help="The name of the experiment/dataset you're training on", required=True)
    parser.add_argument('-train', '--trainTxt', type=str, help="Path to training text file", required=True)
    parser.add_argument('-test', '--testTxt', type=str, help="Path to testing text file", required=True)
    parser.add_argument('--cfg', type=str, help="Path to cfg file to utilize", required=True)
    parser.add_argument('--trainWidth', type=int, help="Width of network input", required=True)
    parser.add_argument('--trainHeight', type=int, help="Height of network input", required=True)
    parser.add_argument('--channels', type=int, help="Channels of network input", required=True)
    parser.add_argument('--dont_show', action='store_true', help="Don't show training plot (necessary for VPS)")
    parser.add_argument('--drawDets', action='store_true', help="Draw detections for each weights file")
    parser.add_argument('--subdivisions', type=int, default=64, help="Number of subdivisions")
    parser.add_argument('-pt', '--pretrainWeights', default=None, help="Path to pretrained weights for initialization")
    parser.add_argument('--classes', nargs='+', type=str, default=["target"], help="Names of classes (must be in order of class index)")
    parser.add_argument('--gpus', default=[0], type=int, nargs='+', help="GPUs available")
    parser.add_argument('--atdTypeEval', action='store_true', help="Run evaluation without marking multiple TP dets as incorrect.")
    parser.add_argument('--maxBatches', default=None, type=int, help="Max number of iterations for training.")
    parser.add_argument('--numInstances', default=1, type=int, help="Number of times you want to run the same experiment")
    parser.add_argument('--autoAnchors', default=0, type=int, help="Use 1 for auto anchors calculated like PyTorch Implementation,"
                                                                   " use 2 for how Alexey recommends on GitHub")
    parser.add_argument('--clear', default=False, action="store_true", help="Clear the pretrained weights to start training from iteration 0.")
    parser.add_argument('--ignoreCache', default=False, action="store_true", help="Ignore the dataset cache and iterate through seen datasets.")

    args = parser.parse_args()
    for i in range(args.numInstances):
        main()