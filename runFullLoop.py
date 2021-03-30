import os
from DarknetController import DarknetController
import detectionEvaluation as detEval
import argparse
import time
import pickle

def main():
    dc = DarknetController(darknetPath=args.darknetPath)
    trainTxt = args.trainTxt
    testTxt = args.testTxt
    classes = args.classes
    datasetName = args.experimentName
    preWeights = args.pretrainWeights
    ogCfg = args.cfg
    outPath = os.path.join(os.getcwd(), datasetName + "_{}".format(time.strftime("%Y%m%d-%H%M%S")))
    trainInfo, testInfo = dc.verifyDataset(outPath, trainTxt=trainTxt, testTxt=testTxt)
    namesFile = dc.createNamesFile(outPath, datasetName=datasetName, classes=classes)
    dataFile, weightsPath = dc.createDataFile(outPath, datasetName, trainTxt, testTxt, namesFile, len(classes))

    cfgFile = None
    if len(args.gpus) > 1:
        args.gpus = [str(x) for x in args.gpus]
        os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(args.gpus)
        gpus = [str(i) for i in range(len(args.gpus))]
        cfgFile1 = dc.createCfgFile(outPath, datasetName + "_first1000", ogCfg, numClasses=len(classes), trainInfo=trainInfo, trainHeight=args.trainHeight,
                                   trainWidth=args.trainWidth, channels=args.channels, subdivisions=args.subdivisions, maxBatches=1000)
        cfgFile2 = dc.createCfgFile(outPath, datasetName, ogCfg, numClasses=len(classes), trainInfo=trainInfo, trainHeight=args.trainHeight,
                                   trainWidth=args.trainWidth, channels=args.channels, subdivisions=args.subdivisions)
        dc.train_multiGPU(outPath, dataFile, cfgFile1, cfgFile2, preWeights, gpus=gpus, doMap=True, dontShow=args.dont_show)
        cfgFile = cfgFile2
    else:
        # make the selected gpu the only one darknet can see, that way we can use it for all steps train/test/validate
        gpu = args.gpus[0]
        os.environ['CUDA_VISIBLE_DEVICES'] = "{}".format(gpu)
        gpu = 0
        cfgFile = dc.createCfgFile(outPath, datasetName, ogCfg, numClasses=len(classes), trainInfo=trainInfo, trainHeight=args.trainHeight,
                                   trainWidth=args.trainWidth, channels=args.channels, subdivisions=args.subdivisions)
        dc.train(outPath, dataFile, cfgFile, preWeights, gpu=gpu, doMap=True, dontShow=args.dont_show)

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
    parser.add_argument('--dont_show', action='store_true', help="Don't show training plot (necessary for domino)")
    parser.add_argument('--drawDets', action='store_true', help="Draw detections for each weights file")
    parser.add_argument('--subdivisions', type=int, default=64, help="Number of subdivisions")
    parser.add_argument('-pt', '--pretrainWeights', default=None, help="Path to pretrained weights for initialization")
    parser.add_argument('--classes', nargs='+', type=str, default=["target"], help="Names of classes (must be in order of class index)")
    parser.add_argument('--gpus', default=[0], type=int, nargs='+', help="GPUs available")
    parser.add_argument('--atdTypeEval', action='store_true', help="Run evaluation without marking multiple TP dets as incorrect.")

    args = parser.parse_args()
    main()