import os
import sys
import subprocess
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import cv2
import eval_utils
import json
from tqdm import tqdm
from autoAnchors import autoAnchors_pytorch, autoAnchors_darknet
from dataset_utils import verifyDataHelper
from logger import Logger

class DarknetController():
    def __init__(self, darknetPath, logger=None):
        self.darknetPath = os.path.abspath(darknetPath)
        self.defaultMaxBatches = 10000
        self.anchorInfo = None
        self.logger = logger if logger is not None else Logger("")

    def verifyDataset(self, outputPath, classes, trainTxt, testTxt, netShape, clearCache=False):
        os.makedirs(outputPath, exist_ok=True)
        trainInfo = verifyDataHelper(self.logger, outputPath, trainTxt, netShape, classes, clearCache)
        if testTxt == trainTxt:
            testInfo = copy.deepcopy(trainInfo)
        else:
            testInfo = verifyDataHelper(self.logger, outputPath, testTxt, netShape, classes, clearCache)
        return trainInfo, testInfo

    def createNamesFile(self, outputPath, datasetName, classes):
        namesTxt = os.path.join(outputPath, "{}.names".format(datasetName))
        with open(namesTxt, "w") as f:
            f.writelines("{}\n".format(x) for x in classes)

        return namesTxt

    def createDataFile(self, outputPath, datasetName, trainTxt, testTxt, namesFile, numClasses):
        # make weights folder
        weightsPath = os.path.join(outputPath, "weights")
        os.makedirs(weightsPath, exist_ok=True)

        # format lines
        lines = ["classes = {}".format(numClasses), "train = {}".format(trainTxt), "valid = {}".format(testTxt),
                 "names = {}".format(namesFile), "backup = {}".format(weightsPath)]

        # write out data file
        dataFile = os.path.join(outputPath, "{}.data".format(datasetName))
        with open(dataFile, "w") as f:
            f.writelines("{}\n".format(x) for x in lines)

        return dataFile, weightsPath

    def createCfgFile(self, outputPath, datasetName, ogCfg, numClasses, trainInfo, trainHeight, trainWidth, channels, subdivisions=64, maxBatches=None, burn_in=False, auto_anchors=0, lr=None):

        if maxBatches is None:
            # calculate max batches (recommended here: https://github.com/AlexeyAB/darknet)
            potentialNum = numClasses*2000
            maxBatches = self.defaultMaxBatches

            if (potentialNum > 6000) and (potentialNum > trainInfo["NumImages"]):
                maxBatches = potentialNum
            else:
                # make sure we train for more than the amount of images we have
                while maxBatches < trainInfo["NumImages"]:
                    maxBatches = maxBatches + self.defaultMaxBatches
        else:
            if maxBatches < trainInfo["NumImages"]:
                self.logger.warn("The training will run for {} iterations."
                                 " It's recommended you train for more than the amount of images ({})".format(maxBatches, trainInfo["NumImages"]))

        # create new cfg file
        ogCfgName = ogCfg.split("/")[-1]
        cfgName = ogCfgName.replace(".cfg", "_{}.cfg".format(datasetName))
        cfgFile = os.path.join(outputPath, cfgName)

        # read original cfg and its info
        with open(ogCfg, "r") as f:
            cfgLines = f.readlines()

        ogCfgBlocks = list()
        for line in cfgLines:
            stripped = self._stripLine(line)
            if len(stripped) == 0: continue

            if self._isHeader(stripped):
                ogCfgBlocks.append((stripped, dict()))
            else:
                attr, value = self._getAttributeValue(stripped)
                ogCfgBlocks[-1][1][attr] = value

        # copy and modify cfg attributes as necessary
        newCfg = copy.deepcopy(ogCfgBlocks)

        anchors, masks, numAnchors = None, None, None
        if auto_anchors > 0:
            if self.anchorInfo is not None:
                anchors, masks, numAnchors = self.anchorInfo
            elif auto_anchors == 1:
                # TODO: change this to return the same as below
                pass
                # anchors = autoAnchors_pytorch(ogCfg, trainInfo["Shapes"], trainInfo["Boxes"], img_size=[trainHeight, trainWidth])
            elif auto_anchors == 2:
                anchors, masks, numAnchors = autoAnchors_darknet(ogCfg, trainInfo["Shapes"], trainInfo["Boxes"], img_size=[trainHeight, trainWidth])

        self.anchorInfo = (anchors, masks, numAnchors)

        numYolo = 0
        for idx, (header, attrDict) in enumerate(newCfg):
            nextHeader = newCfg[idx+1][0] if idx+1 < len(newCfg) else None

            if "net" in header:
                self._modifyNetParams(attrDict, trainWidth, trainHeight, channels, subdivisions, maxBatches, burn_in=burn_in, lr=lr)

            elif "convolution" in header and "yolo" in nextHeader:
                # modify the yolo params first, cause the filters depends on the yolo head masks
                yoloDict = newCfg[idx+1][1]
                currMasks = masks[numYolo] if masks is not None else None
                self._modifyYoloParams(yoloDict, numClasses, anchors=anchors, masks=currMasks, numAnchors=numAnchors)

                numMasks = len(yoloDict["mask"].split(","))
                self._modifyConvParams(attrDict, numClasses, numMasks)
                numYolo += 1

        # write out new cfg
        with open(cfgFile, "w") as f:
            for header, attrDict in newCfg:
                f.write("{}\n".format(header))
                for attr, value in attrDict.items():
                    f.write("{}={}\n".format(attr, value))
                f.write("\n")

        return cfgFile

    def train_multiGPU(self, outputPath, dataFile, cfgFile1, cfgFile2, preWeights, gpus=[0], burnAmount=1000, doMap=True, dontShow=False, clear=False):
        self.train(outputPath, dataFile, cfgFile1, preWeights, gpu=gpus[0], doMap=False, dontShow=dontShow, printTime=False, clear=clear)
        weightsPath = os.path.join(outputPath, "weights")
        weights = [os.path.join(weightsPath, x) for x in os.listdir(weightsPath) if "_burn_in" in x and "_{}.weights".format(burnAmount) in x]
        weights = weights[0]
        self.train(outputPath, dataFile, cfgFile2, weights, gpu=",".join(gpus), doMap=doMap, dontShow=dontShow)
        return

    def train(self, outputPath, dataFile, cfgFile, preWeights, gpu=None, doMap=True, dontShow=False, printTime=True, clear=False):
        outputPath = os.path.abspath(outputPath)
        dataFile = os.path.abspath(dataFile)
        cfgFile = os.path.abspath(cfgFile)
        preWeights = os.path.abspath(preWeights) if preWeights else None

        recordTrainTxt = os.path.join(outputPath, "train_std.txt")

        # change to output path (for saving the charts)
        currPath = os.getcwd()
        os.chdir(outputPath)

        # format training command
        trainLine = "{} detector train {} {} ".format(self.darknetPath, dataFile, cfgFile)

        # add necessary flags
        if preWeights:
            trainLine += "{} ".format(preWeights)

        if doMap:
            trainLine += "{} ".format("-map")

        if gpu:
            trainLine += "-gpus {} ".format(str(gpu))

        if dontShow:
            trainLine += "-dont_show "
            
        if clear:
            trainLine += "-clear "

        # record output to file
        trainLine += "2>&1 | tee {}".format(recordTrainTxt)

        # start training
        t1 = time.time()
        self.logger.print("#"*50)
        self.logger.print()
        subprocess.call(trainLine, shell=True)
        self.logger.print("#"*50)
        self.logger.print()
        t2 = time.time()

        totalTime = t2-t1
        if totalTime < 60.0:
            self.logger.warn("The training seems to be a little short, check {} for errors.".format(recordTrainTxt))
            assert True, ""

        # go back to original directory
        os.chdir(currPath)
        if printTime: self.logger.print("Training finished in {} seconds".format(totalTime))
        return

    def validate(self, outputPath, weightsPath, dataFile, cfgFile, skipExisting=False):

        # get all the weights
        weightFiles = sorted([os.path.join(weightsPath, x) for x in os.listdir(weightsPath) if ".weights" in x])
        resultsPath = os.path.join(outputPath, "resultsByWeights")
        resultTxts = []

        # run the map command for each weight file
        for weights in weightFiles:
            if "_burn_in" in weights: continue # skip the burn_in weights saved from the multi-gpu process
            # make txt file and directory for results
            weightsName =  weights.split("/")[-1].replace(".weights", "")
            txtName = weightsName + "_results.txt"
            currPath = os.path.join(resultsPath, weightsName)

            # skip weights we already processed, if requested
            if skipExisting and os.path.exists(currPath): continue

            os.makedirs(currPath, exist_ok=True)
            txtFile = os.path.join(currPath, txtName)
            resultTxts.append(txtFile)

            # call map
            mapLine = "{} detector map {} {} {} 2>&1 | tee {}".format(self.darknetPath, dataFile, cfgFile, weights, txtFile)
            t1 = time.time()
            subprocess.call(mapLine, shell=True)
            t2 = time.time()
            totalTime = t2-t1
            if totalTime < 60.0:
                self.logger.warn("Validation went much faster than expected, make sure there were no errors here: {}".format(txtFile))

        # plot the pdet vs roc if it exists
        self._plotROCs(resultsPath, resultTxts)

        #TODO: filter by best AUC?
        return resultsPath, weightFiles

    def test(self, resultsPath, weightFiles, dataFile, cfgFile, testTxt):

        predJsons = []
        # run the test command for each weight file
        for weights in weightFiles:
            if "_burn_in" in weights: continue  # skip the burn_in weights saved from the multi-gpu process

            # make json file and get directory for results
            weightsName = weights.split("/")[-1].replace(".weights", "")
            jsonName = weightsName + "_predictions.json"
            currPath = os.path.join(resultsPath, weightsName)
            os.makedirs(currPath, exist_ok=True)
            jsonFile = os.path.join(currPath, jsonName)
            predJsons.append(jsonFile)

            # call test
            testLine = "{} detector test {} {} {} -thresh 0.05 -dont_show -ext_output -out {} < {}"
            testLine = testLine.format(self.darknetPath, dataFile, cfgFile, weights, jsonFile, testTxt)
            t1 = time.time()
            subprocess.call(testLine, shell=True)
            t2 = time.time()
            totalTime = t2-t1
            if totalTime < 60.0:
                self.logger.warn("Testing went much faster than expected, make sure there were no errors above")

        return predJsons

    def drawActivations(self):
        # TODO: implement grad-cam
        pass

    def _isHeader(self, line):
        return line.startswith("[") and line.endswith("]")

    def _stripLine(self, line):
        # this removes comments and whitespace
        sep = '#'
        stripped = line.strip().split(sep, 1)[0]
        return stripped

    def _getAttributeValue(self, line):
        sep = "="
        splitLine = line.split(sep)
        attribute = splitLine[0].strip()
        value = splitLine[-1].strip()
        return attribute, value

    def _modifyNetParams(self, attrDict, trainWidth, trainHeight, channels, subdivisions, maxBatches, burn_in=True, lr=None):
        if "subdivisions" in attrDict:
            attrDict["subdivisions"] = str(subdivisions)

        if "width" in attrDict:
            attrDict["width"] = str(trainWidth)

        if "height" in attrDict:
            attrDict["height"] = str(trainHeight)

        if "channels" in attrDict:
            attrDict["channels"] = str(channels)

        if "max_batches" in attrDict:
            attrDict["max_batches"] = str(maxBatches)

        if lr is not None and "learning_rate" in attrDict:
            attrDict["learning_rate"] = str(lr)

        # this should only happen when we're training multiGPU
        if burn_in:
            if "burn_in" in attrDict:
                attrDict["burn_in"] = str(maxBatches)

        if "steps" in attrDict and not burn_in:
            # TODO: changing the learning rate multiple times isn't working well, for COCO they only do it twice. Explore other CFG's to see what the best option is
            numSteps = 2 #int(np.floor(maxBatches/self.defaultMaxBatches) * 2)
            steps = []
            for i in range(numSteps):
                currPercent = round(1.0 - 0.1*(i+1), 2)
                currSteps = int(maxBatches * currPercent)
                steps.append(currSteps)
            steps = [str(x) for x in sorted(steps)]
            stepsLine = ",".join(steps)
            attrDict["steps"] = stepsLine

            if "scales" in attrDict:
                scalesLine = ",".join([".1"]*numSteps)
                attrDict["scales"] = scalesLine

        if "mosaic" in attrDict:
            #TODO: maybe turn it on or off
            pass

        if "letter_box" in attrDict:
            # this is a useless feature, turn it off
            del attrDict["letter_box"]

        return

    def _modifyConvParams(self, attrDict, numClasses, numCurrMasks):
        # calculate number of filters for conv before each yolo head
        # recommended here: https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects
        if "filters" in attrDict:
            numFilters = (numClasses + 5)*numCurrMasks
            attrDict["filters"] = str(numFilters)
        return

    def _modifyYoloParams(self, attrDict, numClasses, anchors=None, masks=None, numAnchors=None):
        if "anchors" in attrDict and anchors is not None:
            attrDict["anchors"] = anchors

        if "mask" in attrDict and masks is not None:
            attrDict["mask"] = masks

        if "num" in attrDict and numAnchors is not None:
            attrDict["num"] = str(numAnchors)

        if "classes" in attrDict:
            attrDict["classes"] = str(numClasses)
        return

    def _plotROCs(self, resultsPath, txtFiles, infoFrom="darknet"):
        allRoc = os.path.join(resultsPath, "all_darknet_rocs_{}.png".format(infoFrom))
        resultsByWeights = dict()
        for txtFile in txtFiles:
            currWeightName = txtFile.split("/")[-1].replace(".weights", "")
            currResults = self._plotROC(txtFile)
            resultsByWeights[currWeightName] = currResults

        labels = []
        for label, results in resultsByWeights.items():
            labels.append(label)
            currColor = np.random.rand(3,)
            score, pDets, fars = results
            plt.plot(fars, pDets, c=currColor, marker='o', markersize=2.0, label=label)

        plt.grid()
        plt.title("Comparing ROCs")
        plt.xlabel("False Alarm (per Frame)")
        plt.ylabel("Probability of Detection")
        # place legend outside of figure to make sure it doesn't cover the graph(s)
        plt.legend(labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='xx-small')
        plt.savefig(allRoc, bbox_inches='tight', dpi=300)
        plt.clf()
        return

    def _plotROC(self, txtFile, infoFrom="darknet"):
        weightsName = txtFile.split("/")[-1].replace(".txt", "")
        imageFile = txtFile.replace(".txt", "_ROC_{}.png".format(infoFrom))
        scores, pDets, fars = [], [], []
        with open(txtFile, "r") as f:
            lines = f.readlines()

        for line in lines:
            if "pdet" in line.lower():
                splitLine = line.strip().split(", ")
                score, pdet, far = self._getPDetFar(splitLine)
                scores.append(score)
                pDets.append(pdet)
                fars.append(far)

        # plot roc
        plt.plot(fars, pDets, c='b', marker='o', markersize=2.0)
        plt.grid()
        plt.title(weightsName)
        plt.xlabel("False Alarm (per Frame)")
        plt.ylabel("Probability of Detection")
        plt.savefig(imageFile, bbox_inches='tight', dpi=300)
        plt.clf()

        return [scores, pDets, fars]

    def _getPDetFar(self, statList):
        stats = []
        for stat in statList:
            currStat = float(stat.split(": ")[-1])
            stats.append(currStat)
        return stats[0:3]

    def evalDarknetJsons(self, predJsons, testTxt, drawDets=False, noDualEval=False):
        extractedPreds = dict()
        groundtruths, imageList, imageSizes = self._parseGt(testTxt)
        for jsonFile in tqdm(predJsons, desc="Evaluating Darknet Preds"):
            outPath, jsonName = os.path.split(jsonFile)
            jsonName = jsonName.replace(".json", "")
            predictions = self._parseDarknetJson(jsonFile, imageSizes)
            resultDict, pDet, far = eval_utils.evalMetrics(groundtruths, predictions, numImages=len(imageSizes), noDualEval=noDualEval)
            extractedPreds[jsonFile] = (predictions, resultDict['TPByIndex'], resultDict['FPByIndex'])
            eval_utils.drawPlots(resultDict['ThreshDict'], len(groundtruths), outputPath=outPath, outputName=jsonName,
                      numImages=len(imageSizes))

            if drawDets:
                eval_utils.drawDetections(outPath, imageList, groundtruths, predictions, resultDict['TPByIndex'], resultDict['FPByIndex'])

        return imageList, groundtruths, extractedPreds

    def _parseDarknetJson(self, jsonFile, imageSizes=None):
        detections = []
        with open(jsonFile, "r") as f:
            detectionsList = json.load(f)

        for detDict in detectionsList:
            frameId = detDict['frame_id'] - 1
            dets = detDict['objects']
            filename = detDict['filename']

            if imageSizes:
                imgH, imgW = imageSizes[filename]
            else:
                img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
                imgH = img.shape[0]
                imgW = img.shape[1]

            for object in dets:
                conf = object["confidence"]
                clsId = object["class_id"]
                coords = object["relative_coordinates"]
                xcen = coords["center_x"] * imgW
                ycen = coords["center_y"] * imgH
                width = coords["width"] * imgW
                height = coords["height"] * imgH
                xmin = xcen - (width / 2)
                xmax = xcen + (width / 2)
                ymin = ycen - (height / 2)
                ymax = ycen + (height / 2)
                bbox = [xmin, ymin, xmax, ymax]
                bbox = np.array([int(round(x, 0)) for x in bbox])
                detections.append([frameId, clsId, conf, bbox])

        return detections

    def _parseGt(self, txtFile):
        imageSizes = dict()
        imageList = list()
        groundtruths = []
        with open(txtFile, "r") as f:
            imgPaths = f.readlines()

        for idx, imgPath in enumerate(imgPaths):
            imgPath = imgPath.strip()
            imageList.append(imgPath)
            img = cv2.imread(imgPath, cv2.IMREAD_UNCHANGED)
            imgH = img.shape[0]
            imgW = img.shape[1]

            # record height and width so we don't have to read all the images again
            imageSizes[imgPath] = (imgH, imgW)

            fn, ext = os.path.splitext(imgPath)
            txtPath = imgPath.replace(ext, ".txt")

            with open(txtPath, "r") as f:
                truthLines = f.readlines()

            for truth in truthLines:
                line = truth.strip().split(" ")
                line = [float(x) for x in line]
                line[0] = int(line[0])
                line[1] = line[1] * imgW
                line[2] = line[2] * imgH
                line[3] = line[3] * imgW
                line[4] = line[4] * imgH
                xmin = line[1] - (line[3] / 2)
                xmax = line[1] + (line[3] / 2)
                ymin = line[2] - (line[4] / 2)
                ymax = line[2] + (line[4] / 2)
                bbox = [xmin, ymin, xmax, ymax]
                bbox = np.array([int(round(x, 0)) for x in bbox])
                groundtruths.append([idx, line[0], 1.0, bbox])

        return groundtruths, imageList, imageSizes