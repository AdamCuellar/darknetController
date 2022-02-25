import sys
import os
cwd = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(cwd, '..')))

import argparse
import cv2
from tqdm import tqdm
import time
import numpy as np

from DarknetWrapper import Darknet
from KCF_Tracker.KCFTrackManager import TrackManager
from KCF_Tracker.tools.detection import Detection
import faulthandler
faulthandler.enable()

class ImageLoader():
    """ Loads image paths from text file, this must be modified for different preprocessing.
        This should also be adjusted if the images contribute to more than one video
    """
    def __init__(self, txtPath):
        self.count = 0
        with open(txtPath, "r") as f:
            self.imagePaths = f.readlines()

    def __iter__(self):
        self.count = 0
        return self

    def __len__(self):
        return len(self.imagePaths)

    def __next__(self):
        if self.count == len(self.imagePaths):
            raise StopIteration

        imgPath = self.imagePaths[self.count].strip()
        img = cv2.imread(imgPath) #, cv2.IMREAD_UNCHANGED)
        # res, img = cv2.imreadmulti(imgPath, None, cv2.IMREAD_UNCHANGED)
        # img = cv2.merge(img)
        self.count += 1
        return img

    def reset(self):
        self.count = 0

def drawBox(img, obj, color=(0, 255, 0), confThresh=0., putText=False, normalize=False):
    if normalize:
        img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # [frame index, class name, confidence, bounding box as xyxy)
    if obj[2] >= confThresh:
        bbox = obj[3].astype(int)
        cv2.rectangle(img, tuple(bbox[:2]), tuple(bbox[2:]), color, 1)
        if putText:
            cv2.putText(img, '{}: {}'.format(obj[1], round(obj[2], 2)), tuple(bbox[:2]-10), 0, 0.3, color)

    return img

def xlyl2xyxy(box):
    if isinstance(box, list):
        box = np.asarray(box)
    
    box[2] = box[0] + box[2]
    box[3] = box[1] + box[3]
    return box

def main():
    os.makedirs(args.folder, exist_ok=True)

    dataloader = ImageLoader(args.txtFile)
    darknet = Darknet(args.darknetPath)
    network, class_names, colors = darknet.load_network(args.cfg, args.classes, args.weights)

    # get image for height width of video
    img = next(dataloader)
    h, w, ch = img.shape

    dataloader.reset()

    # create video
    vidOut = cv2.VideoWriter(os.path.join(args.folder, args.outputName),
                             cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                             args.fps,
                             (w, h))

    times = []
    tracker = TrackManager(useKCF=True)
    for idx, img in enumerate(tqdm(dataloader, desc="Running Tracker w/ YOLO")):
        h, w, ch = img.shape
        darknetImg = darknet.make_image(w, h, ch)
        darknet.copy_image_from_bytes(darknetImg, img.tobytes())

        t1 = time.time()
        preds = darknet.detect_image(network, class_names, darknetImg, thresh=0.9, hier_thresh=0, nms=0.45, diou=0, letterbox=False)
        darknet.free_image(darknetImg)

        dets = []
        for pred in preds:
            box = pred[-1]
            xmin = box[0] - box[2] / 2
            ymin = box[1] - box[3] / 2
            newBox = [xmin, ymin, box[2], box[3]]
            dets.append(Detection(newBox, pred[1], 0))
        
        tracker.predict()
        tracker.update(dets, img)

        t2 = time.time()
        times.append(t2 - t1)

        drawn = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        for i, pred in enumerate(tracker.tracks):
            trackedBox = pred.to_tlbr()
            # draw tracked box in green
            drawn = drawBox(drawn, [idx, "target", 1., trackedBox], color=(0, 255, 0))

        vidOut.write(drawn)

    vidOut.release()
    times = np.asarray(times)
    print("Min Time Elapsed: {} ms ({} FPS)".format(round(times.min() * 1000, 5), round(1/times.min(), 5)))
    print("Avg Time Elapsed: {} ms ({} FPS)".format(round(times.mean() * 1000, 5), round(1/times.mean(), 5)))
    print("Max Time Elapsed: {} ms ({} FPS)".format(round(times.max() * 1000, 5), round(1/times.max(), 5)))
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--darknetPath', type=str, help='Path to darknet folder', required=True)
    parser.add_argument("-f", "--folder", type=str, help="Path to output folder", required=True)
    parser.add_argument("-w", "--weights", type=str, help="Path to weights", required=True)
    parser.add_argument("-t", "--txtFile", type=str, help="Path to text file containing list of test images", required=True)
    parser.add_argument("-c", "--cfg", type=str, help="Path to CFG file", required=True)
    parser.add_argument("-name", "--outputName", type=str, default="results.avi", help="Name for saved video")
    parser.add_argument('--classes', nargs='+', type=str, default=["target"], help="Names of classes (must be in order of class index)")
    parser.add_argument('--gpu', default=0, type=int, help="GPU available")
    parser.add_argument('--fps', default=1, type=int, help="Number of Frames per Second for output video")
    args = parser.parse_args()
    print("WARNING: THIS IS CURRENTLY ONLY AN EXAMPLE FOR 1 VIDEO WITH 1 OBJECT")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    main()