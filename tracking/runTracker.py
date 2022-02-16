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
from KCF_Tracker.KCFTracker import KCFTracker

class ImageLoader():
    """ Loads image paths from text file, this must be modified for different preprocessing.
        This should also be adjusted if the images contribute to more than one video
    """
    def __init__(self, txtPath):
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
        img = cv2.imread(imgPath, cv2.IMREAD_UNCHANGED)
        self.count += 1
        return img

def main():
    dataloader = ImageLoader(args.txtFile)
    darknet = Darknet(args.darknetPath)
    network, class_names, colors = darknet.load_network(args.cfg, args.dataFile, args.weights)

    tracker = KCFTracker()
    times = []
    for idx, img in enumerate(tqdm(dataloader, desc="Running Tracker w/ YOLO")):
        h, w, ch = img.shape
        darknetImg = darknet.make_image(w, h, ch)
        darknet.copy_image_from_bytes(darknetImg, img.tobytes())

        t1 = time.time()
        preds = darknet.detect_image(network, class_names, darknetImg, thresh=0.005, hier_thresh=0, nms=0.45, diou=1, letterbox=False)
        darknet.free_image(darknetImg)

        t2 = time.time()
        times.append(t2 - t1)

    times = np.asarray(times)
    print("Min Time Elapsed: {} ms ({} FPS)".format(round(times.min() * 1000, 5), round(1/times.min(), 5)))
    print("Avg Time Elapsed: {} ms ({} FPS)".format(round(times.mean() * 1000, 5), round(1/times.mean(), 5)))
    print("Max Time Elapsed: {} ms ({} FPS)".format(round(times.max() * 1000, 5), round(1/times.max(), 5)))
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--darknetPath', type=str, help='Path to darknet executable', required=True)
    parser.add_argument("-f", "--folder", type=str, help="Path to output folder", required=True)
    parser.add_argument("-w", "--weights", type=str, help="Path to weights", required=True)
    parser.add_argument("-t", "--txtFile", type=str, help="Path to text file containing list of test images", required=True)
    parser.add_argument("-c", "--cfg", type=str, help="Path to CFG file", required=True)
    parser.add_argument("-name", "--outputName", type=str, default="results.avi", help="Name for saved video")
    parser.add_argument('--classes', nargs='+', type=str, default=["target"], help="Names of classes (must be in order of class index)")
    parser.add_argument('--gpus', default=[0], type=int, nargs='+', help="GPUs available")
    args = parser.parse_args()
    print("WARNING: THIS IS CURRENTLY ONLY AN EXAMPLE FOR 1 VIDEO WITH 1 OBJECT")
    main()