import os
import cv2
import glob
import argparse
from tqdm import tqdm

def createVid(imgPaths, folder=""):
    assert len(imgPaths) > 0, "ERROR: No Images found"
    outPath = os.path.join(args.folder, folder)
    os.makedirs(outPath, exist_ok=True)

    outName = "{}_{}.avi".format(folder, args.name)

    # get width/height for video
    img = cv2.imread(imgPaths[0])
    vidOut = cv2.VideoWriter(os.path.join(outPath, outName),
                             cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                             args.fps,
                             (img.shape[1], img.shape[0]))

    for imgPth in tqdm(imgPaths, desc="Making video {}".format(folder)):
        img = cv2.imread(imgPth)
        vidOut.write(img)

    vidOut.release()
    return

def makeVideo():
    assert os.path.exists(args.drawnDets), "Folder {} does not exist".format(args.drawnDets)

    # check for subfolders
    tempImages = glob.glob(os.path.join(args.drawnDets, "*.png"))
    if len(tempImages) == 0:
        folders = [os.path.join(args.drawnDets, x) for x in os.listdir(args.drawnDets)]
        for fold in folders:
            images = glob.glob(os.path.join(fold, "*.png"))
            createVid(images, fold.split("/")[-1])
    else:
        createVid(tempImages)

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dd', '--drawnDets', type=str, help='Path to drawn detections', required=True)
    parser.add_argument('-f', '--folder', type=str, help="Path to output folder", required=True)
    parser.add_argument('-n', '--name', type=str, default="drawnDets_video")
    parser.add_argument('--fps', default=5, help="Frames per second")
    args = parser.parse_args()
    makeVideo()