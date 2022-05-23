from turtle import st
import cv2 as cv
import numpy as np
import json
import os


def convertPolygonToMask(jsonfilePath):
    with open(jsonfilePath, "r", encoding='utf-8') as jsonf:
        jsonData = json.load(jsonf)
        img_h = jsonData["imageHeight"]
        img_w = jsonData["imageWidth"]
        # 图片中目标的数量 num=len(jsonData["shapes"])
        num = 0
        for obj in jsonData["shapes"]:
            mask = np.zeros((img_h, img_w), np.uint8)
            label = obj["label"]
            polygonPoints = obj["points"]
            polygonPoints = np.array(polygonPoints, np.int32)
            # print("+" * 50, "\n", polygonPoints)
            print(label)
            num += 1
            cv.drawContours(mask, [polygonPoints], -1, (255), -1)
            cv.imwrite(maskSaveFolder + "/mask_"+str(num)+".png", mask)
    return mask


if __name__ == "__main__":
    # main()
    jsonfilePath = "/Users/maoring/Developer/Cell-Nuclei-Detection-and-Segmentation/train/data/nuclear/stage1_train/005/images/005.json"
    maskSaveFolder = "/Users/maoring/Developer/Cell-Nuclei-Detection-and-Segmentation/train/data/nuclear/stage1_train/005/masks"
    mask = convertPolygonToMask(jsonfilePath)
    # 为了可视化把mask做一下阈值分割
    # _, th = cv.threshold(mask, 0, 255, cv.THRESH_BINARY)
