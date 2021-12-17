from PIL import Image
import numpy as np
import os
import re
import cv2
import shutil
import random

def get_all_images_in_folder(train_folder):
    images = []
    dataCounter = 1
    random.seed(1)

    #Train - Chest XRAY
    for directory in sorted(os.listdir(train_folder)):
        print (directory)
        if "images" in directory:
            for filename in sorted(os.listdir(os.path.join(train_folder,directory))):
                if not("mask" in filename):
                    imgInput = cv2.imread(os.path.join(train_folder,directory,filename))
                    imgMask = cv2.imread(os.path.join(train_folder,"masks",filename[:-4]+"_mask.png"))
                    if imgInput is not None and imgMask is not None:
                        imgInput = cv2.resize(imgInput, (64, 64), interpolation = cv2.INTER_AREA)
                        suffixInput = "input"
                        imgMask = cv2.resize(imgMask, (64, 64), interpolation = cv2.INTER_AREA)
                        ret,imgMask = cv2.threshold(imgMask,70,1,cv2.THRESH_BINARY)
                        suffixMask = "label"
                        print(imgInput.shape, filename, imgInput.max())
                        print(imgMask.shape, filename+"_mask", imgMask.max())
                        nonZeroCount = np.count_nonzero(imgMask)/np.size(imgMask)  
                        if (nonZeroCount < 0.03): #removing images that dont have lung
                            continue
                        imgMask = imgMask[:, :, 0]
                        images.append([str(dataCounter) + "_" + suffixInput,imgInput,str(dataCounter) + "_" + suffixMask,imgMask])
                        dataCounter += 1
    totalImages = len(images)
    random.shuffle(images)
    for i in range(0,int(0.7 * totalImages)):
        np.save("../dataset/Train/" + images[i][0], images[i][1])
        np.save("../dataset/Train/" + images[i][2], images[i][3])
    for i in range(int(0.7 * totalImages),int(0.85 * totalImages)):
        np.save("../dataset/Valid/" + images[i][0], images[i][1])
        np.save("../dataset/Valid/" + images[i][2], images[i][3])
    for i in range(int(0.85 * totalImages),int(totalImages)):
        np.save("../dataset/Test/" + images[i][0], images[i][1])
        np.save("../dataset/Test/" + images[i][2], images[i][3])
 


for root, dirs, files in os.walk('../dataset/Train/'):
    for f in files:
        os.unlink(os.path.join(root, f))
    for d in dirs:
        shutil.rmtree(os.path.join(root, d))
for root, dirs, files in os.walk('../dataset/Test/'):
    for f in files:
        os.unlink(os.path.join(root, f))
    for d in dirs:
        shutil.rmtree(os.path.join(root, d))
for root, dirs, files in os.walk('../dataset/Valid/'):
    for f in files:
        os.unlink(os.path.join(root, f))
    for d in dirs:
        shutil.rmtree(os.path.join(root, d))
for root, dirs, files in os.walk('../network'):
    for f in files:
        os.unlink(os.path.join(root, f))
    for d in dirs:
        shutil.rmtree(os.path.join(root, d))        
get_all_images_in_folder("/home.ORIG/npochhi/Lung Segmentation/")
'''
Replace the path in the above method call with path to Lung_Segmentation/ folder
'''