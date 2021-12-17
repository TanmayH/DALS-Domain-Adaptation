from PIL import Image
import numpy as np
import os
import re
import cv2
import shutil
import random

def get_all_images_in_folder(folder):
    images = []
    dataCounter = 1
    suffixInput = "input"
    suffixMask = "label"
    random.seed(1)

    #Vaihingen
    total_images = len([filename for filename in os.listdir(folder) \
                        if "building" in filename and "mask" not in filename])
    
    print("Total images = ", total_images)

    #Train
    for filename in sorted(os.listdir(folder)):
        if "mask" not in filename:
            imgInput = cv2.imread(os.path.join(folder, filename))
            imgMask = cv2.imread(os.path.join(folder, filename[0:8]+"_mask"+filename[8:]))
            if imgInput is not None and imgMask is not None:
                # Resizing cuz Bing needs to be the same size as Vaihingen
                imgInput = cv2.resize(imgInput, (64, 64), interpolation = cv2.INTER_AREA)  
                imgMask = cv2.resize(imgMask, (64, 64), interpolation = cv2.INTER_AREA)
                ret,imgMask = cv2.threshold(imgMask,70,1,cv2.THRESH_BINARY)
                print(imgInput.shape, filename, imgInput.max())
                print(imgMask.shape, filename+"_mask", imgMask.max())
                nonZeroCount = np.count_nonzero(imgMask)/np.size(imgMask)  
                if (nonZeroCount < 0.03): #removing images that dont have reasonably sized building
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
get_all_images_in_folder("/home.ORIG/npochhi/buildings_vaihingen/buildings/")
'''
Replace the path in the above method call with path to Vaihingen_Building_Dataset/buildings/ folder
'''