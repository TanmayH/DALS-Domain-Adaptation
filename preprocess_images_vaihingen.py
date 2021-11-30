from PIL import Image
import numpy as np
import os
import re
import cv2

def get_all_images_in_folder(folder, need_test=True, put_in_test=False):
    images = []
    idx = 0
    total_images = len([filename for filename in os.listdir(folder) \
                        if "building" in filename and "mask" not in filename and "gt" not in filename])
    
    print("Total images = ", total_images)
    
    for filename in sorted(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            # Resizing cuz bing needs to be the same size as vaihingen
            img = cv2.resize(img, (64, 64), interpolation = cv2.INTER_AREA)
            suffix = "input" if "mask" not in filename else "label"

            # Filtering all building masks
            if "gt" in filename or "all_buildings" in filename:
                continue

            print("Processing file = ", filename, " shape = ", img.shape) 
            
            # Cuz mask is 3d, and network demands only 2d mask, smh :/
            if "mask" in filename:
                img = img[:, :, 0]
            
            # Parsing the id of the image
            name = "".join([x for x in filename if x.isdigit()])
            
            # Returns blank in some cases
            if name == "":
                continue

            id = int(name)
            if not put_in_test:
                if not need_test:
                    # Valid/Train split 70/30
                    if id < 0.7 * total_images:
                        np.save("./dataset/Train/" + name + "_" + suffix, img)
                    else:
                        np.save("./dataset/Valid/" + name + "_" + suffix, img)
                else:
                    # Train/Valid/Test split 50/25/25
                    if id < 0.5 * total_images:
                        np.save("./dataset/Train/" + name + "_" + suffix, img)
                    elif id < 0.75 * total_images:
                        np.save("./dataset/Valid/" + name + "_" + suffix, img)
                    else:
                        np.save("./dataset/Test/" + name + "_" + suffix, img)
            else:
                np.save("./dataset/Test/" + name + "_" + suffix, img)

get_all_images_in_folder("/home.ORIG/npochhi/buildings_vaihingen/buildings/")
# get_all_images_in_folder("/home.ORIG/npochhi/buildings_bing/single_buildings")
