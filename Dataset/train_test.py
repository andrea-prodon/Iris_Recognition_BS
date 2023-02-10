import os
import shutil
from skimage.filters.rank import equalize
from skimage.morphology import disk
import numpy as np


#prendiamo solo le prime 48 righe che non include alcune parti di ciglia e la palpebra
def ImageEnhancement(normalized_iris):
    row=64
    col=512
    normalized_iris = normalized_iris.astype(np.uint8)
    
    
    enhanced_image=normalized_iris
     
    enhanced_image = equalize(enhanced_image, disk(32))
    
    roi = enhanced_image[0:48,:]
    return roi


src_folder = 'Dataset/CASIA_Iris_interval_norm/'

dst_folder_1 = 'Dataset/train'
dst_folder_2 = 'Dataset/test'

num = 1

for folder in os.listdir(src_folder):
    new_folder = src_folder + folder
    elements = os.listdir(new_folder)

    for element in elements:

        if os.path.isfile(os.path.join(new_folder, element)):
            if element[4] == '1':
                dst_folder = dst_folder_1 + "/" + str(num) + "/"
                if not os.path.exists(dst_folder):
                    os.makedirs(dst_folder)
                element = ImageEnhancement(element)
                shutil.copy(os.path.join(new_folder, element), dst_folder)
    

            elif element[4] == '2':
                dst_folder = dst_folder_2 + "/" + str(num) +"/"
                if not os.path.exists(dst_folder):
                    os.makedirs(dst_folder)
                element = ImageEnhancement(element)
                shutil.copy(os.path.join(new_folder, element), dst_folder)
    num+=1