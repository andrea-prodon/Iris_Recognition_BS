import cv2
import os
import numpy as np
from skimage import feature
#dependencies
#python -m pip install -U scikit-image

#local binary pattern
def lbp(image, npoint, radius=8, eps=1e-7):
    lbp = feature.local_binary_pattern(image, npoint,radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, npoint + 3),range=(0, npoint + 2))
    print(lbp.shape)
    # normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)
    return lbp, hist


def img_show(label: str,img: np.ndarray):
    cv2.imshow(label, img)
    # here it should be the pause
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()

if __name__ == '__main__':
    input_path = "Dataset/CASIA1/1/"
    img = cv2.imread(os.path.join(input_path,os.listdir(input_path)[6]))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp_img, lbp_code = lbp(gray,24)
    print(lbp_code.shape)
    img_show('original', img)
    img_show('lbp', lbp_img)
