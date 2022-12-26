import cv2
import os
import numpy as np
from skimage import feature
#dependencies
#python -m pip install -U scikit-image

#local binary pattern
def lbp_code(image: np.ndarray, npoint: int = 24, radius: int = 8, eps: float =1e-7) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = feature.local_binary_pattern(image, npoint,radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, npoint + 3),range=(0, npoint + 2))
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
    input_path = "Dataset/CASIA_Iris_interval_norm/1/"
    img = cv2.imread(os.path.join(input_path,os.listdir(input_path)[6]))
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    n_point = 1
    radius = 1
    lbp_img, lbp_hist = lbp_code(img,n_point,radius)
    img_show('original', img)
    img_show('lbp', lbp_img)
