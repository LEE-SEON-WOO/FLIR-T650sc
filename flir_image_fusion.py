
from flir_image_extractor import FlirImageExtractor

import cv2
from tqdm import tqdm
import os
def KLT(a):
    """
    Reference: 
    [1] http://fourier.eng.hmc.edu/e161/lectures/klt/node3.html
    [2] https://reference.wolfram.com/language/ref/KarhunenLoeveDecomposition.html
    [3] https://sukhbinder.wordpress.com/2014/09/11/karhunen-loeve-transform-in-python/
    
    Returns Karhunen Loeve Transform of the input and the transformation matrix and eigenval
    
    Ex:
    import numpy as np
    a  = np.array([[1,2,4],[2,3,10]])
    
    kk,m = KLT(a)
    print kk
    print m
    
    # to check, the following should return the original a
    print np.dot(kk.T,m).T
        
    """
    val,vec = np.linalg.eig(np.cov(a))
    klt = np.dot(vec,a)
    return klt,vec,val
if __name__ == "__main__":
    fie = FlirImageExtractor(exiftool_path='exiftool.exe', is_debug='True')
    filename = "FLIR0079.jpg"
    fie.process_image(filename)
    result = fie.fusion_image(alpha=0.5)
    # cv2.imshow("test", result['crop_alpha_blending'])
    cv2.imshow("test", result['fusion_image'])
    cv2.waitKey(0)
    
    # for currentdir, folders, filenames in os.walk("D:\\한빛 3발(전송용)\\JPEGImages"):
    #     save_dir ="D:\\한빛 3발(전송용)\\OriImages\\"
    #     for filename in tqdm(filenames):
    #         f_name = os.path.join(currentdir, filename)
            
    #         fie.process_image(f_name)
    #         result = fie.fusion_image(alpha=0.5)
            
    #         fie.imwrite(os.path.join(save_dir, filename), result['rgb_image'])
        