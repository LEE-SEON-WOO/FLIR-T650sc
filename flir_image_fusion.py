
from flir_image_extractor import FlirImageExtractor

import cv2
from tqdm import tqdm
import os

if __name__ == "__main__":
    fie = FlirImageExtractor(exiftool_path='exiftool.exe', is_debug='True')
    filename = "images/FLIR13277.jpg"
    fie.process_image(filename)
    result = fie.fusion_image(alpha=0.2)
    
    
    # cv2.imshow("test", result['crop_alpha_blending'])
    cv2.imshow("test", result['crop_rgb_image'])
    cv2.waitKey(0)
    
    