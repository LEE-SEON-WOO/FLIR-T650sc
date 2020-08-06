
from flir_image_extractor import FlirImageExtractor

import cv2

if __name__ == "__main__":
    fie = FlirImageExtractor(exiftool_path='exiftool.exe', is_debug='True')
    fie.process_image("FLIR0081.jpg")
    result = fie.fusion_image(alpha=0.2)
    print(result['alpha_blending'].shape)
    cv2.imshow("test", result['crop_alpha_blending'])
    cv2.waitKey(0)