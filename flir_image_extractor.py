#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import io
import json
import os
import os.path
import re
import csv
import subprocess
from PIL import Image
from math import sqrt, exp, log
from matplotlib import cm
from matplotlib import pyplot as plt
from collections import OrderedDict

import cv2
import numpy as np
import glob
from tqdm import tqdm

class FlirImageExtractor:

    def __init__(self, exiftool_path="exiftool", is_debug=False):
        self.exiftool_path = exiftool_path
        self.is_debug = is_debug
        self.flir_img_filename = ""
        self.rgb_image_filename = ""
        self.image_suffix = "_rgb_image.jpg"
        self.thumbnail_suffix = "_rgb_thumb.jpg"
        self.thermal_suffix = "_thermal.png"
        self.grayscale_suffix = "_grayscale.png"
        self.default_distance = 1.0

        # valid for PNG thermal images
        self.use_thumbnail = False
        self.fix_endian = True

        self.rgb_image_np = None
        self.thermal_image_np = None
        self.normalized_image_np = None
    pass

    def process_image(self, flir_img_filename):
        """
        Given a valid image path, process the file: extract real thermal values
        and a thumbnail for comparison (generally thumbnail is on the visible spectre)
        :param flir_img_filename:
        :return:
        """
        #if self.is_debug:
            # print("INFO Flir image filepath:{}".format(flir_img_filename))
            
        
        if not os.path.isfile(flir_img_filename):
            print(flir_img_filename)
            raise ValueError("Input file does not exist or this user don't have permission on this file")

        self.flir_img_filename = flir_img_filename
        fn_prefix, _ = os.path.splitext(self.flir_img_filename)
        self.rgb_image_filename = fn_prefix + self.image_suffix
        
        if self.get_image_type().upper().strip() == "TIFF":
            # valid for tiff images from Zenmuse XTR
            self.use_thumbnail = True
            self.fix_endian = False

        self.rgb_image_np = self.extract_embedded_image()
        self.thermal_image_np = self.extract_thermal_image()
        self.normalized_image_np = (self.thermal_image_np - np.amin(self.thermal_image_np)) / (np.amax(self.thermal_image_np) - np.amin(self.thermal_image_np))

    def get_image_type(self):
        """
        Get the embedded thermal image type, generally can be TIFF or PNG
        :return:
        """
        meta_json = subprocess.check_output(
            [self.exiftool_path, '-RawThermalImageType', '-j', self.flir_img_filename])
        meta = json.loads(meta_json.decode())[0]

        return meta['RawThermalImageType']

    def get_rgb_np(self):
        """
        Return the last extracted rgb image
        :return:
        """
        return self.rgb_image_np

    def get_thermal_np(self):
        """
        Return the last extracted thermal image
        :return:
        """
        return self.thermal_image_np

    def extract_embedded_image(self):
        """
        extracts the visual image as 2D numpy array of RGB values
        """
        image_tag = "-EmbeddedImage"
        if self.use_thumbnail:
            image_tag = "-ThumbnailImage"

        visual_img_bytes = subprocess.check_output([self.exiftool_path, image_tag, "-b", self.flir_img_filename])
        visual_img_stream = io.BytesIO(visual_img_bytes)

        visual_img = Image.open(visual_img_stream)
        visual_np = np.array(visual_img)

        return visual_np

    def extract_thermal_image(self):
        """
        extracts the thermal image as 2D numpy array with temperatures in oC
        """

        # read image metadata needed for conversion of the raw sensor values
        # E=1,SD=1,RTemp=20,ATemp=RTemp,IRWTemp=RTemp,IRT=1,RH=50,PR1=21106.77,PB=1501,PF=1,PO=-7340,PR2=0.012545258
        meta_json = subprocess.check_output(
            [self.exiftool_path, self.flir_img_filename, '-Emissivity', '-SubjectDistance', '-AtmosphericTemperature',
             '-ReflectedApparentTemperature', '-IRWindowTemperature', '-IRWindowTransmission', '-RelativeHumidity',
             '-PlanckR1', '-PlanckB', '-PlanckF', '-PlanckO', '-PlanckR2',
             '-OffsetX', '-Real2IR', '-OffsetY', '-PiPX1', '-PiPX2', '-PiPY1', '-PiPY2', '-j'])
        meta = json.loads(meta_json.decode())[0]
        self.PiPTags = {'Real2IR' : meta['Real2IR'],
                                'OffsetX' : meta['OffsetX'],
                                'OffsetY' : meta['OffsetY'],
                                'PiPX1' : meta['PiPX1'],
                                'PiPX2' : meta['PiPX2'],
                                'PiPY2' : meta['PiPY2'],
                                'PiPY1' : meta['PiPY1']}
        # exifread can't extract the embedded thermal image, use exiftool instead
        thermal_img_bytes = subprocess.check_output([self.exiftool_path, "-RawThermalImage", "-b", self.flir_img_filename])
        thermal_img_stream = io.BytesIO(thermal_img_bytes)

        thermal_img = Image.open(thermal_img_stream)
        thermal_np = np.array(thermal_img)

        # raw values -> temperature
        subject_distance = self.default_distance
        if 'SubjectDistance' in meta:
            subject_distance = FlirImageExtractor.extract_float(meta['SubjectDistance'])

        if self.fix_endian:
            # fix endianness, the bytes in the embedded png are in the wrong order
            thermal_np = np.vectorize(lambda x: (x >> 8) + ((x & 0x00ff) << 8))(thermal_np)

        raw2tempfunc = np.vectorize(lambda x: FlirImageExtractor.raw2temp(x, E=meta['Emissivity'], OD=subject_distance,
                                                                          RTemp=FlirImageExtractor.extract_float(
                                                                              meta['ReflectedApparentTemperature']),
                                                                          ATemp=FlirImageExtractor.extract_float(
                                                                              meta['AtmosphericTemperature']),
                                                                          IRWTemp=FlirImageExtractor.extract_float(
                                                                              meta['IRWindowTemperature']),
                                                                          IRT=meta['IRWindowTransmission'],
                                                                          RH=FlirImageExtractor.extract_float(
                                                                              meta['RelativeHumidity']),
                                                                          PR1=meta['PlanckR1'], PB=meta['PlanckB'],
                                                                          PF=meta['PlanckF'],
                                                                          PO=meta['PlanckO'], PR2=meta['PlanckR2']))
        thermal_np = raw2tempfunc(thermal_np)
        return thermal_np

    @staticmethod
    def raw2temp(raw, E=1, OD=1, RTemp=20, ATemp=20, IRWTemp=20, IRT=1, RH=50, PR1=21106.77, PB=1501, PF=1, PO=-7340,
                 PR2=0.012545258):
        """
        convert raw values from the flir sensor to temperatures in C
        # this calculation has been ported to python from
        # https://github.com/gtatters/Thermimage/blob/master/R/raw2temp.R
        # a detailed explanation of what is going on here can be found there
        """

        # constants
        ATA1 = 0.006569
        ATA2 = 0.01262
        ATB1 = -0.002276
        ATB2 = -0.00667
        ATX = 1.9

        # transmission through window (calibrated)
        emiss_wind = 1 - IRT
        refl_wind = 0

        # transmission through the air
        h2o = (RH / 100) * exp(1.5587 + 0.06939 * (ATemp) - 0.00027816 * (ATemp) ** 2 + 0.00000068455 * (ATemp) ** 3)
        tau1 = ATX * exp(-sqrt(OD / 2) * (ATA1 + ATB1 * sqrt(h2o))) + (1 - ATX) * exp(
            -sqrt(OD / 2) * (ATA2 + ATB2 * sqrt(h2o)))
        tau2 = ATX * exp(-sqrt(OD / 2) * (ATA1 + ATB1 * sqrt(h2o))) + (1 - ATX) * exp(
            -sqrt(OD / 2) * (ATA2 + ATB2 * sqrt(h2o)))

        # radiance from the environment
        raw_refl1 = PR1 / (PR2 * (exp(PB / (RTemp + 273.15)) - PF)) - PO
        raw_refl1_attn = (1 - E) / E * raw_refl1
        raw_atm1 = PR1 / (PR2 * (exp(PB / (ATemp + 273.15)) - PF)) - PO
        raw_atm1_attn = (1 - tau1) / E / tau1 * raw_atm1
        raw_wind = PR1 / (PR2 * (exp(PB / (IRWTemp + 273.15)) - PF)) - PO
        raw_wind_attn = emiss_wind / E / tau1 / IRT * raw_wind
        raw_refl2 = PR1 / (PR2 * (exp(PB / (RTemp + 273.15)) - PF)) - PO
        raw_refl2_attn = refl_wind / E / tau1 / IRT * raw_refl2
        raw_atm2 = PR1 / (PR2 * (exp(PB / (ATemp + 273.15)) - PF)) - PO
        raw_atm2_attn = (1 - tau2) / E / tau1 / IRT / tau2 * raw_atm2
        raw_obj = (raw / E / tau1 / IRT / tau2 - raw_atm1_attn -
                   raw_atm2_attn - raw_wind_attn - raw_refl1_attn - raw_refl2_attn)

        # temperature from radiance
        temp_celcius = PB / log(PR1 / (PR2 * (raw_obj + PO)) + PF) - 273.15
        return temp_celcius

    @staticmethod
    def extract_float(dirtystr):
        """
        Extract the float value of a string, helpful for parsing the exiftool data
        :return:
        """
        digits = re.findall(r"[-+]?\d*\.\d+|\d+", dirtystr)
        return float(digits[0])

    def plot(self):
        """
        Plot the rgb + thermal image (easy to see the pixel values)
        :return:
        """
        rgb_np = self.get_rgb_np()
        thermal_np = self.get_thermal_np()

        plt.subplot(1, 2, 1)
        plt.imshow(thermal_np, cmap='hot')
        plt.subplot(1, 2, 2)
        plt.imshow(rgb_np)
        plt.show()

    def save_images(self, options):
        """
        Save the extracted images
        :return:
        """
        options = options.split('|')

        rgb_np = self.rgb_image_np
        thermal_np = self.thermal_image_np

        img_visual = Image.fromarray(rgb_np)
        thermal_normalized = (thermal_np - np.amin(thermal_np)) / (np.amax(thermal_np) - np.amin(thermal_np))
        img_thermal = Image.fromarray(np.uint8(cm.inferno(thermal_normalized) * 255))
        img_gray = Image.fromarray(np.uint8(cm.binary(thermal_normalized) * 255))

        fn_prefix, _ = os.path.splitext(self.flir_img_filename)
        thermal_filename = fn_prefix + self.thermal_suffix
        image_filename = fn_prefix + self.image_suffix
        grayscale_filename = fn_prefix + self.grayscale_suffix

        if self.use_thumbnail:
            image_filename = fn_prefix + self.thumbnail_suffix

        if self.is_debug:
            print("DEBUG Saving RGB image to:{}".format(image_filename))
            print("DEBUG Saving Thermal image to:{}".format(thermal_filename))
            print("DEBUG Saving Gray image to:{}".format(grayscale_filename))
        if 'rgb' in options:
            img_visual.save(image_filename)
        if 'thermal' in options:
            img_thermal.save(thermal_filename)
        if 'gray' in options:
            img_gray.save(grayscale_filename)

    def export_thermal_to_csv(self, directory=False, spec_path=None):
        """
        Convert thermal data in numpy to json
        :return:
        """
        if directory==True:
            print(self.flir_img_filename)
            dir_list = os.path.join(self.flir_img_filename, '*.jpg')
            for filename in glob.iglob(dir_list, recursive=True):
                
                csv_filename = filename[:-4] + '.csv'
                
                thermal_data = np.copy(self.thermal_image_np)
                np.savetxt(csv_filename, thermal_data, fmt="%.2f", delimiter=",")
            
            pass
        fn_prefix, _ = os.path.splitext(self.flir_img_filename)
        csv_filename = fn_prefix + '.csv'
        thermal_data = np.copy(self.thermal_image_np)
        if spec_path:
            save_path = os.path.join(spec_path, csv_filename.split(os.sep)[-1])
            np.savetxt(save_path, thermal_data, fmt="%.2f", delimiter=",")
        else:
            np.savetxt(csv_filename, thermal_data, fmt="%.2f", delimiter=",")

        # if args.normalize:
        #     np.savetxt('normalized_ '+csv_filename, thermal_data, fmt="%.2f", delimiter=",")
            # with open('normalized_'+csv_filename, 'w') as nfh:
            #     writer = csv.writer(nfh, delimiter=',')
            #     writer.writerows(self.normalized_image_np)

    def export_thermal_to_json(self):
        """
        write to json
        """
        fn_prefix, fn_ext = os.path.splitext(self.flir_img_filename)
        json_filename = fn_prefix + '.json'

        thermal_data = self.thermal_image_np

        tempRow = []
        for row in thermal_data:
            temp = []
            for celsius in row:
                temp.append(round(celsius, 2))
            tempRow.append(temp)
        
        with open(json_filename, 'w') as json_file:
            json_data = OrderedDict()
            json_data['eventId'] = os.path.abspath(fn_prefix+fn_ext)
            json_data['processingTime'] = '2019-10-16T14:58:36.998586+09:00'
            json_data['processingLabTime'] = '00:00:00.3757957'
            json_data['processingStatus'] = '0'
            json_data['tempData'] = tempRow

            # json_file.write(out)
            # json.dump(json_data, json_file)
            out = json.dumps(json_data, indent=4)
            json_file.write(out)
            


        if args.normalize:
            print("warning : json normalization is not implemented >_<")
            pass

    def load_thermal_data(self):
        """
        return thermal data array
        """
        pixel_values = []
        for e in np.ndenumerate(self.thermal_image_np):
            x, y = e[0]
            c = e[1]
            pixel_values.append([x,y,c])

        return pixel_values

    
    def export_thermal_to_np(self, flir_input, normalize=False):
        """
            Convert thermal path, open data convert temperature
        
            return Thermal Numpy
        """
        if os.path.isdir(flir_input):

            dir_list = os.path.join(flir_input,'*.jpg')
            termal_list=[]
            for filename in glob.iglob(dir_list, recursive=True):
                print(filename)
                self.process_image(filename)
                if normalize:
                    resized_img = cv2.resize(self.normalized_image_np, (640, 480))
                    termal_list.append(resized_img)
                else:
                    resized_img = cv2.resize(self.thermal_image_np, (640, 480))
                    termal_list.append(resized_img)
            #Stack Thermal-Image List
            return np.stack(termal_list)
        
        else:
            #self.process_image(flir_input)
            if normalize:
                return self.normalized_image_np
            else:
                return self.thermal_image_np

    def export_thermal_to_rgb(self, flir_input, normalize=False):
        """
            Convert thermal path, open data convert temperature
        
            return 3channel rgb
        """

        return self.rgb_image_np
    
    def imread(self, filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8): 
        try: 
            n = np.fromfile(filename, dtype) 
            img = cv2.imdecode(n, flags) 
            return img 
        except Exception as e: 
            print(e) 
            return None
        
    def imwrite(self, filename, img, params=None): 
        try: 
            ext = os.path.splitext(filename)[1] 
            result, n = cv2.imencode(ext, img, params) 
            if result: 
                with open(filename, mode='w+b') as f: 
                    n.tofile(f) 
                    return True 
            else: 
                return False 
        except Exception as e: 
            print(e) 
            return False

    def export_flir_to_rawflr(self, flir_input):
        return self.imread(flir_input)

    def fusion_image(self, alpha=1.0):
        
        
        rgb_np = self.rgb_image_np
        rgb_np = cv2.cvtColor(rgb_np, cv2.COLOR_BGR2RGB)
        
        color_rgb_np = cv2.cvtColor(rgb_np, cv2.COLOR_BGR2GRAY)
        h, w = color_rgb_np.shape[:2]
        real2ir = self.PiPTags['Real2IR']
        
        
        thermal_np = cv2.resize(self.thermal_image_np, (640, 480))
        thermal_normalized2 = cv2.normalize(thermal_np, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        scale_coord, coord = self.convert(real_img = color_rgb_np, 
                                                        thermal_img=thermal_normalized2,
                                                        piptag = self.PiPTags)
        rgb_img = cv2.resize(rgb_np, None, fx=scale_coord[0], fy=scale_coord[1])
        y1, y2, x1, x2 = coord[0], coord[1], coord[2], coord[3]
        rgb_w, rgb_h = rgb_img.shape[:2]
        orit_w, orit_h = thermal_np.shape[:2]
        #test_orit = cv2.resize(thermal_np, (x2-x1, y2-y1))
        test_orit=self.export_flir_to_rawflr(self.flir_img_filename)
        copy_rgb_img = rgb_img.copy()
        try:
            rgb_img[y1:y2, x1:x2] = test_orit
            img = cv2.addWeighted(rgb_img, alpha, copy_rgb_img, 1-alpha, 0)
            
            #crop alpha blending
            crop = img[y1:y1+test_orit.shape[0], x1:x1+test_orit.shape[1]]
            crop_rgb_img = copy_rgb_img[y1:y1+test_orit.shape[0], x1:x1+test_orit.shape[1]]
        except:
            #Rotate Image
            test_orit = np.transpose(test_orit, (1, 0, 2))
            
            rgb_img[y1:y2, x1:x2] = test_orit
            img = cv2.addWeighted(rgb_img, alpha, copy_rgb_img, 1-alpha, 0)
            crop = img[y1:y1+test_orit.shape[0], x1:x1+test_orit.shape[1]]
            crop_rgb_img = copy_rgb_img[y1:y1+test_orit.shape[0], x1:x1+test_orit.shape[1]]
            
        
        
        result = {
                        'alpha_blending':img, 
                        'crop_alpha_blending':crop, 
                        'rgb_image':copy_rgb_img,
                        'resized_thermal_img':thermal_np,
                        'fusion_image':rgb_img,
                        'crop_rgb_image':crop_rgb_img
                        }
        
        return result
    
    def real_to_ir(self, real, ir_ratio):
        return int(real*ir_ratio)
    
    def convert(self, real_img, thermal_img, piptag=None):
        import copy
        
        
        
        for k, v in piptag.items():
            if not k == 'Real2IR':
                piptag[k] = int(v)
        
        h, w = real_img.shape[:2]
        thermal_h, thermal_w = thermal_img.shape[:2]
        
        real2ir = piptag['Real2IR']

        h_resize_factor = int(h/real2ir)
        w_resize_factor = int(w/real2ir)

        h_rf = h/thermal_h/real2ir
        w_rf = w/thermal_w/real2ir

        offsetx, offsety = piptag['OffsetX']-piptag['OffsetX']//real2ir, piptag['OffsetY']-piptag['OffsetY']//real2ir
        # real_img = cv2.resize(real_img,  (w_resize_factor, h_resize_factor))
        real_img = cv2.resize(real_img, None, fx = 1/w_rf, fy=1/h_rf)

        
        h, w = real_img.shape
        
        h_center, w_center = (h//2)+int(offsety), (w//2)+int(offsetx)


        #horizontal = ()
        #vertical = (-)

        #real_img[h_center- piptags['PiPY2']: h_center+piptags['PiPY1']+1, w_center-piptags['PiPX2']-1:w_center+piptags['PiPX1']] = thermal_img
        #real_img[h_center- 240: h_center+240, w_center-320:w_center+320] = thermal_img
        x1, x2, y1, y2 =piptag['PiPX1'], (piptag['PiPX2']+1), (piptag['PiPY1']), (piptag['PiPY2']+1)
        x1, x2, y1, y2 = 320,320, 240, 240
        return (1/w_rf, 1/h_rf), (h_center- y2, h_center+ y1, w_center-x1, w_center+x2)
        
def parsing_dir(fie):
    #2분기 기성 폴더
    path ='C:\\Users\\hci-lab01\\Desktop\\khnp_test'
    #발전소 폴더
    subfolder = os.listdir(path)
    
    
    #조합
    folder_list = [os.path.join(path, i) for i in subfolder]
    parsing_xml_img_match(folder_list, fie)
    
    

## [START]: parsing
def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    import re
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]
## [END]: PARSING

## [START]: Get Parent Directory
def get_parent_dir(directory, im_format='.png'):
    sep = directory.split(os.sep)
    _dir = os.path.dirname(os.path.dirname(directory))
    return _dir, sep[-1][:-4] + im_format
## [END]: Get Parent Directory

## [START]: Choose img file Except Non Xml files.
def except_png_files(annot_path, img_path):
    # print("Todo")
    xml_lists = [annot_path for annot_path in glob.iglob(annot_path+'/*.xml')]
    png_list=[]
    for i in xml_lists:
        _, png = get_parent_dir(i)
        png_path = os.path.join(img_path, png)
        if os.path.isfile(png_path):
            png_list.append(png_path)
    return png_list
## [END]: Choose img file Except Non Xml files.

def parsing_xml_img_match(path, fie):
    
    annotation = []
    csv = [] 
    color=[]
    
    for i in path:
        #설비 리스트
        machine_list = os.listdir(i)
        #각 폴더 안 리스트
        # annotation += [k for j in machine_list  for k in glob.iglob(os.path.join(i, j, 'annotation', '*.xml'))]
        annotation += [k for k in glob.iglob(os.path.join(i, 'annotation', '*.xml'))]

    annotation.sort(key=natural_keys)
    for i in tqdm(annotation):
        _dir, _img = get_parent_dir(i, im_format='.jpg')
        img_path = os.path.join(_dir, 'color', _img)
        
        if os.path.exists(img_path):
            fie.process_image(img_path)
            csv_dir_path = os.path.join(os.path.join(_dir, 'csv'))
            if not os.path.exists(csv_dir_path):
                os.mkdir(csv_dir_path)
            fie.export_thermal_to_csv(directory=False, spec_path=csv_dir_path)
        
        
        
    
    # glob.iglob()
    
    
    
    
if __name__ == '__main__':
    flir_input = 'examples'
    #IMG Example: FLIR13447.jpg
    #DIR example: examples
    fie = FlirImageExtractor(exiftool_path='exiftool.exe', is_debug='True')
    # fie.process_image(flir_input)
    #a = fie.export_thermal_to_np(flir_input)
    #parsing_dir(fie)
    parser = argparse.ArgumentParser(description='Extract and visualize Flir Image data')
    parser.add_argument('-i', '--input', type=str, help='Input image. Ex. img.jpg', required=True)
    # parser.add_argument('-p', '--plot', help='Generate a plot using matplotlib', required=False, action='store_true')
    # parser.add_argument('-exif', '--exiftool', type=str, help='Custom path to exiftool', required=False,
    #                     default='exiftool')
    # parser.add_argument('-c', '--color', help='[rgb|thermal|gray] selec output color option')
    # parser.add_argument('-csv', '--extractcsv', help='Export the thermal data per pixel encoded as csv file',
    #                     required=True, action='store_true')
    # parser.add_argument('-d', '--debug', help='Set the debug flag', default=False, required=False,
    #                     action='store_true')
    # parser.add_argument('-n', '--normalize', help='save normalized data', required=False, action='store_true')
    args = parser.parse_args()
    fie.process_image(args.input)
    #img = fie.fusion_image()    
    img = fie.homography()
    
    cv2.imshow("test", img)
    cv2.waitKey(0)
    
    # if args.plot:
    #     fie.plot()

    # if args.extractcsv:
    #     print(args.extractcsv)        
        
    #     fie.export_thermal_to_csv(directory=False)
    # if args.extractjson:
    #     print(args.extractjson)
    #     fie.export_thermal_to_json()
    # if args.color:
    #     print(args.color)
    #     fie.save_images(args.color)
    
