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

def find_xml_path(root_dir):
    results ={}
    results['xml'] = []
    results['jpg'] = []
    for currentpath, folders, files in os.walk(root_dir):
        #Currentpath ex)path/to/dir
        #folders ex)[dir1, dir2, dir3 ....]
        #files ex) [a.exe, b.txt, c.jpg ... ]
        
        #We find annotation folder and find only .xml files,
        #and then we matching .jpg file, making csv and copy flir jpg image.
        for folder in folders:
            if folder == 'annotation':
                currentpath_folder = os.path.join(currentpath, folder)
                file_list = os.listdir(currentpath_folder)
                for _file in file_list:
                    filename, extention = os.path.splitext(_file)
                    if _file.endswith('.xml'):
                        parent_dir = (os.path.abspath(os.path.join(currentpath_folder, os.pardir)))
                        #Thermal jpg flir name ex) FLIR0000.jpg
                        thermal_img_name = os.path.join(filename+'.jpg')
                        
                        #Thermal jpg flir image path   ex) path/to/folder/FLIR0000.jpg
                        thermal_img_path = os.path.join(parent_dir, thermal_img_name)
                        thermal_xml_path = os.path.join(currentpath_folder, _file)
                        #Input Thermal jpg image path
                        if os.path.exists(thermal_img_path) and os.path.exists(thermal_xml_path):
                            results['xml'].append(thermal_xml_path)
                            results['jpg'].append(thermal_img_path)
    return results

import shutil

def find_all_jpeg(root_dir, filename_set, save_path):
    """모든 jpeg파일 찾아주는 함수

    Args:
        root_dir ([type]): [찾을 최상위 경로]
        filename_set ([set]): [찾고자 하는 파일 이름(확장자 제외)]
    Returns:
        [list]: [모든 jpeg 찾은거]
    """
    
    results= []
    for currentpath, folders, files in os.walk(root_dir):
        #Currentpath ex)path/to/dir
        #folders ex)[dir1, dir2, dir3 ....]
        #files ex) [a.exe, b.txt, c.jpg ... ]
        
        #We find annotation folder and find only .xml files,
        #and then we matching .jpg file, making csv and copy flir jpg image.
        for folder in folders:
            currentpath_folder = os.path.join(currentpath, folder)
            file_list = os.listdir(currentpath_folder)
            for _file in file_list:
                filename, extention = os.path.splitext(_file)
                if _file.endswith('.jpg') and filename in filename_set:
                    _dir = os.path.abspath(os.path.join(currentpath_folder, _file))
                    #Input Thermal jpg image path
                    if os.path.exists(_dir):
                        results.append(_dir)
                        
    for jpgpath in tqdm(results):
        filename = jpgpath.split(os.sep)[-1]
        _file, ext = os.path.splitext(filename)
        #save original flir jpg
        thermal_jpg = os.path.join(save_path, 'color')
        if not os.path.exists(thermal_jpg):
            os.mkdir(thermal_jpg)
        shutil.copy2(jpgpath, thermal_jpg)
        
        #load jpgpath and parsing meta data
        fie.process_image(jpgpath)
        
        ##save thermal csv
        csv_dir = os.path.join(save_path, 'csv')
        if not os.path.exists(csv_dir):
            os.mkdir(csv_dir)
        fie.export_thermal_to_csv(spec_path=csv_dir)
        
        ##save crop rgb
        rgb_dir = os.path.join(save_path, 'rgb')
        result = fie.fusion_image(alpha=0.2)
        if not os.path.exists(rgb_dir):
            os.mkdir(rgb_dir)
        
        fie.imwrite(os.path.join(rgb_dir, filename), result['crop_rgb_image'])

from flir_image_extractor import FlirImageExtractor

def save_recursive(results):
    """[summary]
        열화상 jpg를 rgb jpeg로 crop저장과 csv파일저장        
    Args:
        results ([type]): [description]
    """
    results = results['jpg']
    for jpgpath in tqdm(results):
        parent_dir = (os.path.abspath(os.path.join(jpgpath, os.pardir)))
        if os.path.exists(jpgpath):
            jpg_dir = os.path.join(parent_dir, 'color')
            if not os.path.exists(jpg_dir):
                os.mkdir(jpg_dir)
            shutil.copy2(jpgpath, os.path.join(jpg_dir))
            
            csv_dir = os.path.join(parent_dir, 'csv')
            #Copied file extract test and extract thermal data
            copy_jpg_file = jpgpath.split(os.sep)[-1]
            copy_jpg_file_path = os.path.join(jpg_dir, copy_jpg_file)
            
            fie.process_image(copy_jpg_file_path)
            if not os.path.exists(csv_dir):
                os.mkdir(csv_dir)
            fie.export_thermal_to_csv(spec_path=csv_dir)
            
            rgb_dir = os.path.join(parent_dir, 'rgb')
            result = fie.fusion_image(alpha=0.2)
            if not os.path.exists(rgb_dir):
                os.mkdir(rgb_dir)
            fie.imwrite(os.path.join(rgb_dir, copy_jpg_file), result['crop_rgb_image'])
                
def save_for_train(results, 
                            save_path):
    """[summary]
    열파일 csv파일, crop rgb jpeg와 xml저장
    Args:
        results ([type]): [description]
        save_path ([type]): [description]
    """
    #copy to annotation xml -> save_path, 'Annoatations'
    save_xml_path = os.path.join(save_path, 'Annotations')
    for i in results['xml']:
        if not os.path.exists( save_xml_path):
            os.mkdir( save_xml_path)
        shutil.copy2(i, save_xml_path)
    #copy to thermal jpg -> save_path, 'color'
    save_jpg_path = os.path.join(save_path, 'color')
    for i in results['jpg']:
        if not os.path.exists( save_jpg_path):
            os.mkdir( save_jpg_path)
        shutil.copy2(i, save_jpg_path)
    
    results = results['jpg']
    for jpgpath in tqdm(results):
        parent_dir = (os.path.abspath(os.path.join(jpgpath, os.pardir)))
        if os.path.exists(jpgpath):
            jpg_dir = os.path.join(parent_dir, 'color')
            
            csv_dir = os.path.join(parent_dir, 'csv')
            #Copied file extract test and extract thermal data
            copy_jpg_file = jpgpath.split(os.sep)[-1]
            copy_jpg_file_path = os.path.join(jpg_dir, copy_jpg_file)
            
            
            fie.process_image(copy_jpg_file_path)
            if not os.path.exists(os.path.join(save_path, 'csv')):
                os.mkdir(os.path.join(save_path, 'csv'))
            fie.export_thermal_to_csv(spec_path=os.path.join(save_path, 'csv'))
            
            
            rgb_dir = os.path.join(save_path, 'JPEGImage')
            result = fie.fusion_image(alpha=0.2)
            if not os.path.exists(rgb_dir):
                os.mkdir(rgb_dir)
            fie.imwrite(os.path.join(rgb_dir, copy_jpg_file), result['crop_rgb_image'])
            
def find_txt(path):
    txt = os.path.join(path)
    
    if os.path.exists(txt):
        with open(txt) as f:
            a = f.readlines()
        txt_list = {os.path.splitext(i)[0] for i in a}

    return txt_list
    
        
if __name__ == '__main__':
    fie = FlirImageExtractor(exiftool_path='exiftool.exe', is_debug='True')
    results = find_xml_path(root_dir= "D:/KHNP/전원전 열화상 데이터/한빛 발전소/새 폴더/한빛 3발")
    #save_recursive(results)
    #save_for_train(results, "D:/한빛 3발(전송용)")
        #     print(thermal_img_path)
        
    #########################
    txt_set = find_txt('annotation.txt')
    txt_set = os.listdir("D:/한빛 3발(전송용)/미미")
    txt_set = {os.path.splitext(i)[0] for i in txt_set}
    
    find_all_jpeg(root_dir= "D:/KHNP/전원전 열화상 데이터/고리 발전소/고리3발 목록분류(1차 분류 완료)",
                            filename_set=txt_set,
                            save_path = "D:/한빛 3발(전송용)/미미")
    
    
    
    