from darkflow.net.build import TFNet
import cv2
import numpy as np
import os
import glob
import json
from utils.files import getBaseName, createFolder
#labels20 = ["aeroplane", "bicycle", "bird", "boat", "bottle",
#    "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
#    "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
#    "train", "tvmonitor"]

options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.1}

tfnet = TFNet(options)



def filter_origin(img,from_color,to_color):
    img = np.copy(img)
    replace_color_from = from_color #bgr
    replace_color_to = to_color
    img[np.where((img==replace_color_from).all(axis=2))] = replace_color_to
    cv2.imshow('aaa',img)
    
    return img
def filter_label_color(img):
    img = np.copy(img)
    img = filter_origin(img,[255,0,0],[0,0,0])
    img = filter_origin(img,[0,255,0],[0,0,0])
    img = filter_origin(img,[0,0,255],[0,0,0])
    img = filter_origin(img,[255,255,0],[0,0,0])
    img = filter_origin(img,[255,0,255],[0,0,0])
    img = filter_origin(img,[0,255,255],[0,0,0])
    return img

def binary_color_filter(img, color):
    img = np.copy(img)
    lower_blue = np.array(color) 
    upper_blue = np.array(color) 
    mask = cv2.inRange(img, lower_blue, upper_blue) 
    #result = cv2.bitwise_and(img, img, mask = mask) 
    return mask


set_of_car = ['car', 'bicycle','bus', 'motorbike', 'person','train']
    

dataset = [ 'tlchia-dataset-v2_day' ]
def main(path):
    dataset_name = path.split('/')[0]
    filename = os.path.basename(path)
    filename = filename.split('.')[0]
    print(path,filename)

    imgcv = cv2.imread(path)
    dim = (1920,1080)
    imgcv = cv2.resize(imgcv, dim) #, interpolation = cv2.INTER_LINEAR

    result = tfnet.return_predict(imgcv)

    f_result =  [a for a in result if a['label']  in set_of_car]
    img = np.copy(imgcv)
    h,w,c = img.shape
    img = filter_label_color(img)
    print(w,h, c)
    img_filtered_origin = np.copy(img)

    createFolder('out')
    createFolder('out/'+dataset_name)
    createFolder('out/'+dataset_name+'/origin')
    createFolder('out/'+dataset_name+'/binary')
    createFolder('out/'+dataset_name+'/yolo_data')

    with open('out/'+dataset_name+'/yolo_data/'+filename+'.json', 'w') as f:
        json.dump(str(f_result), f)

    
    for r in f_result:
        print(r)
        x1=r['topleft']['x']
        y1=r['topleft']['y']
        x2=r['bottomright']['x']
        y2=r['bottomright']['y']
        color=[]
        if r['label'] in ['car','motorbike']:
            color = [0,255,0]
            cv2.rectangle(img_filtered_origin, (x1, y1), (x2, y2),color,cv2.FILLED)
            """
            cv2.imshow("img_nakagami", img_filtered_origin)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            """

        """
        if r['label']=='car':
            if x2>w/2:
                color = [0,255,255]
            else:
                color = [255,0,0]
        elif r['label']=='motorbike':
            if x2>w/2:
                color = [255,0,255]
            else:
                color=[0,255,0]
        elif r['label']=='bicycle':
            if x2>w/2:
                color = [255,255,0]
            else:
                color = [0,0,255]
        else:
            continue
        """
    
    
    

    img_b = binary_color_filter(img_filtered_origin, [0,255,0])
    cv2.imwrite('out/'+dataset_name+'/origin/'+filename+'.png',img_filtered_origin)
    cv2.imwrite('out/'+dataset_name+'/binary/'+filename+'.png',img_b)
    #cv2.imshow("1", img_b)
    #cv2.imshow("2", img_filtered_origin)
for d in dataset:
    files = glob.glob(d+'/*.jpg')
    for f in files:
        print(f)
        main(f)
