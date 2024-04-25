import os, sys
import numpy as np
import random
import json
import argparse
from tqdm import tqdm
import logging
import torch
from transformers import ViTFeatureExtractor

import cv2


def get_args():
    parser = argparse.ArgumentParser()

    # get arguments from outside
    parser.add_argument('--datadir', default='./../../Dataset', type=str, help='Dir saves the datasource information')
    parser.add_argument('--featdir', default='./../../Dataset/Features', type=str, help='which data stream')
    parser.add_argument('--cachedir', default='./../../CACHE', type=str, help='which data stream')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = get_args()
    print("Arguments:")
    [print(key, val) for key, val in vars(args).items()]

    datadir = args.datadir
    featdir = args.featdir
    cachedir = args.cachedir

    if not os.path.exists(featdir):
            os.makedirs(featdir)
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-large-patch16-224', cache_dir=cachedir)

    logging.basicConfig(filename=os.path.join(featdir, './train.log'), level=logging.INFO)

    with open(os.path.join(datadir, "train.json"), encoding="utf8") as json_file:
        traindict = json.load(json_file)
    with open(os.path.join(datadir, "dev.json"), encoding="utf8") as json_file:
        devdict = json.load(json_file)
    with open(os.path.join(datadir, "test.json"), encoding="utf8") as json_file:
        testdict = json.load(json_file)

    for dsetdict, dset in zip([devdict, testdict, traindict], ['dev', 'test', 'train']): #traindict, 'train',
        dsetsavedir = os.path.join(featdir, dset)
        if not os.path.exists(dsetsavedir):
            os.makedirs(dsetsavedir)
        logging.info('processing {dset} set')
        for i in tqdm(list(dsetdict.keys())):
            videodir = dsetdict[i]['videodir']
            video = []
            cap = cv2.VideoCapture(videodir)
            while cap.isOpened():
                ret, frame = cap.read()  # BGR
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # frame = frame.resize((224, 224))
                    video.append(frame)
                else:
                    break
            cap.release()
            if len(video) < 10:
                logging.info(i)
            else:
                videozero = np.zeros(np.shape(video[0]))
                if len(video) > 10:
                    video = random.sample(video, 10)
                else:
                    video = video
                    for i in range(11 - len(video)):
                        video.append(videozero)
                video = np.asarray(video)
                video = [feature_extractor(images=i, return_tensors="pt").data['pixel_values'].squeeze(0) for i in video ]
                video = torch.stack(video)
                torch.save(video, os.path.join(dsetsavedir, i + '.pt'))




    

