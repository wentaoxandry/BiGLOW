import json
import os, sys
import random

import pandas as pd
import argparse


def get_args():
    parser = argparse.ArgumentParser()

    # get arguments from outside
    parser.add_argument('--datadir', default='./../../Dataset', type=str, help='Dir saves the processed meta information')
    parser.add_argument('--sourcedatadir', default='./../../Dataset', type=str, help='Dir saves the source data')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    print("Arguments:")
    [print(key, val) for key, val in vars(args).items()]

    datadir = args.datadir
    sourcedatadir = args.sourcedatadir
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    datasetinfordir = os.path.join(sourcedatadir, 'Development', 'meta.csv')
    CSV = pd.read_csv(datasetinfordir)
    savedict = {}
    for i in range(len(CSV.values)):
        datas = CSV.values[i, 0].split('\t')
        id = datas[0].split('/')[1]
        id = id.split('.')[0]
        savedict.update({id: {}})
        savedict[id].update({'audiodir': os.path.join(sourcedatadir, 'Development', datas[0])})
        savedict[id].update({'videodir': os.path.join(sourcedatadir, 'Development', datas[1])})
        savedict[id].update({'label': datas[2]})

    splitdataset = {}
    for dset in ["train", "evaluate", "test"]:
        datasetinfordir = os.path.join(sourcedatadir, 'Development', 'evaluation_setup', 'fold1_' + dset + '.csv')
        CSV = pd.read_csv(datasetinfordir)
        identifierlist = []
        for i in range(len(CSV.values)):
            datas = CSV.values[i][0].split('\t')
            dataname = datas[0].split('/')[-1]
            dataname = dataname.split('.')[0]
            identifierlist.append(dataname)
        dsetdict = {}
        for id in identifierlist:
            dsetdict.update({id: savedict[id]})
        splitdataset.update({dset: dsetdict})
    #devlist = random.sample(list(splitdataset['train'].keys()), len(list(splitdataset['evaluate'].keys())))
    trainlines, devlines, testlines = [], [], []
    with open(os.path.join(datadir, "train.txt"), 'r') as file:
        # Read all lines from the file into a list
        for line in file:
            trainlines.append(line.strip('\n'))
    with open(os.path.join(datadir, "dev.txt"), 'r') as file:
        # Read all lines from the file into a list
        for line in file:
            devlines.append(line.strip('\n'))
    devdict = {}
    traindict = {}
    for i in list(splitdataset['train'].keys()):
        if i in devlines:
            devdict.update({i: splitdataset['train'][i]})
        elif i in trainlines:
            traindict.update({i: splitdataset['train'][i]})
    with open(os.path.join(datadir, "train.json"), 'w', encoding='utf-8') as f:
        json.dump(traindict, f, ensure_ascii=False, indent=4)
    with open(os.path.join(datadir, "dev.json"), 'w', encoding='utf-8') as f:
        json.dump(devdict, f, ensure_ascii=False, indent=4)
    with open(os.path.join(datadir, "test.json"), 'w', encoding='utf-8') as f:
        json.dump(splitdataset['evaluate'], f, ensure_ascii=False, indent=4)




