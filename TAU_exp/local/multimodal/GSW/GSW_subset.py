import os, sys
import json
import math
import torch
from tqdm import tqdm
import json
import kaldiio
import numpy as np
import argparse
import random
from copy import deepcopy
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture

def acc(pred, label):
    return (np.sum((pred.data.numpy() == label.data.numpy()), axis=-1) / pred.size())[0]
logsoft = torch.nn.LogSoftmax(dim=-1)

def find_best(accdict):
    Keymax = max(zip(accdict.values(), accdict.keys()))[1]
    return Keymax
def _get_from_loader(filepath, filetype):
    """Return ndarray

    In order to make the fds to be opened only at the first referring,
    the loader are stored in self._loaders

    >>> ndarray = loader.get_from_loader(
    ...     'some/path.h5:F01_050C0101_PED_REAL', filetype='hdf5')

    :param: str filepath:
    :param: str filetype:
    :return:
    :rtype: np.ndarray
    """
    if filetype in ['mat', 'vec']:
        # e.g.
        #    {"input": [{"feat": "some/path.ark:123",
        #                "filetype": "mat"}]},
        # In this case, "123" indicates the starting points of the matrix
        # load_mat can load both matrix and vector
        #filepath = filepath.replace('/home/wentao', '.')
        return kaldiio.load_mat(filepath)
    elif filetype == 'scp':
        # e.g.
        #    {"input": [{"feat": "some/path.scp:F01_050C0101_PED_REAL",
        #                "filetype": "scp",
        filepath, key = filepath.split(':', 1)
        loader = self._loaders.get(filepath)
        if loader is None:
            # To avoid disk access, create loader only for the first time
            loader = kaldiio.load_scp(filepath)
            self._loaders[filepath] = loader
        return loader[key]
    else:
        raise NotImplementedError(
            'Not supported: loader_type={}'.format(filetype))
def Global_grid_search(devdict, testdict, savedir, modal):
    traindict = devdict
    alpha_vector = np.linspace(0.0, 1.0, 100)
    trainsetacc = {}
    for i, alpha in enumerate(tqdm(alpha_vector)):
        multipred = []
        labels = []
        for uttid in list(traindict.keys()):
            if 'linear' in modal:
                audioprob = torch.softmax(torch.FloatTensor(traindict[uttid]['audioprob'][0]), dim=0)
                videoprob = torch.softmax(torch.FloatTensor(traindict[uttid]['videoprob'][0]), dim=0)
            elif 'logarithmic' in modal:
                audioprob = logsoft(torch.FloatTensor(traindict[uttid]['audioprob'][0]))
                videoprob = logsoft(torch.FloatTensor(traindict[uttid]['videoprob'][0]))
            elif 'logit' in modal:
                audioprob = torch.FloatTensor(traindict[uttid]['audioprob'][0])
                videoprob = torch.FloatTensor(traindict[uttid]['videoprob'][0])
            multi = alpha * audioprob + (1 - alpha) * videoprob
            pred = torch.argmax(multi, dim=-1)
            multipred.append(pred)
            labels.append(torch.LongTensor(traindict[uttid]['label'][0]))
        multipred = torch.LongTensor(multipred)
        labels = torch.LongTensor(labels)
        allscore = acc(multipred, labels)
        trainsetacc.update({alpha: allscore})
    opt_alpha = find_best(trainsetacc)

    #opt_alpha = 0.31313131
    videopred = []
    audiopred = []
    multipred = []
    labels = []
    output = {}
    for i, uttid in enumerate(tqdm(list(testdict.keys()))):
        if 'linear' in modal:
            audioprob = torch.softmax(torch.FloatTensor(testdict[uttid]['audioprob'][0]), dim=0)
            videoprob = torch.softmax(torch.FloatTensor(testdict[uttid]['videoprob'][0]), dim=0)
        elif 'logarithmic' in modal:
            audioprob = logsoft(torch.FloatTensor(testdict[uttid]['audioprob'][0]))
            videoprob = logsoft(torch.FloatTensor(testdict[uttid]['videoprob'][0]))
        elif 'logit' in modal:
            audioprob = torch.FloatTensor(testdict[uttid]['audioprob'][0])
            videoprob = torch.FloatTensor(testdict[uttid]['videoprob'][0])

        videopred.append(torch.argmax(videoprob, dim=-1))
        audiopred.append(torch.argmax(audioprob, dim=-1))
        multi = opt_alpha * audioprob + (1 - opt_alpha) * videoprob
        pred = torch.argmax(multi, dim=-1)
        multipred.append(pred)
        labels.append(torch.LongTensor(testdict[uttid]['label'][0]))
        output.update({uttid: {}})
        output[uttid].update({'prob': multi.tolist()})
        output[uttid].update({'predict': int(pred.data.numpy())})
        output[uttid].update({'label': int(testdict[uttid]['label'][0])})

    videopred = torch.LongTensor(videopred)
    audiopred = torch.LongTensor(audiopred)
    multipred = torch.LongTensor(multipred)
    labels = torch.LongTensor(labels)
    videoscore = acc(videopred, labels)
    audioscore = acc(audiopred, labels)
    multiscore = acc(multipred, labels)


    with open(os.path.join(savedir, 'videoacc_' + str(videoscore) + '_audiopacc_' + str(
        audioscore) + '_multiacc_' + str(multiscore) + ".json"), 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=4)



def get_args():
    parser = argparse.ArgumentParser()

    # get arguments from outside
    parser.add_argument('--datasdir', default='./../../../Dataset', type=str, help='Dir saves the datasource information')
    parser.add_argument('--modal', default='GSW_linear', type=str, help='which data stream')
    parser.add_argument('--videomodal', default='video', type=str, help='which data stream')
    parser.add_argument('--audiomodal', default='audio', type=str, help='single or multi images')
    parser.add_argument('--savedir', default='./../../../trained/Global_fixed', type=str, help='Dir to save trained model and results')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    print("Arguments:")
    [print(key, val) for key, val in vars(args).items()]

    datasdir = args.datasdir
    modal = args.modal
    videomodal = args.videomodal
    audiomodal = args.audiomodal
    savedir = args.savedir
    savedir = os.path.join(savedir, 'Subset', 'Global_fixed', modal)
    #if not os.path.exists(savedir):
    #    os.makedirs(savedir)
    N = 6  # the number of classes

    cvfolder = os.path.join(datasdir, 'subsets')
    if not os.path.exists(cvfolder):
        os.makedirs(cvfolder)
    numdir = os.listdir(cvfolder)
    if len(numdir) == 0:

        with open(os.path.join(datasdir, "train.json"), encoding="utf8") as json_file:
            traindict = json.load(json_file)
        with open(os.path.join(datasdir, "dev.json"), encoding="utf8") as json_file:
            traindict_org = json.load(json_file)
        with open(os.path.join(datasdir, "test.json"), encoding="utf8") as json_file:
            testdict = json.load(json_file)

        labeltraindict_org = {}
        for n in range(N):
            labeltraindict_org.update({n: {}})
        for keys in list(traindict_org.keys()):
            label = traindict_org[keys]['label']
            label = str2id(label)
            labeltraindict_org[label].update({keys: traindict_org[keys]})

        traindict = deepcopy(traindict_org)
        labeltraindict = deepcopy(labeltraindict_org)
        for M in [24, 32, 64, 128, 256]:  # [4, 16, 64, 256, 1024, 2048]:
            
            cvsubfolder = os.path.join(cvfolder, str(M))
            if not os.path.exists(cvsubfolder):
                os.makedirs(cvsubfolder)
                i = 0
                while i < 10:  # substruct 10 subsets
                    trainsubdict_0 = {}
                    for n in range(N):
                        # print(len(list(labeltraindict[n].keys())))
                        selectkeys = random.sample(list(labeltraindict[n].keys()), 2)
                        for selectkey in selectkeys:
                            trainsubdict_0.update({selectkey: labeltraindict[n][selectkey]})
                        # del labeltraindict[n][selectkey[0]]
                    M_new = M - 2 * N
                    trainsubdict = {k: traindict[k] for k in random.sample(list(traindict.keys()), M_new)}
                    trainsubdict.update(trainsubdict_0)
                    filelist = list(trainsubdict.keys())
                    with open(os.path.join(cvsubfolder, str(i) + ".txt"), 'w') as file:
                        for item in filelist:
                            file.write(str(item) + '\n')  # Convert item to string if necessary
                    i = i + 1
                    for subkeys in list(trainsubdict.keys()):
                        label = traindict[subkeys]['label']
                        label = str2id(label)
                        if M == 24 or M == 32:
                            del labeltraindict[label][subkeys]
                            del traindict[subkeys]
                        else:
                            pass
            else:
                pass
    else:
        pass


    for M in [24, 32, 64, 128, 256]:
        cvsubfolder = os.path.join(cvfolder, str(M))
        for id in range(10):
            cvsavedir = os.path.join(savedir, str(M), str(id))
            if not os.path.exists(cvsavedir):
                os.makedirs(cvsavedir)

            with open(os.path.join(datasdir, "subsets", str(M), str(id) + ".txt"), 'r') as file:
                # Read all lines from the file into a list
                lines = file.readlines()
            with open(os.path.join(datasdir, "dev.json"), encoding="utf8") as json_file:
                devorgdict = json.load(json_file)
            with open(os.path.join(datasdir, "test.json"), encoding="utf8") as json_file:
                testdict = json.load(json_file)
            devdict = {}
            for i in lines:
                i = i.strip('\n')
                devdict.update({i: devorgdict[i]})

            prior = {}
            for i in range(N):
                prior.update({i: []})
            for workdict, dset in zip([devdict, testdict], ["dev", "test"]):
                videofeatscpdir = os.path.join(args.datasdir, 'data', videomodal, dset + 'feats.scp')
                videoprobscpdir = os.path.join(args.datasdir, 'data', videomodal, dset + 'prob.scp')
                audiofeatscpdir = os.path.join(args.datasdir, 'data', audiomodal, dset + 'feats.scp')
                audioprobscpdir = os.path.join(args.datasdir, 'data', audiomodal, dset + 'prob.scp')
                labelscpdir = os.path.join(args.datasdir, 'data', audiomodal, dset + 'label.scp')

                videofeat, videoprob, audiofeat, audioprob, label = {}, {}, {}, {}, {}
                for scpfiles in [videofeatscpdir, videoprobscpdir, audiofeatscpdir, audioprobscpdir, labelscpdir]:
                    with open(scpfiles) as f:
                        srcdata = f.readlines()
                    datadict = {}
                    for j in srcdata:
                        fullname = j.split(' ')[0]
                        # subnames = fullname.split('_')
                        datadir = j.split(' ')[1].strip('\n')
                        datadict.update({fullname: datadir})
                    if scpfiles == videofeatscpdir:
                        videofeat = datadict
                    elif scpfiles == videoprobscpdir:
                        videoprob = datadict
                    elif scpfiles == audiofeatscpdir:
                        audiofeat = datadict
                    elif scpfiles == audioprobscpdir:
                        audioprob = datadict
                    elif scpfiles == labelscpdir:
                        label = datadict

                savedict = {}
                for filename in list(workdict.keys()):
                    try:
                        savedict.update({filename: {}})
                        savedict[filename].update({'audioprob': _get_from_loader(
                            filepath=audioprob[filename],
                            filetype='mat')})
                        savedict[filename].update({'videoprob': _get_from_loader(
                            filepath=videoprob[filename],
                            filetype='mat')})
                        labeldata = _get_from_loader(filepath=label[filename],
                                                     filetype='mat')

                        if dset == 'dev':
                            prior[labeldata[0][0]].append(filename)
                        savedict[filename].update(
                            {'label': labeldata})


                    except:
                        del savedict[filename]

                if dset == 'dev':
                    devdict = savedict
                elif dset == 'test':
                    testdict = savedict

            Global_grid_search(devdict, testdict, cvsavedir, modal)









