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
from torchmetrics.classification import MulticlassF1Score

f1score = MulticlassF1Score(num_classes=6, average='weighted')
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
                imageprob = torch.softmax(torch.FloatTensor(traindict[uttid]['imageprob'][0]), dim=0)
                textprob = torch.softmax(torch.FloatTensor(traindict[uttid]['textprob'][0]), dim=0)
            elif 'logarithmic' in modal:
                imageprob = logsoft(torch.FloatTensor(traindict[uttid]['imageprob'][0]))
                textprob = logsoft(torch.FloatTensor(traindict[uttid]['textprob'][0]))
            elif 'logit' in modal:
                imageprob = torch.FloatTensor(traindict[uttid]['imageprob'][0])
                textprob = torch.FloatTensor(traindict[uttid]['textprob'][0])
            multi = alpha * imageprob + (1 - alpha) * textprob
            pred = torch.argmax(multi, dim=-1)
            multipred.append(pred)
            labels.append(torch.LongTensor(traindict[uttid]['label'][0]))
        multipred = torch.LongTensor(multipred)
        labels = torch.LongTensor(labels)
        allscore = f1score(multipred, labels).cpu().data.numpy()
        trainsetacc.update({alpha: allscore})
    opt_alpha = find_best(trainsetacc)

    #opt_alpha = 0.31313131
    textpred = []
    imagepred = []
    multipred = []
    labels = []
    output = {}
    for i, uttid in enumerate(tqdm(list(testdict.keys()))):
        if 'linear' in modal:
            imageprob = torch.softmax(torch.FloatTensor(testdict[uttid]['imageprob'][0]), dim=0)
            textprob = torch.softmax(torch.FloatTensor(testdict[uttid]['textprob'][0]), dim=0)
        elif 'logarithmic' in modal:
            imageprob = logsoft(torch.FloatTensor(testdict[uttid]['imageprob'][0]))
            textprob = logsoft(torch.FloatTensor(testdict[uttid]['textprob'][0]))
        elif 'logit' in modal:
            imageprob = torch.FloatTensor(testdict[uttid]['imageprob'][0])
            textprob = torch.FloatTensor(testdict[uttid]['textprob'][0])

        textpred.append(torch.argmax(textprob, dim=-1))
        imagepred.append(torch.argmax(imageprob, dim=-1))
        multi = opt_alpha * imageprob + (1 - opt_alpha) * textprob
        pred = torch.argmax(multi, dim=-1)
        multipred.append(pred)
        labels.append(torch.LongTensor(testdict[uttid]['label'][0]))
        output.update({uttid: {}})
        output[uttid].update({'prob': multi.tolist()})
        output[uttid].update({'predict': int(pred.data.numpy())})
        output[uttid].update({'label': int(testdict[uttid]['label'][0])})

    textpred = torch.LongTensor(textpred)
    imagepred = torch.LongTensor(imagepred)
    multipred = torch.LongTensor(multipred)
    labels = torch.LongTensor(labels)
    textscore = (sum(textpred == labels) / labels.size(0)).cpu().data.numpy()
    imagescore = (sum(imagepred == labels) / labels.size(0)).cpu().data.numpy()
    multiscore = (sum(multipred == labels) / labels.size(0)).cpu().data.numpy()


    with open(os.path.join(savedir, 'textf1score_' + str(textscore) + '_imagef1score' + str(
        imagescore) + '_multif1score' + str(multiscore) + ".json"), 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=4)



def get_args():
    parser = argparse.ArgumentParser()

    # get arguments from outside
    parser.add_argument('--datasdir', default='./../../../Dataset', type=str, help='Dir saves the meta information')
    parser.add_argument('--modal', default='GSW_linear', type=str, help='which data modality. It could be GSW_linear, GSW_logarithmic, GSW_logit')
    parser.add_argument('--textmodal', default='Title', type=str, help='Text stream')
    parser.add_argument('--imagemodal', default='Image', type=str, help='Image stream')
    parser.add_argument('--savedir', default='./../../../trained', type=str, help='Dir to save trained model and results')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    print("Arguments:")
    [print(key, val) for key, val in vars(args).items()]

    datasdir = args.datasdir
    modal = args.modal
    modal = modal
    textmodal = args.textmodal
    imagemodal = args.imagemodal
    savedir = args.savedir
    savedir = os.path.join(savedir, 'Subset', 'Global_fixed', modal)
    N = 6  # the number of classes

    cvfolder = os.path.join(datasdir, 'cvsplit')
    if not os.path.exists(cvfolder):
        os.makedirs(cvfolder)
    numdir = os.listdir(cvfolder)
    if len(numdir) == 0:
        # creating cross-validation sub-training set
        #with open(os.path.join(datasdir, "train.json"), encoding="utf8") as json_file:
        #    traindict_org = json.load(json_file)
        with open(os.path.join(datasdir, "val.json"), encoding="utf8") as json_file:
            devdict_org = json.load(json_file)
        with open(os.path.join(datasdir, "test.json"), encoding="utf8") as json_file:
            testdict = json.load(json_file)

        labeldevdict_org = {}
        for n in range(N):
            labeldevdict_org.update({n: {}})
        for keys in list(devdict_org.keys()):
            label = devdict_org[keys]['6_way_label']
            labeldevdict_org[label].update({keys: devdict_org[keys]})

        for M in [18, 32, 64, 128, 256, 512]:  # [4, 16, 64, 256, 1024, 2048]:
            devdict = deepcopy(devdict_org)
            labeldevdict = deepcopy(labeldevdict_org)
            cvsubfolder = os.path.join(cvfolder, str(M))
            if not os.path.exists(cvsubfolder):
                os.makedirs(cvsubfolder)
                i = 0
                while i < 10:  # substruct 10 subsets
                    devsubdict_0 = {}
                    for n in range(N):
                        # print(len(list(labeltraindict[n].keys())))
                        selectkeys = random.sample(list(labeldevdict[n].keys()), 3)
                        for selectkey in selectkeys:
                            devsubdict_0.update({selectkey: labeldevdict[n][selectkey]})
                        # del labeltraindict[n][selectkey[0]]
                    M_new = M - 3 * N
                    devsubdict = {k: devdict[k] for k in random.sample(list(devdict.keys()), M_new)}
                    devsubdict.update(devsubdict_0)
                    filelist = list(devsubdict.keys())
                    with open(os.path.join(cvsubfolder, str(i) + ".txt"), 'w') as file:
                        for item in filelist:
                            file.write(str(item) + '\n')  # Convert item to string if necessary
                    i = i + 1
                    for subkeys in list(devsubdict.keys()):
                        label = devdict[subkeys]['6_way_label']
                        del labeldevdict[label][subkeys]
                        del devdict[subkeys]
            else:
                pass
    else:
        pass

    #import random
    #traindict = {k: traindict[k] for k in list(random.sample(list(traindict.keys()), 20))}
    #devdict = {k: devdict[k] for k in list(random.sample(list(devdict.keys()), 20))}
    #testdict = {k: testdict[k] for k in list(random.sample(list(testdict.keys()), 20))}

    for M in [18, 32, 64, 128, 256, 512]:
        cvsubfolder = os.path.join(cvfolder, str(M))
        for id in range(10):
            cvsavedir = os.path.join(savedir, str(M), str(id))
            if not os.path.exists(cvsavedir):
                os.makedirs(cvsavedir)


            with open(os.path.join(datasdir, "cvsplit", str(M), str(id) + ".txt"), 'r') as file:
                # Read all lines from the file into a list
                lines = file.readlines()
            with open(os.path.join(datasdir, "val.json"), encoding="utf8") as json_file:
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
                textfeatscpdir = os.path.join(args.datasdir, 'data', textmodal, dset + 'feats.scp')
                textprobscpdir = os.path.join(args.datasdir, 'data', textmodal, dset + 'prob.scp')
                imagefeatscpdir = os.path.join(args.datasdir, 'data', imagemodal, dset + 'feats.scp')
                imageprobscpdir = os.path.join(args.datasdir, 'data', imagemodal, dset + 'prob.scp')
                labelscpdir = os.path.join(args.datasdir, 'data', imagemodal, dset + 'label.scp')

                textfeat, textprob, imagefeat, imageprob, label = {}, {}, {}, {}, {}
                for scpfiles in [textfeatscpdir, textprobscpdir, imagefeatscpdir, imageprobscpdir, labelscpdir]:
                    with open(scpfiles) as f:
                        srcdata = f.readlines()
                    datadict = {}
                    for j in srcdata:
                        fullname = j.split(' ')[0]
                        # subnames = fullname.split('_')
                        datadir = j.split(' ')[1].strip('\n')
                        datadict.update({fullname: datadir})
                    if scpfiles == textfeatscpdir:
                        textfeat = datadict
                    elif scpfiles == textprobscpdir:
                        textprob = datadict
                    elif scpfiles == imagefeatscpdir:
                        imagefeat = datadict
                    elif scpfiles == imageprobscpdir:
                        imageprob = datadict
                    elif scpfiles == labelscpdir:
                        label = datadict

                savedict = {}
                for filename in list(workdict.keys()):
                    try:
                        savedict.update({filename: {}})
                        savedict[filename].update({'imageprob': _get_from_loader(
                            filepath=imageprob[filename],
                            filetype='mat')})
                        savedict[filename].update({'textprob': _get_from_loader(
                            filepath=textprob[filename],
                            filetype='mat')})
                        labeldata = _get_from_loader(filepath=label[filename],
                                                     filetype='mat')

                        if dset == 'train':
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









