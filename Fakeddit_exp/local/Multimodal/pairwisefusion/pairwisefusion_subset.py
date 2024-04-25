import os
import json
import torch
from tqdm import tqdm
import json
import random
import kaldiio
import numpy as np
from copy import deepcopy
import argparse
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
from torchmetrics.classification import MulticlassF1Score

f1score = MulticlassF1Score(num_classes=7, average='weighted')
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

def alpha_erf(alpha, mean, cov):
    mu1 = alpha * mean[0]
    mu2 = (1 - alpha) * mean[1]

    v1 = (alpha * alpha) * cov[0, 0]
    v2 = (1 - alpha) * (1 - alpha) * cov[1, 1]
    term = alpha * (1 - alpha) * (cov[0, 1] + cov[1, 0])

    num = mu1 + mu2
    den = v1 + v2 + term
    den = torch.sqrt(2 * den)
    return torch.erf(num / den)


def optimize_alpha(trainimagedata, traintextdata, trainlabel, N, i, j, priorlist, savedir, evaltype, ifplot):
    # split pair i and j samples based on the ground truth labels
    image_i = trainimagedata[trainlabel == i]#[:, i]
    image_j = trainimagedata[trainlabel == j]#[:, j]
    text_i = traintextdata[trainlabel == i]#[:, i]
    text_j = traintextdata[trainlabel == j]#[:, j]

    betaimagei = image_i[:, i] - image_i[:, j]
    betatexti = text_i[:, i] - text_i[:, j]
    betai = torch.stack([betaimagei, betatexti], dim=1)

    betaimagej = image_j[:, i] - image_j[:, j]
    betatextj = text_j[:, i] - text_j[:, j]
    betaj = torch.stack([betaimagej, betatextj], dim=1)

    # bnecause in this function, we only consider pair i and j, so it is similar with the binary classification task
    betai_mean = np.mean(betai.numpy(), axis=0) #torch.mean(betai, dim=0)
    betai_cov = np.cov(betai, rowvar=False)
    #betai_cov = torch.cov(torch.transpose(betai, 0, 1))

    betaj_mean = np.mean(betaj.numpy(), axis=0) #torch.mean(betai, dim=0)
    betaj_cov = np.cov(betaj, rowvar=False)
    #betaj_mean = torch.mean(betaj, dim=0)
    #betaj_cov = torch.cov(torch.transpose(betaj, 0, 1))

    trainsetacc = {}
    alpha_vector = np.linspace(0.0, 1.0, 100)
    alpha = torch.FloatTensor(alpha_vector)
    Aij = ((0.5 + 0.5 * alpha_erf(alpha, betai_mean, betai_cov)) * priorlist[i] +
               (0.5 - 0.5 * alpha_erf(alpha, betaj_mean, betaj_cov)) * priorlist[j])
    kopt = torch.argmax(Aij)
    alpha_opt = alpha_vector[kopt]
    if ('Delta_representation' in evaltype) and (ifplot is True):
        fontsize = 10
        # if not os.path.exists(os.path.join(savedir, dset + 'beta_verteilung.pdf')):
        plt.figure()
        plt.ylabel(r'$\beta$image', labelpad=0.1, rotation=0, fontsize=fontsize)
        plt.xlabel(r'$\beta$text', fontsize=fontsize)
        # plt.title(dset + 'beta_verteilung')
        plt.scatter(betai[:, 0], betai[:, 1], color='red', alpha=0.1)
        plt.scatter(betaj[:, 0], betaj[:, 1], color='blue', alpha=0.1)
        plt.axline((0, 0), slope=alpha_opt / (alpha_opt - 1.0), color='green', label='by slope')
        plt.axhline(0, color='black', linestyle='--')
        plt.axvline(0, color='black', linestyle='--')
        plt.yticks(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.savefig(os.path.join(savedir, 'class_' + str(i) + '_class_' + str(j) + '_beta_verteilung.pdf'))
    else:
        pass

    return alpha_opt




def Pairwise_grid_search(devdict, testdict, savedir, evaltype, N, priorlist, ifplot):
    # alpha is a N * N matrix, where N is the number of classes
    traindict = devdict
    alpha = np.zeros((N, N))
    trainimagedata = []
    traintextdata = []
    trainlabel = []
    for uttid in list(traindict.keys()):
        trainimagedata.append(torch.FloatTensor(traindict[uttid]['imageprob'][0]).unsqueeze(0))
        traintextdata.append(torch.FloatTensor(traindict[uttid]['textprob'][0]).unsqueeze(0))
        trainlabel.append(torch.LongTensor(traindict[uttid]['label'][0]))
    trainimagedata = torch.concatenate(trainimagedata, dim=0)
    traintextdata = torch.concatenate(traintextdata, dim=0)
    trainlabel = torch.concatenate(trainlabel, dim=0)
    for i in range(N):
        for j in range(N):
            if i == j:
                alpha_opt = 0.5
            else:
                alpha_opt = optimize_alpha(trainimagedata, traintextdata, trainlabel, N, i, j, priorlist, savedir, evaltype, ifplot)
            alpha[i][j] = alpha_opt
    print(alpha)
    if 'Delta_representation' in evaltype:
        textpred = []
        imagepred = []
        multipred = []
        labels = []
        output = {}
        for i, uttid in enumerate(tqdm(list(testdict.keys()))):
            DM = torch.zeros((N, N))
            for i in range(N):
                for j in range(N):
                    alpha_ij = alpha[i][j]
                    logitmix = alpha_ij * torch.FloatTensor(testdict[uttid]['imageprob'][0]) + (
                                1 - alpha_ij) * torch.FloatTensor(testdict[uttid]['textprob'][0])
                    p = torch.softmax(logitmix, dim=-1)
                    DM[i][j] = p[i] - p[j]
            pp = (1 + torch.sum(DM, dim=1)) / N

            textpred.append(torch.argmax(torch.FloatTensor(testdict[uttid]['textprob'][0]), dim=-1))
            imagepred.append(torch.argmax(torch.FloatTensor(testdict[uttid]['imageprob'][0]), dim=-1))
            multipred.append(torch.argmax(pp, dim=-1))

            labels.append(torch.LongTensor(testdict[uttid]['label'][0]))
            output.update({uttid: {}})
            output[uttid].update({'prob': pp.tolist()})
            output[uttid].update({'predict': int(torch.argmax(pp, dim=-1).data.numpy())})
            output[uttid].update({'label': int(testdict[uttid]['label'][0])})

        textpred = torch.LongTensor(textpred)
        imagepred = torch.LongTensor(imagepred)
        multipred = torch.LongTensor(multipred)
        labels = torch.LongTensor(labels)

        textscore = (sum(textpred == labels) / labels.size(0)).cpu().data.numpy()
        imagescore = (sum(imagepred == labels) / labels.size(0)).cpu().data.numpy()
        multiscore = (sum(multipred == labels) / labels.size(0)).cpu().data.numpy()

        with open(os.path.join(savedir, 'textaccscore_' + str(textscore) + '_imageaccscore' + str(
                imagescore) + '_multiaccscore' + str(multiscore) + ".json"), 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)
    else:
        N_samples = len(testdict.keys())
        textpreds = []
        imagepreds = []
        multipred = []
        multiforcepred = []
        labels = []
        output = {}
        for i, uttid in enumerate(tqdm(list(testdict.keys()))):
            imagepred = torch.argmax(torch.softmax(torch.FloatTensor(testdict[uttid]['imageprob'][0]), dim=-1), dim=-1)
            textpred = torch.argmax(torch.softmax(torch.FloatTensor(testdict[uttid]['textprob'][0]), dim=-1), dim=-1)

            if imagepred == textpred:
                results = imagepred
                results_force = imagepred

            else:
                i = imagepred
                j = textpred
                alpha_ij = alpha[i][j]
                x = alpha_ij * torch.FloatTensor(testdict[uttid]['imageprob'][0]) + (
                                1 - alpha_ij) * torch.FloatTensor(testdict[uttid]['textprob'][0])
                p = torch.softmax(x, dim=-1)
                results = torch.argmax(p, dim=-1)


                if results != i or results !=j:
                    if p[i] > p[j]:
                        results_force = i
                    else:
                        results_force = j
                else:
                    results_force = results

            multipred.append(results)
            multiforcepred.append(results_force)
            textpreds.append(textpred)
            imagepreds.append(imagepred)

            labels.append(torch.LongTensor(testdict[uttid]['label'][0]))
            output.update({uttid: {}})
            output[uttid].update({'predict': int(results)})
            output[uttid].update({'predict_force': int(results_force)})
            output[uttid].update({'label': int(testdict[uttid]['label'][0])})

        textpreds = torch.LongTensor(textpreds)
        imagepreds = torch.LongTensor(imagepreds)
        multipred = torch.LongTensor(multipred)
        multiforcepred = torch.LongTensor(multiforcepred)
        labels = torch.LongTensor(labels)
        textscore = (sum(textpreds == labels) / labels.size(0)).cpu().data.numpy()
        imagescore = (sum(imagepreds == labels) / labels.size(0)).cpu().data.numpy()
        multiscore = (sum(multipred == labels) / labels.size(0)).cpu().data.numpy()
        multiforcescore = (sum(multiforcepred == labels) / labels.size(0)).cpu().data.numpy()

        with open(os.path.join(savedir, 'textaccscore_' + str(textscore) + '_imageaccscore' + str(
                imagescore) + '_multiaccscore' + str(multiscore) + '_multiforceaccscore' + str(multiforcescore) + ".json"), 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)





def get_args():
    parser = argparse.ArgumentParser()

    # get arguments from outside
    parser.add_argument('--datasdir', default='./../../../Dataset', type=str, help='Dir saves the datasource information')
    parser.add_argument('--evaltype', default='Pairwise', type=str, help='Evaluation type, possible are Delta_representation and Pairwise')
    parser.add_argument('--textmodal', default='Title', type=str, help='which data stream')
    parser.add_argument('--imagemodal', default='Image', type=str, help='single or multi images')
    parser.add_argument('--savedir', default='./../../../trained', type=str, help='Dir to save trained model and results')
    parser.add_argument('--ifplot', default='false', type=str, help='if plot data distributation')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    print("Arguments:")
    [print(key, val) for key, val in vars(args).items()]

    datasdir = args.datasdir
    evaltype = args.evaltype
    textmodal = args.textmodal
    imagemodal = args.imagemodal
    savedir = args.savedir
    ifplot = args.ifplot
    if ifplot == 'true':
        ifplot = True
    else:
        ifplot = False

    savedir = os.path.join(savedir, 'Subset', evaltype)
    N = 6  # the number of classes

    cvfolder = os.path.join(datasdir, 'cvsplit')
    if not os.path.exists(cvfolder):
        os.makedirs(cvfolder)
    numdir = os.listdir(cvfolder)
    if len(numdir) == 0:
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
                        # savedict[filename].update({'textfeat': textfeat[filename]})
                        savedict[filename].update({'imageprob': _get_from_loader(
                            filepath=imageprob[filename],
                            filetype='mat')})  
                        savedict[filename].update({'textprob': _get_from_loader(
                            filepath=textprob[filename],
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

            priorlist = [len(prior[i]) / len(devdict.keys()) for i in range(N)]

            Pairwise_grid_search(devdict, testdict, cvsavedir, evaltype, N, priorlist, ifplot)













