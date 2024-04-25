import os
import torch
from tqdm import tqdm
import json
import random
import kaldiio
import numpy as np
from copy import deepcopy
import argparse
from matplotlib import pyplot as plt


def acc(pred, label):
    return (np.sum((pred.data.numpy() == label.data.numpy()), axis=-1) / pred.size())[0]

logsoft = torch.nn.LogSoftmax(dim=-1)

def find_best(videodict):
    Keymax = max(zip(videodict.values(), videodict.keys()))[1]
    return Keymax


def str2id(label):
    if label == 'airport':
        label = 0
    elif label == 'shopping_mall':
        label = 1
    elif label == 'metro_station':
        label = 2
    elif label == 'street_pedestrian':
        label = 3
    elif label == 'public_square':
        label = 4
    elif label == 'street_traffic':
        label = 5
    elif label == 'tram':
        label = 6
    elif label == 'bus':
        label = 7
    elif label == 'metro':
        label = 8
    elif label == 'park':
        label = 9
    return label
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
            # To avoid disk videoess, create loader only for the first time
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


def optimize_alpha(trainaudiodata, trainvideodata, trainlabel, N, i, j, priorlist, savedir, evaltype, ifplot):
    # split pair i and j samples based on the ground truth labels
    audio_i = trainaudiodata[trainlabel == i]#[:, i]
    audio_j = trainaudiodata[trainlabel == j]#[:, j]
    video_i = trainvideodata[trainlabel == i]#[:, i]
    video_j = trainvideodata[trainlabel == j]#[:, j]

    betaaudioi = audio_i[:, i] - audio_i[:, j]
    betavideoi = video_i[:, i] - video_i[:, j]
    #betai = torch.stack([betaaudioi, betavideoi], dim=1)
    betai = torch.stack([betavideoi, betaaudioi], dim=1)

    betaaudioj = audio_j[:, i] - audio_j[:, j]
    betavideoj = video_j[:, i] - video_j[:, j]
    #betaj = torch.stack([betaaudioj, betavideoj], dim=1)
    betaj = torch.stack([betavideoj, betaaudioj], dim=1)

    # bnecause in this function, we only consider pair i and j, so it is similar with the binary classification task
    betai_mean = np.mean(betai.numpy(), axis=0) #torch.mean(betai, dim=0)
    betai_cov = np.cov(betai, rowvar=False)
    #betai_cov = torch.cov(torch.transpose(betai, 0, 1))

    betaj_mean = np.mean(betaj.numpy(), axis=0) #torch.mean(betai, dim=0)
    betaj_cov = np.cov(betaj, rowvar=False)
    #betaj_mean = torch.mean(betaj, dim=0)
    #betaj_cov = torch.cov(torch.transpose(betaj, 0, 1))

    trainsetvideo = {}
    alpha_vector = np.linspace(0.0, 1.0, 100)
    alpha = torch.FloatTensor(alpha_vector)
    Aij = ((0.5 + 0.5 * alpha_erf(alpha, betai_mean, betai_cov)) * priorlist[i] +
               (0.5 - 0.5 * alpha_erf(alpha, betaj_mean, betaj_cov)) * priorlist[j])
    #maxvalue = torch.max(Aij)
    #maxidx = (Aij == maxvalue).nonzero(as_tuple=False).squeeze(-1)#.tolist()
    #kopt = maxidx[int(len(maxidx) / 2)]
    kopt = torch.argmax(Aij)
    alpha_opt = alpha_vector[kopt]
    if ('Delta_representation' in evaltype) and (ifplot is True):
        fontsize = 15
        # if not os.path.exists(os.path.join(savedir, dset + 'beta_verteilung.pdf')):
        plt.figure()
        plt.ylabel(r'$\beta^a$', labelpad=0.1, rotation=0, fontsize=fontsize)
        plt.xlabel(r'$\beta^v$', fontsize=fontsize)
        # plt.title(dset + 'beta_verteilung')
        plt.scatter(betai[:, 0], betai[:, 1], color='red', alpha=0.1)
        plt.scatter(betaj[:, 0], betaj[:, 1], color='blue', alpha=0.1)
        plt.axline((0, 0), slope=alpha_opt / (alpha_opt - 1.0), color='green', label='by slope')
        plt.axhline(0, color='black', linestyle='--')
        plt.axvline(0, color='black', linestyle='--')
        plt.yticks(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.savefig(os.path.join(savedir, 'class_' + str(i) + '_class_' + str(j) + '_beta_verteilung.pdf'), bbox_inches='tight')
    else:
        pass

    return alpha_opt




def Pairwise_grid_search(devdict, testdict, savedir, evaltype, N, priorlist, ifplot):
    #for i in list(devdict.keys()):
    #    traindict.update({i + '_dev': devdict[i]})
    #traindict = devdict

    # alpha is a N * N matrix, where N is the number of classes
    traindict = devdict
    alpha = np.zeros((N, N))
    trainaudiodata = []
    trainvideodata = []
    trainlabel = []
    for uttid in list(traindict.keys()):
        trainaudiodata.append(torch.FloatTensor(traindict[uttid]['audioprob'][0]).unsqueeze(0))
        trainvideodata.append(torch.FloatTensor(traindict[uttid]['videoprob'][0]).unsqueeze(0))
        trainlabel.append(torch.LongTensor(traindict[uttid]['label'][0]))
    trainaudiodata = torch.concatenate(trainaudiodata, dim=0)
    trainvideodata = torch.concatenate(trainvideodata, dim=0)
    trainlabel = torch.concatenate(trainlabel, dim=0)
    for i in range(N):
        for j in range(N):
            if i == j:
                alpha_opt = 0.5
            else:
                alpha_opt = optimize_alpha(trainaudiodata, trainvideodata, trainlabel, N, i, j, priorlist, savedir, evaltype, ifplot)
            alpha[i][j] = alpha_opt
    print(alpha)
    if 'Delta_representation' in evaltype:
        videopred = []
        audiopred = []
        multipred = []
        labels = []
        output = {}
        for i, uttid in enumerate(tqdm(list(testdict.keys()))):
            DM = torch.zeros((N, N))
            for i in range(N):
                for j in range(N):
                    alpha_ij = alpha[i][j]
                    logitmix = alpha_ij * torch.FloatTensor(testdict[uttid]['audioprob'][0]) + (
                                1 - alpha_ij) * torch.FloatTensor(testdict[uttid]['videoprob'][0])
                    p = torch.softmax(logitmix, dim=-1)
                    DM[i][j] = p[i] - p[j]
            pp = (1 + torch.sum(DM, dim=1)) / N

            videopred.append(torch.argmax(torch.FloatTensor(testdict[uttid]['videoprob'][0]), dim=-1))
            audiopred.append(torch.argmax(torch.FloatTensor(testdict[uttid]['audioprob'][0]), dim=-1))
            multipred.append(torch.argmax(pp, dim=-1))

            labels.append(torch.LongTensor(testdict[uttid]['label'][0]))
            output.update({uttid: {}})
            output[uttid].update({'prob': pp.tolist()})
            output[uttid].update({'predict': int(torch.argmax(pp, dim=-1).data.numpy())})
            output[uttid].update({'label': int(testdict[uttid]['label'][0])})

        videopred = torch.LongTensor(videopred)
        audiopred = torch.LongTensor(audiopred)
        multipred = torch.LongTensor(multipred)
        labels = torch.LongTensor(labels)
        videoscore = acc(videopred, labels)
        audioscore = acc(audiopred, labels)
        multiscore = acc(multipred, labels)

        with open(os.path.join(savedir, 'videoacc_' + str(videoscore) + '_audioacc' + str(
                audioscore) + '_multiacc' + str(multiscore) + ".json"), 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)
    else:
        N_samples = len(testdict.keys())
        videopreds = []
        audiopreds = []
        multipred = []
        multiforcepred = []
        labels = []
        output = {}
        for i, uttid in enumerate(tqdm(list(testdict.keys()))):
            audiopred = torch.argmax(torch.softmax(torch.FloatTensor(testdict[uttid]['audioprob'][0]), dim=-1), dim=-1)
            videopred = torch.argmax(torch.softmax(torch.FloatTensor(testdict[uttid]['videoprob'][0]), dim=-1), dim=-1)

            if audiopred == videopred:
                results = audiopred
                results_force = audiopred

            else:
                i = audiopred
                j = videopred
                alpha_ij = alpha[i][j]
                x = alpha_ij * torch.FloatTensor(testdict[uttid]['audioprob'][0]) + (
                                1 - alpha_ij) * torch.FloatTensor(testdict[uttid]['videoprob'][0])
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
            videopreds.append(videopred)
            audiopreds.append(audiopred)

            labels.append(torch.LongTensor(testdict[uttid]['label'][0]))
            output.update({uttid: {}})
            output[uttid].update({'predict': int(results)})
            output[uttid].update({'predict_force': int(results_force)})
            output[uttid].update({'label': int(testdict[uttid]['label'][0])})

        videopreds = torch.LongTensor(videopreds)
        audiopreds = torch.LongTensor(audiopreds)
        multipred = torch.LongTensor(multipred)
        multiforcepred = torch.LongTensor(multiforcepred)
        labels = torch.LongTensor(labels)
        videoscore = acc(videopreds, labels)
        audioscore = acc(audiopreds, labels)
        multiscore = acc(multipred, labels)
        multiforcescore = acc(multiforcepred, labels)

        with open(os.path.join(savedir, 'videoacc_' + str(videoscore) + '_audioacc' + str(
                audioscore) + '_multiacc' + str(multiscore) + '_multiforceacc' + str(multiforcescore) + ".json"), 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)





def get_args():
    parser = argparse.ArgumentParser()

    # get arguments from outside
    parser.add_argument('--datasdir', default='./../../../Dataset', type=str, help='Dir saves the datasource information')
    parser.add_argument('--evaltype', default='Delta_representation', type=str, help='which data stream')
    parser.add_argument('--videomodal', default='video', type=str, help='which data stream')
    parser.add_argument('--audiomodal', default='audio', type=str, help='single or multi images')
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
    videomodal = args.videomodal
    audiomodal = args.audiomodal
    savedir = args.savedir
    ifplot = args.ifplot
    if ifplot == 'true':
        ifplot = True
    else:
        ifplot = False

    savedir = os.path.join(savedir, 'Subset', evaltype)
    N = 10  # the number of classes

    cvfolder = os.path.join(datasdir, 'subsets')
    if not os.path.exists(cvfolder):
        os.makedirs(cvfolder)
    numdir = os.listdir(cvfolder)
    if len(numdir) == 0:
        #with open(os.path.join(datasdir, "train.json"), encoding="utf8") as json_file:
        #    traindict_org = json.load(json_file)
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
                        if M != 256:
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
                        # savedict[filename].update({'videofeat': videofeat[filename]})
                        savedict[filename].update({'audioprob': _get_from_loader(
                            filepath=audioprob[filename],
                            filetype='mat')})  # .replace('./', './../../../'), filetype='mat')})
                        savedict[filename].update({'videoprob': _get_from_loader(
                            filepath=videoprob[filename],
                            filetype='mat')})  # .replace('./', './../../../'), filetype='mat')})
                        # savedict[filename].update({'audiofeat': audiofeat[filename]})
                        labeldata = _get_from_loader(filepath=label[filename],
                                                     filetype='mat')  # .replace('./', './../../../'), filetype='mat')

                        '''savedict[filename].update({'audioprob': _get_from_loader(
                            filepath=audioprob[filename].replace('./', './../../../'), filetype='mat')})
                        savedict[filename].update({'videoprob': _get_from_loader(
                            filepath=videoprob[filename].replace('./', './../../../'), filetype='mat')})
                        # savedict[filename].update({'audiofeat': audiofeat[filename]})
                        labeldata = _get_from_loader(filepath=label[filename].replace('./', './../../../'),
                                                     filetype='mat')'''

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
                # elif dset == 'train':
                #    traindict = savedict

            priorlist = [len(prior[i]) / len(devdict.keys()) for i in range(N)]

            # print(priorlist)
            Pairwise_grid_search(devdict, testdict, cvsavedir, evaltype, N, priorlist, ifplot)













