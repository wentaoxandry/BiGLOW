import os
import torch
from tqdm import tqdm
import json
import time
import kaldiio
import numpy as np
import argparse
import pingouin as pg
from matplotlib import pyplot as plt
import logging

# = MulticlassF1Score(num_classes=7, average='weighted')
logsoft = torch.nn.LogSoftmax(dim=-1)
def Henze_Zirkler_normality_test(data):
    """
    Args:
        data: the datato test if it is multivariant gaussian distribution

    Returns: p-value, if p-value > 0.05, it is gaussian distribution

    """
    data = data.numpy()
    mardia_test = pg.multivariate_normality(data)
    return mardia_test.pval, mardia_test.normal

def acc(pred, label):
    return (np.sum((pred.data.numpy() == label.data.numpy()), axis=-1) / pred.size())[0]

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


def optimize_alpha(trainaudiodata, trainvideodata, trainlabel, N, i, j, priorlist, savedir, evaltype):
    # split pair i and j samples based on the ground truth labels
    audio_i = trainaudiodata[trainlabel == i]#[:, i]
    audio_j = trainaudiodata[trainlabel == j]#[:, j]
    video_i = trainvideodata[trainlabel == i]#[:, i]
    video_j = trainvideodata[trainlabel == j]#[:, j]

    betaaudioi = (audio_i[:, i] - audio_i[:, j])
    betavideoi = (video_i[:, i] - video_i[:, j])
    betai = torch.stack([betavideoi, betaaudioi], dim=1)

    betaaudioj = (audio_j[:, i] - audio_j[:, j]) 
    betavideoj = (video_j[:, i] - video_j[:, j])
    betaj = torch.stack([betavideoj, betaaudioj], dim=1)

    # bnecause in this function, we only consider pair i and j, so it is similar with the binary classification task
    betai_mean = np.mean(betai.numpy(), axis=0) #torch.mean(betai, dim=0)
    betai_cov = np.cov(betai, rowvar=False)

    betaj_mean = np.mean(betaj.numpy(), axis=0) #torch.mean(betai, dim=0)
    betaj_cov = np.cov(betaj, rowvar=False)

    trainsetacc = {}
    alpha_vector = np.linspace(0.0, 1.0, 100)
    alpha = torch.FloatTensor(alpha_vector)
    Aij = ((0.5 + 0.5 * alpha_erf(alpha, betai_mean, betai_cov)) * priorlist[i] +
               (0.5 - 0.5 * alpha_erf(alpha, betaj_mean, betaj_cov)) * priorlist[j])
    kopt = torch.argmax(Aij)
    alpha_opt = alpha_vector[kopt]
    if 'Delta' in evaltype:
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
def Gaussian_test(trainaudiodata, trainvideodata, trainlabel, i, j):
    # split pair i and j samples based on the ground truth labels
    audio_i = trainaudiodata[trainlabel == i]#[:, i]
    audio_j = trainaudiodata[trainlabel == j]#[:, j]
    video_i = trainvideodata[trainlabel == i]#[:, i]
    video_j = trainvideodata[trainlabel == j]#[:, j]

    betaaudioi = (audio_i[:, i] - audio_i[:, j])
    betavideoi = (video_i[:, i] - video_i[:, j])
    betai = torch.stack([betavideoi, betaaudioi], dim=1)

    betaaudioj = (audio_j[:, i] - audio_j[:, j])
    betavideoj = (video_j[:, i] - video_j[:, j])
    betaj = torch.stack([betavideoj, betaaudioj], dim=1)

    skpval, sknormal = Henze_Zirkler_normality_test(betai)
    logging.info(f"p-value of the Henze-Zirkler normality test for class pairs {i} and {j} with ground-truth class {i} is {skpval}.")


    skpval, sknormal = Henze_Zirkler_normality_test(betaj)
    logging.info(f"p-value of the Henze-Zirkler normality test for class pairs {i} and {j} with ground-truth class {j} is {skpval}.")

def Pairwise_grid_search(traindict, devdict, savedir, evaltype, N, priorlist, ifgaussiantest):
    logging.basicConfig(filename=os.path.join(savedir, 'train.log'), level=logging.INFO)

    # alpha is a N * N matrix, where N is the number of classes
    alpha = np.zeros((N, N))

    start_time = time.time()
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
                alpha_opt = optimize_alpha(trainaudiodata, trainvideodata, trainlabel, N, i, j, priorlist, savedir, evaltype)
            alpha[i][j] = alpha_opt
    print(alpha)
    end_time = time.time()
    time_usage = end_time - start_time
    logging.info(f"{evaltype} training stage takes {time_usage} seconds to complete.")
    # Gaussian Test
    if ifgaussiantest is True:
        for i in range(N):
            for j in range(N):
                if i == j:
                    pass
                else:
                    Gaussian_test(trainaudiodata, trainvideodata, trainlabel, i, j)
    else:
        pass
    start_time = time.time()
    if 'Delta' in evaltype:
        videopred = []
        audiopred = []
        multipred = []
        labels = []
        output = {}
        for i, uttid in enumerate(tqdm(list(devdict.keys()))):
            DM = torch.zeros((N, N))
            for i in range(N):
                for j in range(N):
                    alpha_ij = alpha[i][j]
                    logitmix = alpha_ij * torch.FloatTensor(devdict[uttid]['audioprob'][0]) + (
                                1 - alpha_ij) * torch.FloatTensor(devdict[uttid]['videoprob'][0])
                    p = torch.softmax(logitmix, dim=-1)
                    DM[i][j] = p[i] - p[j]
            pp = (1 + torch.sum(DM, dim=1)) / N

            videopred.append(torch.argmax(torch.softmax(torch.FloatTensor(devdict[uttid]['videoprob'][0]), dim=-1), dim=-1))
            audiopred.append(torch.argmax(torch.softmax(torch.FloatTensor(devdict[uttid]['audioprob'][0]), dim=-1), dim=-1))
            multipred.append(torch.argmax(pp, dim=-1))

            labels.append(torch.LongTensor(devdict[uttid]['label'][0]))
            output.update({uttid: {}})
            output[uttid].update({'prob': pp.tolist()})
            output[uttid].update({'predict': int(torch.argmax(pp, dim=-1).data.numpy())})
            output[uttid].update({'label': int(devdict[uttid]['label'][0])})

        videopred = torch.LongTensor(videopred)
        audiopred = torch.LongTensor(audiopred)
        multipred = torch.LongTensor(multipred)
        labels = torch.LongTensor(labels)
        print(videopred.size())
        print(audiopred.size())
        print(labels.size())

        videoscore = acc(videopred, labels)
        audioscore = acc(audiopred, labels)
        multiscore = acc(multipred, labels)

        with open(os.path.join(savedir, 'videoacc_' + str(videoscore) + '_audioacc' + str(
                audioscore) + '_multiacc' + str(multiscore) + ".json"), 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)
    else:
        N_samples = len(devdict.keys())
        videopreds = []
        audiopreds = []
        multipred = []
        multiforcepred = []
        labels = []
        output = {}
        for i, uttid in enumerate(tqdm(list(devdict.keys()))):
            audiopred = torch.argmax(torch.softmax(torch.FloatTensor(devdict[uttid]['audioprob'][0]), dim=-1), dim=-1)
            videopred = torch.argmax(torch.softmax(torch.FloatTensor(devdict[uttid]['videoprob'][0]), dim=-1), dim=-1)

            if audiopred == videopred:
                results = audiopred
                results_force = audiopred

            else:
                i = audiopred
                j = videopred
                alpha_ij = alpha[i][j]
                x = alpha_ij * torch.FloatTensor(devdict[uttid]['audioprob'][0]) + (
                                1 - alpha_ij) * torch.FloatTensor(devdict[uttid]['videoprob'][0])
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

            labels.append(torch.LongTensor(devdict[uttid]['label'][0]))
            output.update({uttid: {}})
            output[uttid].update({'predict': int(results)})
            output[uttid].update({'predict_force': int(results_force)})
            output[uttid].update({'label': int(devdict[uttid]['label'][0])})

        videopreds = torch.LongTensor(videopreds)
        audiopreds = torch.LongTensor(audiopreds)
        multipred = torch.LongTensor(multipred)
        multiforcepred = torch.LongTensor(multiforcepred)
        labels = torch.LongTensor(labels)
        print(videopreds.size())
        print(audiopreds.size())
        print(labels.size())
        videoscore = acc(videopreds, labels)
        audioscore = acc(audiopreds, labels)
        multiscore = acc(multipred, labels)
        multiforcescore = acc(multiforcepred, labels)

        with open(os.path.join(savedir, 'videoacc_' + str(videoscore) + '_audioacc' + str(
                audioscore) + '_multiacc' + str(multiscore) + '_multiforceacc' + str(multiforcescore) + ".json"), 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)
    end_time = time.time()
    time_usage = end_time - start_time
    logging.info(f"Test stage takes {time_usage} seconds to complete.")





def get_args():
    parser = argparse.ArgumentParser()

    # get arguments from outside
    parser.add_argument('--datasdir', default='./../../../Dataset', type=str, help='Dir saves the datasource information')
    parser.add_argument('--evaltype', default='Pairwise', type=str, help='which data stream')
    parser.add_argument('--audiomodal', default='audio', type=str, help='which data stream')
    parser.add_argument('--videomodal', default='video', type=str, help='single or multi images')
    parser.add_argument('--savedir', default='./../../../trained', type=str, help='Dir to save trained model and results')
    parser.add_argument('--ifgaussiantest', default='true', type=str, help='if using Henze-Zirkler multivariate normality test')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    print("Arguments:")
    [print(key, val) for key, val in vars(args).items()]

    datasdir = args.datasdir
    evaltype = args.evaltype
    audiomodal = args.audiomodal
    videomodal = args.videomodal
    savedir = args.savedir
    ifgaussiantest = args.ifgaussiantest
    if ifgaussiantest == 'true':
        ifgaussiantest = True
    else:
        ifgaussiantest = False

    savedir = os.path.join(savedir, 'Multimodal', evaltype)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    N = 10  # the number of classes
    with open(os.path.join(datasdir, "train.json"), encoding="utf8") as json_file:
        traindict = json.load(json_file)
    with open(os.path.join(datasdir, "dev.json"), encoding="utf8") as json_file:
        devdict = json.load(json_file)
    with open(os.path.join(datasdir, "test.json"), encoding="utf8") as json_file:
        testdict = json.load(json_file)

    '''import random
    traindict = {k: traindict[k] for k in list(random.sample(list(traindict.keys()), 100))}
    devdict = {k: devdict[k] for k in list(random.sample(list(devdict.keys()), 100))}'''

    prior = {}
    for i in range(N):
        prior.update({i: []})
    for workdict, dset in zip([devdict, testdict], ["dev", "test"]):
        videofeatscpdir = os.path.join(args.datasdir, 'data', videomodal, dset + 'feats.scp')
        videoprobscpdir = os.path.join(args.datasdir, 'data', videomodal, dset + 'prob.scp')
        audiofeatscpdir = os.path.join(args.datasdir, 'data', audiomodal, dset + 'feats.scp')
        audioprobscpdir = os.path.join(args.datasdir, 'data', audiomodal, dset + 'prob.scp')
        labelscpdir = os.path.join(args.datasdir, 'data', audiomodal, dset + 'label.scp')

        audiofeat, audioprob, videofeat, videoprob, label = {}, {}, {}, {}, {}
        for scpfiles in [videofeatscpdir, videoprobscpdir, audiofeatscpdir, audioprobscpdir, labelscpdir]:
            with open(scpfiles) as f:
                srcdata = f.readlines()
            datadict = {}
            for j in srcdata:
                fullname = j.split(' ')[0]
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
                    filepath=audioprob[filename], filetype='mat')})
                savedict[filename].update({'videoprob': _get_from_loader(
                    filepath=videoprob[filename], filetype='mat')})
                labeldata = _get_from_loader(filepath=label[filename], filetype='mat')

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

    #print(priorlist)
    Pairwise_grid_search(devdict, testdict, savedir, evaltype, N, priorlist, ifgaussiantest)









