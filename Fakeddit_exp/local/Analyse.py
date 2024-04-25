import os, sys
import json
import numpy as np
import argparse
import matplotlib.pyplot as plt
import logging
import statistics
def draw_plot(data, offset,edge_color, fill_color, ax, widths):
    pos = np.arange(data.shape[1])+offset
    bp = ax.boxplot(data, positions= pos, widths=widths, patch_artist=True, manage_ticks=False, sym='')
    for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)
    for patch in bp['boxes']:
        patch.set(facecolor=fill_color)

def plot(allacc, allacc1, modal, modal1, imagedir):
    fontsize = 15
    ticks = ['18', '32', '64', '128', '256', '512']  # , '1024', '2048']
    data = []
    labels = []
    x = [0, 1, 2, 3, 4, 5]  # , 8, 9]
    for nsplit in ticks:
        data.append(allacc[nsplit])
        labels.append(int(nsplit))

    data1 = []
    for nsplit in ['18', '32', '64', '128', '256', '512']:  # , '1024', '2048']:
        data1.append(allacc1[nsplit])
    plt.figure(figsize=(4, 3))
    plt.ylabel('Accuracy')
    plt.xlabel('Trainsubset size')

    plt.title('Accuracy changed based on trainsubset size')
    fig, ax = plt.subplots()
    draw_plot(np.transpose(np.array(data)), -0.2, "tomato", "white", ax)
    draw_plot(np.transpose(np.array(data1)), +0.2, "skyblue", "white", ax)
    plt.xticks(x)
    # plt.savefig(__file__+'.png', bbox_inches='tight')

    # datasw = plt.boxplot(data)
    # datagaussian = plt.boxplot(datagaussian)
    # set_box_color(datasw, '#D7191C')  # colors are from http://colorbrewer2.org/
    # set_box_color(datagaussian, '#2C7BB6')

    plt.plot([], c='tomato', label=modal)
    plt.plot([], c='skyblue', label=modal1)
    plt.legend(fontsize=fontsize)
    plt.xlabel('The size of sub-training set', fontsize=fontsize)
    plt.ylabel('Accuracy', fontsize=fontsize)

    plt.xticks(range(len(labels)), labels, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    # plt.xlim(-2, len(ticks))
    plt.tight_layout()

    plt.savefig(os.path.join(imagedir, modal + '-' + modal1 + '.pdf'))

def multi_plot(allacc, allmodals, imagedir):
    fontsize = 15
    ticks = ['18', '32', '64', '128', '256', '512']  # , '1024', '2048']
    data = {}
    for i in range(len(allacc)):
        data.update({i: []})
    labels = []
    x = [0, 1, 2, 3, 4, 5]  # , 8, 9]
    for nsplit in ticks:
        for i in range(len(allacc)):
            data[i].append(allacc[i][nsplit])
        labels.append(int(nsplit))

    plt.figure(figsize=(4, 3))
    plt.ylabel('Accuracy')
    plt.xlabel('Trainsubset size')

    plt.title('Accuracy changed based on trainsubset size')
    fig, ax = plt.subplots()
    if len(allacc) == 3:
        draw_plot(np.transpose(np.array(data[0])), -0.2, "tomato", "white", ax, widths=0.1)
        draw_plot(np.transpose(np.array(data[1])), +0, "skyblue", "white", ax, widths=0.1)
        draw_plot(np.transpose(np.array(data[2])), +0.2, "darkgreen", "white", ax, widths=0.1)
        plt.xticks(x)

        plt.plot([], c='tomato', label=allmodals[0])
        plt.plot([], c='skyblue', label=allmodals[1])
        plt.plot([], c='darkgreen', label=allmodals[2])
        plt.legend(fontsize=10, loc='upper center', bbox_to_anchor=(0.5, 1.1),
                   ncol=3, fancybox=True, shadow=True)
    else:
        draw_plot(np.transpose(np.array(data[0])), -0.3, "tomato", "white", ax, widths=0.1)
        draw_plot(np.transpose(np.array(data[1])), -0.1, "skyblue", "white", ax, widths=0.1)
        draw_plot(np.transpose(np.array(data[2])), +0.1, "darkgreen", "white", ax, widths=0.1)
        draw_plot(np.transpose(np.array(data[3])), +0.3, "dimgray", "white", ax, widths=0.1)
        plt.xticks(x)
        plt.plot([], c='tomato', label=allmodals[0])
        plt.plot([], c='skyblue', label=allmodals[1])
        plt.plot([], c='darkgreen', label=allmodals[2])
        plt.plot([], c='dimgray', label=allmodals[3])
        plt.ylim(0.8, 0.95)
        plt.legend(fontsize=fontsize, loc='upper center', bbox_to_anchor=(0.5, 1.3),
                   ncol=2, fancybox=True, shadow=True)
        #plt.legend(fontsize=fontsize)

    plt.xlabel('The size of sub-training set', fontsize=fontsize)
    plt.ylabel('Accuracy', fontsize=fontsize)

    plt.xticks(range(len(labels)), labels, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    # plt.xlim(-2, len(ticks))
    plt.tight_layout()

    plt.savefig(os.path.join(imagedir, '-'.join(allmodals) + '.pdf'))

def print_statistic(datadict, modelname):
    keys = list(datadict.keys())
    for key in keys:
        mean = statistics.mean(datadict[key])
        std = statistics.stdev(datadict[key])
        maxacc = max(datadict[key])
        minacc = min(datadict[key])
        dis = abs(maxacc - minacc)
        print(modelname + ' ' + key + ' samples mean is ' + str(round(mean, 3))
              + ' standard dividition is ' + str(round(std, 3)) +
                " acc difference is is:" + str(round(dis, 3)))
def get_args():
    parser = argparse.ArgumentParser()

    # get arguments from outside
    parser.add_argument('--imagedir', default='./../Images', type=str, help='Dir saves the datasource information')
    parser.add_argument('--savedir', default='./../trained', type=str, help='Dir to save trained model and results')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    print("Arguments:")
    [print(key, val) for key, val in vars(args).items()]
    fontsize = 7
    imagedir = args.imagedir
    savedir = args.savedir
    if not os.path.exists(imagedir):
        os.makedirs(imagedir)

    logging.basicConfig(filename=os.path.join(imagedir, 'Analyse.log'), level=logging.INFO)

    ## Global results ##
    globalsavedir = os.path.join(savedir, 'Multimodal', 'Global_fixed')
    for modal in ['GSW_linear', 'GSW_logarithmic', 'GSW_logit']:
        filedir = os.path.join(globalsavedir, modal)
        for subdir, type in zip([filedir], [modal]):
            with open(os.path.join(subdir, os.listdir(subdir)[0]), encoding="utf8") as json_file:
                filedict = json.load(json_file)
            counts = len(filedict)
            i = 0
            for j in list(filedict.keys()):
                if filedict[j]['predict'] == filedict[j]['label']:
                    i = i + 1
            acc = i / counts
            logging.info(f'{type} accuracy is {acc}.')
    for modal in ['DSW', 'representation']:
        filedir = os.path.join(savedir, 'Multimodal', modal)
        for subdir, type in zip([filedir], [modal]):
            new_subdir = os.path.join(subdir, 'results')
            for i in os.listdir(new_subdir):
                if 'best' in i:
                    filename = i
            with open(os.path.join(new_subdir, filename), encoding="utf8") as json_file:
                filedict = json.load(json_file)
            counts = len(filedict)
            i = 0
            for j in list(filedict.keys()):
                if filedict[j]['predict'] == filedict[j]['label']:
                    i = i + 1
            acc = i / counts
            logging.info(f'{type} accuracy is {acc}.')

    ## pairwise results ##
    Delta_rep_acc_dict = {}
    modal = 'Delta_representation'
    filedir = os.path.join(savedir, 'Multimodal', modal)
    for i in os.listdir(filedir):
        if i.endswith('.json'):
            jsondir = os.path.join(filedir, i)
            with open(jsondir, encoding="utf8") as json_file:
                filedict = json.load(json_file)
            counts = len(filedict)
            i = 0
            for j in list(filedict.keys()):
                if filedict[j]['predict'] == filedict[j]['label']:
                    i = i + 1
            acc = i / counts
            logging.info(f'{modal} accuracy is {acc}.')
            Delta_rep_acc_dict.update({'GM': [acc for i in range(4)]})
    for covariance_type in ["full", "tied", "diag", "spherical"]:  # "full", "tied", "diag", "spherical"
        Delta_rep_acc_dict.update({'GMM_' + covariance_type: []})
        for ncomp in range(2, 6):
            subsavedir = os.path.join(savedir, 'Multimodal', modal + '_GMM', covariance_type, str(ncomp) + 'GMM')
            for i in os.listdir(subsavedir):
                if i.endswith('.json'):
                    jsondir = os.path.join(subsavedir, i)
                    with open(jsondir, encoding="utf8") as json_file:
                        filedict = json.load(json_file)
                    counts = len(filedict)
                    i = 0
                    for j in list(filedict.keys()):
                        if filedict[j]['predict'] == filedict[j]['label']:
                            i = i + 1
                    acc = i / counts
                    logging.info(f'{modal} + GMM with {covariance_type} and {ncomp} components accuracy is {acc}.')
                    Delta_rep_acc_dict['GMM_' + covariance_type].append(acc)

    plt.figure(figsize=(4, 3))
    plt.ylabel('Accuracy', fontsize=fontsize)
    plt.xlabel('GMM component size', fontsize=fontsize)
    x = [2, 3, 4, 5]
    plt.plot(x, Delta_rep_acc_dict['GM'], label='GM')
    plt.plot(x, Delta_rep_acc_dict['GMM_full'], label='GMM_full')
    plt.plot(x, Delta_rep_acc_dict['GMM_tied'], label='GMM_tied')
    plt.plot(x, Delta_rep_acc_dict['GMM_diag'], label='GMM_diag')
    plt.plot(x, Delta_rep_acc_dict['GMM_spherical'], label='GMM_spherical')

    plt.xticks(x, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(fontsize=fontsize, loc='upper center', bbox_to_anchor=(0.5, 1.15),
               ncol=3, fancybox=True, shadow=True)
    plt.tight_layout()
    plt.savefig(os.path.join(imagedir, 'Delta_representation_GMMs.pdf'))

    modal = 'Pairwise'
    Pairwise_acc_dict = {}
    Pairwise_force_acc_dict = {}
    filedir = os.path.join(savedir, 'Multimodal', modal)
    with open(os.path.join(filedir, os.listdir(filedir)[0]), encoding="utf8") as json_file:
        filedict = json.load(json_file)
    counts = len(filedict)
    i = 0
    for j in list(filedict.keys()):
        if filedict[j]['predict'] == filedict[j]['label']:
            i = i + 1
    acc = i / counts
    k = 0
    for j in list(filedict.keys()):
        if filedict[j]['predict_force'] == filedict[j]['label']:
            k = k + 1
    acc_force = k / counts
    logging.info(f'{modal} accuracy is {acc}.')
    logging.info(f'{modal}_force accuracy is {acc_force}.')
    Pairwise_acc_dict.update({'GM': [acc for i in range(4)]})
    Pairwise_force_acc_dict.update({'GM': [acc_force for i in range(4)]})

    for covariance_type in ["full", "tied", "diag", "spherical"]:  # "full", "tied", "diag", "spherical"
        Pairwise_acc_dict.update({'GMM_' + covariance_type: []})
        Pairwise_force_acc_dict.update({'GMM_' + covariance_type: []})
        for ncomp in range(2, 6):
            subsavedir = os.path.join(savedir, 'Multimodal', modal + '_GMM', covariance_type, str(ncomp) + 'GMM')
            with open(os.path.join(subsavedir, os.listdir(subsavedir)[0]), encoding="utf8") as json_file:
                filedict = json.load(json_file)
            counts = len(filedict)
            i = 0
            for j in list(filedict.keys()):
                if filedict[j]['predict'] == filedict[j]['label']:
                    i = i + 1
            acc = i / counts
            k = 0
            for j in list(filedict.keys()):
                if filedict[j]['predict_force'] == filedict[j]['label']:
                    k = k + 1
            acc_force = k / counts
            logging.info(f'{modal} + GMM with {covariance_type} and {ncomp} components accuracy is {acc}.')
            logging.info(f'{modal} + GMM with {covariance_type} and {ncomp} components forced accuracy is {acc_force}.')
            Pairwise_acc_dict['GMM_' + covariance_type].append(acc)
            Pairwise_force_acc_dict['GMM_' + covariance_type].append(acc_force)
    for workdict, type in zip([Pairwise_acc_dict, Pairwise_force_acc_dict], ['Pairwise', 'Pairwise_force']):
        plt.figure(figsize=(4, 3))
        plt.ylabel('Accuracy', fontsize=fontsize)
        plt.xlabel('GMM component size', fontsize=fontsize)
        x = [2, 3, 4, 5]
        plt.plot(x, workdict['GM'], label='GM')
        plt.plot(x, workdict['GMM_full'], label='GMM_full')
        plt.plot(x, workdict['GMM_tied'], label='GMM_tied')
        plt.plot(x, workdict['GMM_diag'], label='GMM_diag')
        plt.plot(x, workdict['GMM_spherical'], label='GMM_spherical')

        plt.xticks(x, fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.legend(fontsize=fontsize, loc='upper center', bbox_to_anchor=(0.5, 1.15),
                   ncol=3, fancybox=True, shadow=True)
        plt.tight_layout()
        # plt.show()
        plt.savefig(os.path.join(imagedir, type + '_GMM.pdf'))

    ###  CV comparison  ###
    modaldelta = "Delta_representation"
    deltaacc = {}
    for M in ['18', '32', '64', '128', '256', '512']:
        deltaacc.update({str(M): []})
        for i in range(10):
            filedir = os.path.join(savedir, 'Subset', modaldelta, str(M), str(i))
            for filename in os.listdir(filedir):
                if filename.endswith('.json'):
                    with open(os.path.join(filedir, filename), encoding='utf-8') as json_file:
                        filedict = json.load(json_file)
                    counts = len(filedict)
                    i = 0
                    for j in list(filedict.keys()):
                        if filedict[j]['predict'] == filedict[j]['label']:
                            i = i + 1
                    acc = i / counts
                    deltaacc[str(M)].append(acc)

    modalpairwise = "Pairwise"
    pairwiseacc = {}
    for M in ['18', '32', '64', '128', '256', '512']:
        pairwiseacc.update({str(M): []})
        for i in range(10):
            filedir = os.path.join(savedir, 'Subset', modalpairwise, str(M), str(i))
            with open(os.path.join(filedir, os.listdir(filedir)[0]), encoding="utf8") as json_file:
                filedict = json.load(json_file)
            counts = len(filedict)
            i = 0
            for j in list(filedict.keys()):
                if filedict[j]['predict'] == filedict[j]['label']:
                    i = i + 1
            acc = i / counts
            pairwiseacc[str(M)].append(acc)

    pairwiseforceacc = {}
    for M in ['18', '32', '64', '128', '256', '512']:
        pairwiseforceacc.update({str(M): []})
        for i in range(10):
            filedir = os.path.join(savedir, 'Subset', modalpairwise, str(M), str(i))
            with open(os.path.join(filedir, os.listdir(filedir)[0]), encoding="utf8") as json_file:
                filedict = json.load(json_file)
            counts = len(filedict)
            i = 0
            for j in list(filedict.keys()):
                if filedict[j]['predict_force'] == filedict[j]['label']:
                    i = i + 1
            acc = i / counts
            pairwiseforceacc[str(M)].append(acc)

    multi_plot([deltaacc, pairwiseacc, pairwiseforceacc], ['delta fusion', 'pairwise fusion', 'forced pairwise fusion'],
               imagedir)

    for modal in ['GSW_logarithmic']:
        modalreference = modal
        globalacc = {}
        for M in ['18', '32', '64', '128', '256', '512']:
            globalacc.update({str(M): []})
            for i in range(10):
                filedir = os.path.join(savedir, 'Subset', 'Global_fixed', modalreference, str(M), str(i))
                with open(os.path.join(filedir, os.listdir(filedir)[0]), encoding="utf8") as json_file:
                    filedict = json.load(json_file)
                counts = len(filedict)
                i = 0
                for j in list(filedict.keys()):
                    if filedict[j]['predict'] == filedict[j]['label']:
                        i = i + 1
                acc = i / counts
                globalacc[str(M)].append(acc)

    modal = 'DSW'
    modalreference = modal.split('_')[0]
    DSWacc = {}
    for M in ['18', '32', '64', '128', '256', '512']:
        DSWacc.update({str(M): []})
        for i in range(10):
            filedir = os.path.join(savedir, 'Subset', modal, str(M), str(i), 'results')
            for i in os.listdir(filedir):
                if i.startswith('best'):
                    jsondir = os.path.join(filedir, i)
                    with open(jsondir, encoding="utf8") as json_file:
                        filedict = json.load(json_file)
                    counts = len(filedict)
                    i = 0
                    for j in list(filedict.keys()):
                        if filedict[j]['predict'] == filedict[j]['label']:
                            i = i + 1
                    acc = i / counts
            DSWacc[str(M)].append(acc)

    modal = 'representation'
    modalreference = modal.split('_')[0]
    rfacc = {}
    for M in ['18', '32', '64', '128', '256', '512']:
        rfacc.update({str(M): []})
        for i in range(10):
            filedir = os.path.join(savedir, 'Subset', modal, str(M), str(i), 'results')
            for i in os.listdir(filedir):
                if i.startswith('best'):
                    jsondir = os.path.join(filedir, i)
                    with open(jsondir, encoding="utf8") as json_file:
                        filedict = json.load(json_file)
                    counts = len(filedict)
                    i = 0
                    for j in list(filedict.keys()):
                        if filedict[j]['predict'] == filedict[j]['label']:
                            i = i + 1
                    acc = i / counts
            rfacc[str(M)].append(acc)
    # for datadict, modelname in zip([deltaacc, pairwiseacc, pairwiseforceacc], ['delta fusion', 'pairwise fusion', 'forced pairwise fusion']):
    #    print_statistic(datadict, modelname)
    # for datadict, modelname in zip([deltaacc, globalacc, DSWacc, rfacc], ['delta fusion', 'global logarithmic', 'DSW', 'RF']):
    # print_statistic(datadict, modelname)
    multi_plot([deltaacc, globalacc, DSWacc, rfacc], ['delta fusion', 'global logarithmic', 'DSW', 'RF'],
               imagedir)
    





