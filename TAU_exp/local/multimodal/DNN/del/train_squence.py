import os, sys
import json
import argparse
import numpy as np
from tqdm import tqdm
from model import *
from utils import *
from transformers import get_linear_schedule_with_warmup, AdamW
from torchmetrics.classification import MulticlassF1Score
import random

SEED=666
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
def find_best(filename):
    filedict = {}
    for i in filename:
        score = float(i.split('_')[-1].strip('.pkl'))
        filedict.update({i: score})
    Keymax = max(zip(filedict.values(), filedict.keys()))[1]
    return Keymax, filedict[Keymax]
def training(config):
    traindict = config["traindict"]
    devdict = config["devdict"]
    testdict = config["testdict"]

    dataset = Loaddatasetclass(train_file=traindict,
                                          dev_file=devdict,
                                          test_file=testdict)  # ,
    # max_len=max_uttlen)

    print(len(dataset.train_dataset))
    print(len(dataset.dev_dataset))
    print(len(dataset.test_dataset))

    resultsdir = config["resultsdir"]
    modeldir = config["modeldir"]

    evalacc_best = 0
    early_wait = 10
    run_wait = 1
    continuescore = 0
    stop_counter = 0
    f1score = MulticlassF1Score(num_classes=7, average='weighted')
    #loss_weights = torch.FloatTensor([0.30427062, 1.19699616, 5.47007183, 1.95437696,
    #                                  0.84847735, 5.42461417, 1.21859721])
    #criterion = torch.nn.CrossEntropyLoss(weight=loss_weights.to(config["device"]), ignore_index=-7)
    criterion = torch.nn.CrossEntropyLoss()

    model = Embeddingfusion(iadim=1024, itdim=1024, hiddendim=512, n_head=2, dropout_rate=0.0, num_blocks=2, odim=7)
    model = model.to(config["device"])
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
    optimizer = AdamW(model.parameters(),
                                  lr=config["lr"],
                                  weight_decay=config["weight_decay"], no_deprecation_warning=True
                                  )
    train_examples_len = len(dataset.train_dataset)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(
                                                    train_examples_len / config["batch_size"]) * 10,
                                                num_training_steps=int(
                                                    train_examples_len / config["batch_size"]) * config["epochs"])

    data_loader_train = torch.utils.data.DataLoader(dataset.train_dataset, shuffle=True,
                                                        batch_size=config["batch_size"],
                                                        num_workers=config["NWORKER"],
                                                        collate_fn=Padding)

    data_loader_dev = torch.utils.data.DataLoader(dataset.dev_dataset, shuffle=True,
                                                      batch_size=config["batch_size"],
                                                      num_workers=config["NWORKER"],
                                                      collate_fn=Padding)
    data_loader_test = torch.utils.data.DataLoader(dataset.test_dataset, shuffle=True,
                                                      batch_size=config["batch_size"],
                                                      num_workers=config["NWORKER"],
                                                      collate_fn=Padding)

    for epoch in range(config["epochs"]):  # loop over the dataset multiple times
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        model.train()
        trainpredict = []
        trainprob = []
        trainlabel = []
        for i, data_all in enumerate(tqdm(data_loader_train), 0):
            audiofeat, audioprob, textfeat, textprob, labels, filenames = data_all
            data = [i.to(config["device"]) for i in [audiofeat, audioprob, textfeat, textprob]]
            labels = labels.to(config["device"]).squeeze(-1)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            #print("\r%f" % loss, end='')
            # print statistics
            tr_loss += loss.item()
            nb_tr_steps += 1
            prob = torch.softmax(outputs, dim=-1)
            predicted = torch.argmax(torch.softmax(outputs, dim=-1), dim=-1)
            trainprob.append(prob)
            trainpredict.extend(predicted.cpu().detach().tolist())
            trainlabel.extend(labels.cpu().data.numpy().tolist())
        trainlabel = torch.LongTensor(trainlabel)
        trainpredict = torch.LongTensor(trainpredict)
        trainallscore = f1score(trainpredict, trainlabel).cpu().data.numpy()
        trainallscore = float(trainallscore)

        #trainallscore = np.sum((np.array(trainpredict) == np.array(trainlabel)), axis=-1) / len(trainlabel)

        # Validation loss
        torch.cuda.empty_cache()
        evallossvec = []
        evalpred = []
        evallabel = []
        model.eval()
        outpre = {}
        total = 0
        for i, data_all in enumerate(tqdm(data_loader_dev), 0):
            with torch.no_grad():
                audiofeat, audioprob, textfeat, textprob, labels, filenames = data_all
                data = [i.to(config["device"]) for i in [audiofeat, audioprob, textfeat, textprob]]
                labels = labels.to(config["device"]).squeeze(-1)
                outputs  = model(data)
                dev_loss = criterion(outputs, labels)

                evallossvec.append(dev_loss.cpu().data.numpy())
                predicted = torch.argmax(torch.softmax(outputs, dim=-1), dim=-1)
                prob = torch.softmax(outputs, dim=-1).cpu().detach().tolist()
                evallabel.extend(labels.cpu().data.numpy().tolist())
                evalpred.extend(predicted.cpu().detach().tolist())
                total += labels.size(0)
                for i in range(len(filenames)):
                    outpre.update({filenames[i]: {}})
                    outpre[filenames[i]].update({'label': int(labels[i].cpu().detach())})
                    outpre[filenames[i]].update({'predict': int(predicted[i].cpu().detach())})
                    outpre[filenames[i]].update({'prob': prob[i]})

        evalpred = torch.LongTensor(evalpred)
        evallabel = torch.LongTensor(evallabel)
        allscore = f1score(evalpred, evallabel).cpu().data.numpy()
        allscore = float(allscore)
        #allscore = correct / total
        # evalacc = evalacc / len(evallabel)
        evallossmean = np.mean(np.array(evallossvec))
        for param_group in optimizer.param_groups:
            currentlr = param_group['lr']
        OUTPUT_DIR = os.path.join(modeldir,
                          str(epoch) + '_' + str(evallossmean) + '_' + str(
                              currentlr) + '_' + str(trainallscore)[:6] + '_' + str(
                              allscore)[:6] + '.pkl')
        torch.save(model, OUTPUT_DIR)
        with open(os.path.join(resultsdir, str(epoch) + '_' + str(evallossmean) + '_' + str(
                                        currentlr) + '_' + str(trainallscore)[:6] + '_' + str(
                                        allscore)[:6] + ".json"), 'w', encoding='utf-8') as f:
            json.dump(outpre, f, ensure_ascii=False, indent=4)


        torch.cuda.empty_cache()
        if allscore <= evalacc_best:
            stop_counter = stop_counter + 1
            print('no improvement')
            continuescore = 0
        else:
            print('new score')
            evalacc_best = allscore
            continuescore = continuescore + 1

        if continuescore >= run_wait:
            stop_counter = 0
        print(stop_counter)
        print(early_wait)
        if stop_counter < early_wait:
            pass
        else:
            break


    besttrainmodel, besttrainacc = find_best(os.listdir(modeldir))
    model = torch.load(os.path.join(modeldir, besttrainmodel), map_location=config["device"])
    testpred = []
    testlabel = []
    model.eval()
    outpre = {}
    total = 0
    for i, data_all in enumerate(tqdm(data_loader_test), 0):
        with torch.no_grad():
            audiofeat, audioprob, textfeat, textprob, labels, filenames = data_all
            data = [i.to(config["device"]) for i in [audiofeat, audioprob, textfeat, textprob]]
            labels = labels.to(config["device"]).squeeze(-1)
            outputs = model(data)

            predicted = torch.argmax(torch.softmax(outputs, dim=-1), dim=-1)
            prob = torch.softmax(outputs, dim=-1).cpu().detach().tolist()
            testlabel.extend(labels.cpu().data.numpy().tolist())
            testpred.extend(predicted.cpu().detach().tolist())
            total += labels.size(0)
            for i in range(len(filenames)):
                outpre.update({filenames[i]: {}})
                outpre[filenames[i]].update({'label': int(labels[i].cpu().detach())})
                outpre[filenames[i]].update({'predict': int(predicted[i].cpu().detach())})
                outpre[filenames[i]].update({'prob': prob[i]})

    testpred = torch.LongTensor(testpred)
    testlabel = torch.LongTensor(testlabel)
    allscore = f1score(testpred, testlabel).cpu().data.numpy()
    testacc = float(allscore)
    with open(os.path.join(resultsdir,
                           'bestdevf1_' + str(besttrainacc)[:6] + '_besttestf1_' + str(testacc)[:6] + ".json"), 'w',
              encoding='utf-8') as f:
        json.dump(outpre, f, ensure_ascii=False, indent=4)

def get_args():
    parser = argparse.ArgumentParser()

    # get arguments from outside
    parser.add_argument('--datasdir', default='./../../../Dataset', type=str, help='Dir saves the datasource information')
    parser.add_argument('--modal', default='embedding_fusion', type=str, help='which data stream')
    parser.add_argument('--textmodal', default='text_SPCL', type=str, help='which data stream')
    parser.add_argument('--audiomodal', default='audio_DWFormer', type=str, help='single or multi images')
    parser.add_argument('--savedir', default='./../../../trained', type=str, help='Dir to save trained model and results')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = get_args()
    print("Arguments:")
    [print(key, val) for key, val in vars(args).items()]

    datasetdir = args.datasdir
    modal = args.modal
    textmodal = args.textmodal
    audiomodal = args.audiomodal
    savedir = args.savedir

    if torch.cuda.is_available() == True:
        device = 'cuda'
    else:
        device = 'cpu'

    modeldir = os.path.join(savedir, modal, 'model')
    resultsdir = os.path.join(savedir, modal, 'results')

    max_num_epochs = 100

    for makedir in [modeldir, resultsdir]:
        if not os.path.exists(makedir):
            os.makedirs(makedir)


    with open(os.path.join(datasetdir, "train.json"), encoding="utf8") as json_file:
        traindict = json.load(json_file)
    with open(os.path.join(datasetdir, "dev.json"), encoding="utf8") as json_file:
        devdict = json.load(json_file)
    with open(os.path.join(datasetdir, "test.json"), encoding="utf8") as json_file:
        testdict = json.load(json_file)

    '''import random
    traindict = {k: traindict[k] for k in list(random.sample(list(traindict.keys()), 20))}
    devdict = {k: devdict[k] for k in list(random.sample(list(devdict.keys()), 20))}
    testdict = {k: testdict[k] for k in list(random.sample(list(testdict.keys()), 20))}'''

    for workdict, dset in zip([devdict, testdict, traindict], ["dev", "test", "train"]):
        textfeatscpdir = os.path.join(datasetdir, 'data', textmodal, dset + 'featssequence.scp')
        textprobscpdir = os.path.join(datasetdir, 'data', textmodal, dset + 'prob.scp')
        audiofeatscpdir = os.path.join(datasetdir, 'data', audiomodal, dset + 'featssequence.scp')
        audioprobscpdir = os.path.join(datasetdir, 'data', audiomodal, dset + 'prob.scp')
        labelscpdir = os.path.join(datasetdir, 'data', audiomodal, dset + 'label.scp')

        textfeat, textprob, audiofeat, audioprob, label = {}, {}, {}, {}, {}
        for scpfiles in [textfeatscpdir, textprobscpdir, audiofeatscpdir, audioprobscpdir, labelscpdir]:
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
                savedict[filename].update({'textfeat': textfeat[filename]})
                savedict[filename].update({'audioprob': audioprob[filename]})
                savedict[filename].update({'textprob': textprob[filename]})  # .replace('./', './../../../')
                savedict[filename].update({'audiofeat': audiofeat[filename]})

                savedict[filename].update({'label': label[filename]})


            except:
                del savedict[filename]

        if dset == 'dev':
            devdict = savedict
        elif dset == 'test':
            testdict = savedict
        elif dset == 'train':
            traindict = savedict



    config = {
        "traindict": traindict,
        "testdict": testdict,
        "devdict": devdict,
        "NWORKER": 0,
        "device": device,
        "modal": modal,
        "weight_decay": 0.1,
        "lr": 0.0005,  # tune.choice([5e-2, 5e-3, 5e-4, 5e-5]),
        "batch_size": 64,
        "modeldir": modeldir,
        "resultsdir": resultsdir,
        "epochs": max_num_epochs  # tune.choice([3, 5, 10, 15])
    }
    training(config)
    #"hidden_dropout": 0.1,
    #"attention_dropout": 0.1,
    #"n_atts": 2,  # 6, #
    #"max_uttlen": 174,



