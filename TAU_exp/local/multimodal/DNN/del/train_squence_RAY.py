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
from ray import tune, ray_constants
from ray.tune import CLIReporter
os.environ["RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"] = "1"
from ray.tune.schedulers import ASHAScheduler

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
def training(config, dataset=None):
    print(len(dataset.train_dataset))
    print(len(dataset.dev_dataset))
    print(len(dataset.test_dataset))

    f1score = MulticlassF1Score(num_classes=7, average='weighted')
    #loss_weights = torch.FloatTensor([0.30427062, 1.19699616, 5.47007183, 1.95437696,
    #                                  0.84847735, 5.42461417, 1.21859721])
    #criterion = torch.nn.CrossEntropyLoss(weight=loss_weights.to(config["device"]), ignore_index=-7)
    criterion = torch.nn.CrossEntropyLoss()

    model = Embeddingfusion(iadim=1024, itdim=1024, hiddendim=config["hiddendim"], n_head=config["n_head"],
                            dropout_rate=config["dropout_rate"], num_blocks=config["num_blocks"], odim=7)
    model = model.to(config["device"])
    '''if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)'''
    optimizer = AdamW(model.parameters(),
                                  lr=config["lr"],
                                  weight_decay=config["weight_decay"], no_deprecation_warning=True
                                  )
    train_examples_len = len(dataset.train_dataset)
    '''scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(
                                                    train_examples_len / config["batch_size"]) * 10,
                                                num_training_steps=int(
                                                    train_examples_len / config["batch_size"]) * config["epochs"])'''

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
            #scheduler.step()
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
        for i, data_all in enumerate(tqdm(data_loader_test), 0):
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
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=evallossmean, accuracy=allscore)
        print("Finished Training")

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

    max_num_epochs = 15

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
                savedict[filename].update({'textfeat': textfeat[filename].replace('./', '/data/wentao/MELD_laden/')})
                savedict[filename].update({'audioprob': audioprob[filename].replace('./', '/data/wentao/MELD_laden/')})
                savedict[filename].update({'textprob': textprob[filename].replace('./', '/data/wentao/MELD_laden/')})  # .replace('./', './../../../')
                savedict[filename].update({'audiofeat': audiofeat[filename].replace('./', '/data/wentao/MELD_laden/')})
                savedict[filename].update({'label': label[filename].replace('./', '/data/wentao/MELD_laden/')})


            except:
                del savedict[filename]

        if dset == 'dev':
            devdict = savedict
        elif dset == 'test':
            testdict = savedict
        elif dset == 'train':
            traindict = savedict

    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    dataset = Loaddatasetclass(train_file=traindict,
                                          dev_file=devdict,
                                          test_file=testdict)  # ,

    config = {
        "NWORKER": 0,
        "hiddendim": tune.choice([256, 512, 1024]),
        "n_head": tune.choice([2, 4, 8, 16]),
        "dropout_rate": tune.choice([0.0, 0.1, 0.5]),
        "num_blocks": tune.choice([2, 4, 6]),
        "device": device,
        "modal": modal,
        "weight_decay": 0.1,
        "lr": tune.choice([5e-2, 5e-3, 5e-4, 5e-5]),
        "batch_size": 64,
        "modeldir": modeldir,
        "resultsdir": resultsdir,
        "epochs": max_num_epochs  # tune.choice([3, 5, 10, 15])
    }
    #training(config, dataset=dataset)
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    # training(config, dataset=dataset)
    result = tune.run(
        tune.with_parameters(training, dataset=dataset),
        resources_per_trial={"cpu": 8, "gpu": 1},
        config=config,
        num_samples=20,
        local_dir=os.path.join(savedir, modal, "RAY"),
        scheduler=scheduler,
        progress_reporter=reporter)
    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    #"hidden_dropout": 0.1,
    #"attention_dropout": 0.1,
    #"n_atts": 2,  # 6, #
    #"max_uttlen": 174,



