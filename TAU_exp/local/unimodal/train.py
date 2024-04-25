import json
import argparse
from tqdm import tqdm
from model import *
from utils import *
from kaldiio import WriteHelper
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score
import random
import logging

SEED=666
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def training(config):
    traindict = config["traindict"]
    devdict = config["devdict"]
    testdict = config["testdict"]

    if 'audio' in config["modal"]:
        dataset = Audiofeatdatasetclass(train_file=traindict,
                                        dev_file=devdict,
                               test_file=testdict,
                                max_len=5000)
        padding = pad_audio_sequence
        model = Audiospectrumtransformer(odim=10, cachedir=config["cashedir"])
    elif 'video' in config["modal"]:
        dataset = Videofeatdatasetclass(train_file=traindict,
                                        dev_file=devdict,
                                        test_file=testdict,
                                        cashedir=config["cashedir"])
        padding = pad_video_sequence
        model = E2EViTsingle(odim=10, cachedir=config["cashedir"])


    resultsdir = config["resultsdir"]
    modeldir = config["modeldir"]
    logging.basicConfig(filename=os.path.join(modeldir, 'train.log'), level=logging.INFO)

    evalacc_best = 0
    early_wait = 5
    run_wait = 1
    continuescore = 0
    stop_counter = 0
    criterion = torch.nn.CrossEntropyLoss()


    model = model.to(config["device"])
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'The model has {trainable_parameters} trainable parameters')
    logging.info(f'Train set contains {len(dataset.train_dataset)} samples')
    logging.info(f'Dev set contains {len(dataset.dev_dataset)} samples')
    logging.info(f'Test set contains {len(dataset.test_dataset)} samples')
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config["lr"]
                                  )
    train_examples_len = len(dataset.train_dataset)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(
                                                    train_examples_len / config["batch_size"]) * 5,
                                                num_training_steps=int(
                                                    train_examples_len / config["batch_size"]) * config["epochs"])

    data_loader_train = torch.utils.data.DataLoader(dataset.train_dataset, shuffle=True,
                                                        batch_size=config["batch_size"],
                                                        num_workers=config["NWORKER"],
                                                        collate_fn=padding)

    data_loader_dev = torch.utils.data.DataLoader(dataset.dev_dataset, shuffle=True,
                                                      batch_size=config["batch_size"],
                                                      num_workers=config["NWORKER"],
                                                      collate_fn=padding)

    for epoch in range(config["epochs"]):  # loop over the dataset multiple times
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        model.train()
        trainpredict = []
        trainprob = []
        trainlabel = []
        for i, data in enumerate(tqdm(data_loader_train), 0):
            if 'audio' in config["modal"]:
                audio_feat = data[0].to(config["device"])
                label = data[1].to(config["device"])
                label = label.squeeze(-1)
                filename = data[2]
                # zero the parameter gradients
                optimizer.zero_grad()
                outputs, _ = model(audio_feat)
            elif 'video' in config["modal"]:
                image = data[0].to(config["device"])
                label = data[1].to(config["device"])
                label = label.squeeze(-1)
                # zero the parameter gradients
                optimizer.zero_grad()
                outputs, _ = model(image)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            scheduler.step()
            tr_loss += loss.item()
            nb_tr_steps += 1
            prob = torch.softmax(outputs, dim=-1)
            predicted = torch.argmax(torch.softmax(outputs, dim=-1), dim=-1)
            trainprob.append(prob)
            trainpredict.extend(predicted.cpu().detach().tolist())
            trainlabel.extend(label.cpu().data.numpy().tolist())
        trainprob = torch.concat(trainprob, dim=0)
        trainlabel = torch.LongTensor(trainlabel)
        trainallscore = accuracy_score(trainlabel, trainpredict)
        trainallscore = float(trainallscore)
        logging.info(f'Epoch {epoch} training accuracy score is {trainallscore}.')

        # Validation loss
        torch.cuda.empty_cache()
        evallossvec = []
        evalpred = []
        evallabel = []
        evalacc = 0
        model.eval()
        correct = 0
        outpre = {}
        total = 0
        for i, data in enumerate(tqdm(data_loader_dev), 0):
            with torch.no_grad():
                if 'audio' in config["modal"]:
                    audio_feat = data[0].to(config["device"])
                    label = data[1].to(config["device"])
                    labels = label.squeeze(-1)
                    filename = data[2]
                    outputs, _ = model(audio_feat)

                elif 'video' in config["modal"]:
                    image = data[0].to(config["device"])
                    label = data[1].to(config["device"])
                    labels = label.squeeze(-1)
                    filename = data[2]
                    outputs, _ = model(image)
                dev_loss = criterion(outputs, labels)
                evallossvec.append(dev_loss.cpu().data.numpy())
                predicted = torch.argmax(torch.softmax(outputs, dim=-1), dim=-1)
                prob = torch.softmax(outputs, dim=-1).cpu().detach().tolist()
                probsave = torch.softmax(outputs, dim=-1)
                evalpred.append(predicted.cpu().data.numpy().tolist())
                evallabel.extend(labels.cpu().data.numpy().tolist())
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                for i in range(len(filename)):
                    outpre.update({filename[i]: {}})
                    outpre[filename[i]].update({'label': int(labels[i].cpu().detach())})
                    outpre[filename[i]].update({'predict': int(predicted[i].cpu().detach())})
                    outpre[filename[i]].update({'prob': prob[i]})

        allscore = correct / total
        allscore = float(allscore)
        evallossmean = np.mean(np.array(evallossvec))
        logging.info(f'Epoch {epoch} evaluation accuracy score is {allscore}, loss is {evallossmean}.')

        for param_group in optimizer.param_groups:
            currentlr = param_group['lr']
        OUTPUT_DIR = os.path.join(modeldir,'bestmodel.pkl')

        with open(os.path.join(resultsdir, str(epoch) + '_' + str(evallossmean) + '_' + str(
                                        currentlr) + '_' + str(trainallscore)[:6] + '_' + str(
                                        allscore)[:6] + ".json"), 'w', encoding='utf-8') as f:
            json.dump(outpre, f, ensure_ascii=False, indent=4)


        torch.cuda.empty_cache()
        if allscore < evalacc_best:
            stop_counter = stop_counter + 1
            print('no improvement')
            continuescore = 0
        else:
            print('new score')
            evalacc_best = allscore
            continuescore = continuescore + 1
            torch.save(model, OUTPUT_DIR)

        if continuescore >= run_wait:
            stop_counter = 0
        print(stop_counter)
        print(early_wait)
        if stop_counter < early_wait:
            pass
        else:
            break

    model = torch.load(os.path.join(modeldir, 'bestmodel.pkl'), map_location=config["device"])
    data_loader_test = torch.utils.data.DataLoader(dataset.test_dataset, shuffle=True,
                                                  batch_size=config["batch_size"],
                                                  num_workers=config["NWORKER"],
                                                  collate_fn=padding)
    torch.cuda.empty_cache()
    evalpred = []
    evallabel = []
    model.eval()
    correct = 0
    outpre = {}
    total = 0
    for i, data in enumerate(tqdm(data_loader_test), 0):
        with torch.no_grad():
            if 'audio' in config["modal"]:
                audio_feat = data[0].to(config["device"])
                label = data[1].to(config["device"])
                labels = label.squeeze(-1)
                filename = data[2]
                outputs, _ = model(audio_feat)

            elif 'video' in config["modal"]:
                image = data[0].to(config["device"])
                label = data[1].to(config["device"])
                labels = label.squeeze(-1)
                filename = data[2]
                outputs, _ = model(image)
            predicted = torch.argmax(torch.softmax(outputs, dim=-1), dim=-1)
            prob = torch.softmax(outputs, dim=-1).cpu().detach().tolist()
            evalpred.append(predicted.cpu().data.numpy().tolist())
            evallabel.extend(labels.cpu().data.numpy().tolist())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for i in range(len(filename)):
                outpre.update({filename[i]: {}})
                outpre[filename[i]].update({'label': int(labels[i].cpu().detach())})
                outpre[filename[i]].update({'predict': int(predicted[i].cpu().detach())})
                outpre[filename[i]].update({'prob': prob[i]})
    allscore = correct / total
    allscore = float(allscore)

    with open(os.path.join(resultsdir, 'test_' + str(
        allscore)[:6] + ".json"), 'w', encoding='utf-8') as f:
        json.dump(outpre, f, ensure_ascii=False, indent=4)



    for dset, dataloader in zip(['dev', 'test', 'train'], [data_loader_dev, data_loader_test, data_loader_train]): #'dev', 'test' , data_loader_dev, data_loader_test
        probarksavedir = 'ark,scp:' + os.path.join(config["savefeaturesdir"], dset + 'prob.ark') + ',' + os.path.join(
            config["savefeaturesdir"],
            dset + 'prob.scp')
        featsarksavedir = 'ark,scp:' + os.path.join(config["savefeaturesdir"], dset + 'feats.ark') + ',' + os.path.join(
            config["savefeaturesdir"], dset + 'feats.scp')
        labelarksavedir = 'ark,scp:' + os.path.join(config["savefeaturesdir"], dset + 'label.ark') + ',' + os.path.join(
            config["savefeaturesdir"], dset + 'label.scp')
        model.eval()
        outpre = {}
        for i, data in enumerate(tqdm(dataloader), 0):
            with torch.no_grad():
                if 'audio' in config["modal"]:
                    audio_feat = data[0].to(config["device"])
                    label = data[1].to(config["device"])
                    labels = label.squeeze(-1)
                    filename = data[2]
                    outputs, feats = model(audio_feat)
                elif 'video' in config["modal"]:
                    image = data[0].to(config["device"])
                    label = data[1].to(config["device"])
                    labels = label.squeeze(-1)
                    filename = data[2]
                    outputs, feats = model(image)
                for i in range(len(filename)):
                    logit = np.expand_dims(outputs[i, :].cpu().data.numpy(), axis=0)
                    feat = np.expand_dims(feats[i, :].cpu().data.numpy(), axis=0)
                    label = np.expand_dims(np.expand_dims(labels[i].cpu().data.numpy(), axis=0), axis=0)
                    outpre.update({filename[i]: {}})
                    outpre[filename[i]].update({'logit': logit})
                    outpre[filename[i]].update({'feat': feat})
                    outpre[filename[i]].update({'label': label})
        with WriteHelper(probarksavedir, compression_method=2) as writer1:
            with WriteHelper(featsarksavedir, compression_method=2) as writer2:
                with WriteHelper(labelarksavedir, compression_method=2) as writer3:
                    for ids in tqdm(list(outpre.keys())):
                        writer1(ids, outpre[ids]['logit'])
                        writer2(ids, outpre[ids]['feat'])
                        writer3(ids, outpre[ids]['label'])



def get_args():
    parser = argparse.ArgumentParser()

    # get arguments from outside
    parser.add_argument('--datadir', default='./../../Dataset', type=str, help='Dir saves the datasource information')
    parser.add_argument('--featdir', default='./../../Dataset/Features', type=str, help='which data stream')
    parser.add_argument('--savedir', default='./../../trained', type=str, help='which data stream')
    parser.add_argument('--modal', default='video', type=str, help='which data stream')
    parser.add_argument('--cashedir', default='./../../CACHE', type=str, help='which data stream')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    print("Arguments:")
    [print(key, val) for key, val in vars(args).items()]

    datadir = args.datadir
    vfeatdir = args.featdir
    savedir = args.savedir
    modal = args.modal
    cashedir = args.cashedir

    savefeaturesdir = os.path.join(datadir, 'data', modal)

    if torch.cuda.is_available() == True:
        device = 'cuda'
    else:
        device = 'cpu'
    modeldir = os.path.join(savedir, 'Unimodal', modal, 'model')
    resultsdir = os.path.join(savedir, 'Unimodal', modal, 'results')

    max_num_epochs = 70

    for makedir in [modeldir, resultsdir, cashedir, savefeaturesdir]:
        if not os.path.exists(makedir):
            os.makedirs(makedir)


    with open(os.path.join(datadir, "train.json"), encoding="utf8") as json_file:
        traindict = json.load(json_file)
    with open(os.path.join(datadir, "dev.json"), encoding="utf8") as json_file:
        devdict = json.load(json_file)
    with open(os.path.join(datadir, "test.json"), encoding="utf8") as json_file:
        testdict = json.load(json_file)
    if modal == 'video':
        del devdict['airport-lyon-1095-40158']

        for workdict, dset in zip([traindict, devdict, testdict], ["train", "dev", "test"]):
            for i in list(workdict.keys()):
                videodir = os.path.join(vfeatdir, dset, i + '.pt')
                workdict[i]['videodir'] = videodir
            if dset == 'train':
                traindict = workdict
            elif dset == 'dev':
                devdict = workdict
            elif dset == "test":
                testdict = workdict

    traindict = {k: traindict[k] for k in random.sample(list(traindict), 20)}
    devdict = {k: devdict[k] for k in random.sample(list(devdict), 20)}
    #testdict = {k: testdict[k] for k in random.sample(list(testdict), 20)}

    if 'audio' in modal:
        BS = 8
    elif 'video' in modal:
        BS = 4

    config = {
            "NWORKER": 0,
            "device": device,
            "lr": 2e-5,
            "batch_size": BS,
            "modeldir": modeldir,
            "resultsdir": resultsdir,
            "savefeaturesdir": savefeaturesdir,
            "epochs": max_num_epochs,  # tune.choice([3, 5, 10, 15])
            "traindict": traindict,
            "devdict": devdict,
            "testdict": testdict,
            "modal": modal,
            "cashedir": cashedir
        }
    training(config)






