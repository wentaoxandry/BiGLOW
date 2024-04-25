import json
import argparse
from tqdm import tqdm
from kaldiio import WriteHelper
from model import *
from utils import *
from transformers import BertTokenizer, AutoTokenizer, get_linear_schedule_with_warmup, AdamW, ViTFeatureExtractor
from torchmetrics.classification import MulticlassF1Score
import random
import logging

SEED=666
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def training(config, dataset=None):
    modal = config["modal"]
    resultsdir = config["resultsdir"]
    modeldir = config["modeldir"]
    logging.basicConfig(filename=os.path.join(modeldir, 'train.log'), level=logging.INFO)

    evalacc_best = 0
    early_wait = 5
    run_wait = 1
    continuescore = 0
    stop_counter = 0
    f1score = MulticlassF1Score(num_classes=6, average='weighted')
    criterion = torch.nn.CrossEntropyLoss()
    if 'Title' in modal:
        model = TitleBERT(odim=6, MODEL=config['MODEL'], cachedir=config["cachedir"])
        padding = pad_title
    elif 'Image' in modal:
        model = ImageViT(odim=6, MODEL=config['MODEL'], cachedir=config["cachedir"])            
        padding = pad_image_LRP

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
    model = model.to(config["device"])
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'The model has {trainable_parameters} trainable parameters')
    logging.info(f'Train set contains {len(dataset.train_dataset)} samples')
    logging.info(f'Dev set contains {len(dataset.dev_dataset)} samples')
    logging.info(f'Test set contains {len(dataset.test_dataset)} samples')



    optimizer = AdamW(model.parameters(),
                                  lr=config["lr"], no_deprecation_warning=True
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
    data_loader_test = torch.utils.data.DataLoader(dataset.test_dataset, shuffle=True,
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
            if 'Title' in modal:
                token_ids = data[0].to(config["device"])
                mask = data[1].to(config["device"])
                label = data[2].squeeze(-1).to(config["device"])
                filename = data[3]
                input = [token_ids, mask]
                optimizer.zero_grad()
                outputs, _ = model(input)
            elif 'Image' in modal:
                image = data[0].to(config["device"])
                label = data[1].squeeze(-1).to(config["device"])
                filename = data[2]
                input = image
                optimizer.zero_grad()
                outputs = model(input)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            scheduler.step()
            tr_loss += loss.item()
            nb_tr_steps += 1
            prob = torch.softmax(outputs, dim=-1)
            predicted = torch.argmax(torch.softmax(outputs, dim=-1), dim=-1)
            trainprob.append(prob.cpu())
            trainpredict.extend(predicted.cpu().detach().tolist())
            trainlabel.extend(label.cpu().data.numpy().tolist())
        trainlabel = torch.LongTensor(trainlabel)
        trainpredict = torch.LongTensor(trainpredict)
        trainallf1score = f1score(trainpredict, trainlabel).cpu().data.numpy()
        trainallaccscore = (sum(trainpredict == trainlabel) / trainlabel.size(0)).cpu().data.numpy()
        trainallf1score = float(trainallf1score)
        trainallaccscore = float(trainallaccscore)
        logging.info(f'Epoch {epoch} training f1 score is {trainallf1score}, accuracy score is {trainallaccscore}.')

        # Validation loss
        torch.cuda.empty_cache()
        evallossvec = []
        evalpred = []
        evallabel = []
        model.eval()
        outpre = {}
        total = 0
        for i, data in enumerate(tqdm(data_loader_dev), 0):
            with torch.no_grad():
                if 'Title' in modal:
                    token_ids = data[0].to(config["device"])
                    mask = data[1].to(config["device"])
                    labels = data[2].squeeze(-1).to(config["device"])
                    filename = data[3]
                    input = [token_ids, mask]
                    outputs, _ = model(input)
                elif 'Image' in modal:
                    image = data[0].to(config["device"])
                    labels = data[1].squeeze(-1).to(config["device"])
                    filename = data[2]
                    input = image
                    outputs = model(input)

                
                dev_loss = criterion(outputs, labels)
                evallossvec.append(dev_loss.cpu().data.numpy())
                predicted = torch.argmax(torch.softmax(outputs, dim=-1), dim=-1)
                prob = torch.softmax(outputs, dim=-1).cpu().detach().tolist()
                evallabel.extend(labels.cpu().data.numpy().tolist())
                evalpred.extend(predicted.cpu().detach().tolist())
                total += labels.size(0)
                for i in range(len(filename)):
                    outpre.update({filename[i]: {}})
                    outpre[filename[i]].update({'label': int(labels[i].cpu().detach())})
                    outpre[filename[i]].update({'predict': int(predicted[i].cpu().detach())})
                    outpre[filename[i]].update({'prob': prob[i]})

        evalpred = torch.LongTensor(evalpred)
        evallabel = torch.LongTensor(evallabel)
        evalallf1score = f1score(evalpred, evallabel).cpu().data.numpy()
        evalallaccscore = (sum(evalpred == evallabel) / evallabel.size(0)).cpu().data.numpy()
        evalallf1score = float(evalallf1score)
        evalallaccscore = float(evalallaccscore)

        evallossmean = np.mean(np.array(evallossvec))

        logging.info(f'Epoch {epoch} evaluation f1 score is {evalallf1score}, accuracy score is {evalallaccscore}, loss is {evallossmean}.')

        for param_group in optimizer.param_groups:
            currentlr = param_group['lr']
        OUTPUT_DIR = os.path.join(modeldir, 'bestmodel.pkl')

        with open(os.path.join(resultsdir, str(epoch) + '_' + str(evallossmean) + '_' + str(
                              currentlr) + '_' + str(trainallf1score)[:6] + '_' + str(trainallaccscore)[:6] + '_' + str(
                              evalallf1score)[:6]+ '_' + str(
                              evalallaccscore)[:6] + ".json"), 'w', encoding='utf-8') as f:
            json.dump(outpre, f, ensure_ascii=False, indent=4)


        torch.cuda.empty_cache()
        if evalallaccscore <= evalacc_best:
            stop_counter = stop_counter + 1
            print('no improvement')
            continuescore = 0
        else:
            print('new score')
            evalacc_best = evalallaccscore
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


    model = torch.load(os.path.join(modeldir,'bestmodel.pkl'), map_location=config["device"])
    testpred = []
    testlabel = []
    model.eval()
    outpre = {}
    total = 0
    for i, data in enumerate(tqdm(data_loader_test), 0):
        with torch.no_grad():
            if 'Title' in modal:
                token_ids = data[0].to(config["device"])
                mask = data[1].to(config["device"])
                labels = data[2].squeeze(-1).to(config["device"])
                filename = data[3]
                input = [token_ids, mask]
                outputs, _ = model(input)
            elif 'Image' in modal:
                image = data[0].to(config["device"])
                labels = data[1].squeeze(-1).to(config["device"])
                filename = data[2]
                input = image
                outputs = model(input)
            

            predicted = torch.argmax(torch.softmax(outputs, dim=-1), dim=-1)
            prob = torch.softmax(outputs, dim=-1).cpu().detach().tolist()
            testlabel.extend(labels.cpu().data.numpy().tolist())
            testpred.extend(predicted.cpu().detach().tolist())
            total += labels.size(0)
            for i in range(len(filename)):
                outpre.update({filename[i]: {}})
                outpre[filename[i]].update({'label': int(labels[i].cpu().detach())})
                outpre[filename[i]].update({'predict': int(predicted[i].cpu().detach())})
                outpre[filename[i]].update({'prob': prob[i]})

    testpred = torch.LongTensor(testpred)
    testlabel = torch.LongTensor(testlabel)
    testallf1score = f1score(testpred, testlabel).cpu().data.numpy()
    testallaccscore = (sum(testpred == testlabel) / testlabel.size(0)).cpu().data.numpy()
    testallf1score = float(testallf1score)
    testallaccscore = float(testallaccscore)

    with open(os.path.join(resultsdir,
                           'testf1_' + str(testallf1score)[:6] + '_testacc_' + str(testallaccscore)[:6] + ".json"), 'w',
              encoding='utf-8') as f:
        json.dump(outpre, f, ensure_ascii=False, indent=4)

    for dset, dataloader in zip(['dev', 'train', 'test'], [data_loader_dev, data_loader_train, data_loader_test]):
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
                if 'Title' in modal:
                    token_ids = data[0].to(config["device"])
                    mask = data[1].to(config["device"])
                    labels = data[2].squeeze(-1).to(config["device"])
                    filename = data[3]
                    input = [token_ids, mask]
                    outputs, feats = model(input)
                elif 'Image' in modal:
                    image = data[0].to(config["device"])
                    labels = data[1].squeeze(-1).to(config["device"])
                    filename = data[2]
                    input = image
                    outputs, feats = model(input, ifreturnfeats=True)

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
    parser.add_argument('--datasdir', default='./../../Dataset', type=str, help='Dir saves the meta information')
    parser.add_argument('--modal', default='Image', type=str, help='which data modality. It could be Text and Image')
    parser.add_argument('--savedir', default='./../../trained', type=str, help='Dir to save trained model and results')
    parser.add_argument('--lr', default=5e-6, type=float, help='The learning rate')
    parser.add_argument('--BS', default=32, type=float, help='The batch size')
    parser.add_argument('--cachedir', default='./../../CACHE', type=str, help='Cache dir to save the downloaded pretrained transformer model')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = get_args()
    print("Arguments:")
    [print(key, val) for key, val in vars(args).items()]


    datasetdir = args.datasdir
    modal = args.modal
    modal = modal
    lr = float(args.lr)
    BS = int(args.BS)
    savedir = args.savedir
    cachedir = args.cachedir

    savefeaturesdir = os.path.join(datasetdir, 'data', modal)

    if torch.cuda.is_available() == True:
        device = 'cuda'
    else:
        device = 'cpu'

    modeldir = os.path.join(savedir, 'Unimodal', modal, 'model')
    resultsdir = os.path.join(savedir, 'Unimodal', modal, 'results')

    max_num_epochs = 15
    max_uttlen = 500

    for makedir in [modeldir, resultsdir, savefeaturesdir]:
        if not os.path.exists(makedir):
            os.makedirs(makedir)


    with open(os.path.join(datasetdir, "train.json"), encoding="utf8") as json_file:
        traindict = json.load(json_file)
    with open(os.path.join(datasetdir, "val.json"), encoding="utf8") as json_file:
        devdict = json.load(json_file)
    with open(os.path.join(datasetdir, "test.json"), encoding="utf8") as json_file:
        testdict = json.load(json_file)

    '''traindict = {k: traindict[k] for k in random.sample(list(traindict), 100)}
    devdict = {k: devdict[k] for k in random.sample(list(devdict), 100)}
    testdict = {k: testdict[k] for k in random.sample(list(testdict), 100)}'''

    if 'Title' in modal:
        MODEL = "bert-large-uncased"
        tokenizer = AutoTokenizer.from_pretrained(MODEL, cache_dir=cachedir)
        dataset = Titledatasetclass(train_file=traindict,
                               dev_file=devdict,
                               test_file=testdict,
                                tokenizer=tokenizer,
                               max_len=max_uttlen)
    elif 'Image' in modal:
        MODEL = 'vit_large_patch16_384.augreg_in21k_ft_in1k'
        dataset = Imagedatasetclass_LRP(train_file=traindict,
                                    dev_file=devdict,
                                    test_file=testdict)

    config = {
        "NWORKER": 0,
        "device": device,
        "lr": lr,  # tune.choice([5e-2, 5e-3, 5e-4, 5e-5]),
        "batch_size": BS,
        "MODEL": MODEL,
        "modal": modal,
        "savefeaturesdir": savefeaturesdir,
        "cachedir": cachedir,
        "modeldir": modeldir,
        "resultsdir": resultsdir,
        "epochs": max_num_epochs  # tune.choice([3, 5, 10, 15])
    }
    training(config, dataset=dataset)



