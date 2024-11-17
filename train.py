import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime
from pprint import pprint
import pandas as pd
import torch.optim
from torchvision import transforms
from torch.optim import Adam
from torch.utils.data import DataLoader
import configs
from hashing.utils import calculate_accuracy, get_hamm_dist, calculate_mAP
from networks.loss import RelaHashLoss
from networks.resnet import ResNet
from networks.ca_net import CANet
from networks.label_net import LabelNet
from utils import io
from utils.misc import AverageMeter, Timer
from tqdm import tqdm
from networks.triplet_loss import TripletLoss
from utils.attention_zoom import *
from utils.datasets import MLRSs
from networks.MSLOSS import MultiSimilarityLoss
from loguru import logger
def smooth_CE(logits, labels, peak):
    # logits - [batch, num_cls]
    # label - [batch]
    A = ((labels == 0).sum(dim=1) == labels.shape[1])
    labels[A == True] = 1
    batch, num_cls = logits.size()
    label_logits = labels
    smooth_label = torch.ones(logits.size()) * (1 - peak) / (num_cls - 1)
    smooth_label[label_logits == 1] = peak

    logits = F.log_softmax(logits, -1)
    ce = torch.mul(logits, smooth_label.to(logits.device))
    loss = torch.mean(-torch.sum(ce, -1))  # batch average

    return loss
def train_hashing(optimizer, model, train_loader, loss_param,model2,ep):
    model.train()
    device = loss_param['device']
    nclass = loss_param['arch_kwargs']['nclass']
    nbit = loss_param['arch_kwargs']['nbit']
    meters = defaultdict(AverageMeter)
    total_timer = Timer()
    timer = Timer()
    total_timer.tick()
    # LNet = LabelNet(code_len=nbit,label_dim=nclass)
    # Triplet = TripletLoss()
    # 多相似性损失
    Triplet = MultiSimilarityLoss()
    train_loss=[]
    pbar = tqdm(train_loader, desc='Training', ascii=True, bar_format='{l_bar}{bar:10}{r_bar}')
    for i, (data, labels) in enumerate(pbar):
        timer.tick()
        optimizer.zero_grad()

        data, labels = data.to(device), labels.to(device)  # hen trainning MLRS delete this line
        u ,logits = model(data,SGE=True)

        loss2 = Triplet( u , labels.float() )
        loss1 = smooth_CE(logits,labels,peak=0.9)
        loss = loss2 + 0.1 * loss1
        loss.backward()
        optimizer.step()
        timer.toc()
        total_timer.toc()

        # store results
        meters['loss_total'].update(loss.item(), data.size(0))

        meters['time'].update(timer.total)

        pbar.set_postfix({'Train_loss': meters['loss_total'].avg,
                          'A(CE)': meters['acc'].avg,
                          'A(CB)': meters['cbacc'].avg})

    print()
    total_timer.toc()

    # train_loss.append(loss.item())
    # list1 = [loss.cpu().data.numpy()]
    # data = pd.DataFrame([list1])
    # data.to_csv('/home/admin01/桌面/CAN/loss/DFC_16_loss.csv', mode='a', header=False,
    #             index=False)
    meters['total_time'].update(total_timer.total)
    return meters


def test_hashing(model, test_loader, loss_param, return_codes=False):
    model.eval()
    device = loss_param['device']
    meters = defaultdict(AverageMeter)
    total_timer = Timer()
    timer = Timer()
    nclass = loss_param['arch_kwargs']['nclass']
    nbit = loss_param['arch_kwargs']['nbit']
    total_timer.tick()

    ret_codes = []
    ret_labels = []

    criterion = RelaHashLoss(**loss_param)
    Triplet = TripletLoss()
    pbar = tqdm(test_loader, desc='Test', ascii=True, bar_format='{l_bar}{bar:10}{r_bar}')
    for i, (data, labels) in enumerate(pbar):
        timer.tick()

        with torch.no_grad():
            data, labels = data.to(device), labels.to(device)  # hen trainning MLRS delete this line

            codes, _ = model(data,SGE=True)
            # codes= model(data)


            if return_codes:
                ret_codes.append(codes)
                ret_labels.append(labels)

        timer.toc()
        total_timer.toc()

        # store results
        meters['loss_total'].update(0, data.size(0))

        meters['time'].update(timer.total)

        pbar.set_postfix({'Eval_loss': meters['loss_total'].avg,
                          'A(CE)': meters['acc'].avg,
                          'A(CB)': meters['cbacc'].avg})

    print()
    meters['total_time'].update(total_timer.total)

    if return_codes:
        res = {
            'codes': torch.cat(ret_codes),
            'labels': torch.cat(ret_labels)
        }
        return meters, res

    return meters


def prepare_dataloader(config):
    logging.info('Creating Datasets')
    if config['dataset'] == 'MLRS':
        MLRSs.init('./data/MLRS/', 1000, 5000)
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        trainset = MLRSs('./data/', 'train', transform=transform)
        testset = MLRSs('./data/', 'query', transform=transform)
        database = MLRSs('./data/', 'retrieval', transform=transform)
        train_loader = DataLoader(trainset, config['batch_size'])
        test_loader = DataLoader(testset, config['batch_size'], shuffle=False, drop_last=False)
        db_loader = DataLoader(database, config['batch_size'], shuffle=False, drop_last=False)
        return train_loader, test_loader, db_loader
    # num_train, num_test, num_database = len(trainset), len(testset), len(database)
    else:
        train_dataset = configs.dataset(config, filename='train.txt', transform_mode='train')

        separate_multiclass = config['dataset_kwargs'].get('separate_multiclass', False)
        config['dataset_kwargs']['separate_multiclass'] = False
        test_dataset = configs.dataset(config, filename='test.txt', transform_mode='test')
        db_dataset = configs.dataset(config, filename='database.txt', transform_mode='test')
        config['dataset_kwargs']['separate_multiclass'] = separate_multiclass  # during mAP, no need to separate

        logging.info(f'Number of DB data: {len(db_dataset)}')
        logging.info(f'Number of Train data: {len(train_dataset)}')

        print(train_dataset)

        train_loader = configs.dataloader(train_dataset, config['batch_size'])
        test_loader = configs.dataloader(test_dataset, config['batch_size'], shuffle=False, drop_last=False)
        db_loader = configs.dataloader(db_dataset, config['batch_size'], shuffle=False, drop_last=False)
        return train_loader, test_loader, db_loader


def main(config):
    Best_map = 0
    device = torch.device(config.get('device', 'cuda:0'))

    io.init_save_queue()

    start_time = time.time()
    configs.seeding(config['seed'])

    logdir = config['logdir']
    assert logdir != '', 'please input logdir'

    pprint(config)

    if config['wandb_enable']:
        import wandb
        ## initiaze wandb ##
        wandb_dir = logdir
        wandb.init(project="relahash", config=config, dir=wandb_dir)
        # wandb run name
        wandb.run.name = logdir.split('logs/')[1]

    os.makedirs(f'{logdir}/models', exist_ok=True)
    os.makedirs(f'{logdir}/optims', exist_ok=True)
    os.makedirs(f'{logdir}/outputs', exist_ok=True)
    json.dump(config, open(f'{logdir}/config.json', 'w+'), indent=4, sort_keys=True)

    nclass = config['arch_kwargs']['nclass']
    nbit = config['arch_kwargs']['nbit']

    train_loader, test_loader, db_loader = prepare_dataloader(config)
    # model = RelaHash(**config['arch_kwargs'])
    # model = ResNet(nbit)
    model = CANet(bit=nbit,classes=nclass)
    model2 = LabelNet(code_len=nbit,label_dim=nclass).to(device)
    model.to(device)
    print(model)
    # 建立日志文件（Create log file）
    logger.add('logs/{time}' + config["dataset"] +'_'+ str(nbit) + '.log',
               rotation='50 MB', level='DEBUG')
    logging.info(f'Total Bit: {nbit}')
    # centroids = model.get_centroids()
    # io.fast_save(centroids, f'{logdir}/outputs/centroids.pth')

    if config['wandb_enable']:
        wandb.watch(model)

    optimizer = torch.optim.AdamW([
        {'params': model.get_backbone_param(), 'lr': config['optim_kwargs']['lr'] },
        # {'params': model.get_sge_param(), 'lr': 5e-3 },
        {'params': model2.parameters(), 'lr': 1e-5 },
    ],
        lr=config['optim_kwargs']['lr'],
        betas=config['optim_kwargs'].get('betas', (0.9, 0.999)),
        weight_decay=config['optim_kwargs'].get('weight_decay', 0))
    scheduler = configs.scheduler(config, optimizer)

    train_history = []
    test_history = []

    loss_param = config.copy()
    loss_param.update({'device': device})

    best = 0
    curr_metric = 0

    nepochs = config['epochs']
    neval = config['eval_interval']

    logging.info('Training Start')

    for ep in range(nepochs):
        logging.info(f'Epoch [{ep + 1}/{nepochs}]')
        res = {'ep': ep + 1}
        model2.set_alpha(ep)
        train_meters = train_hashing(optimizer, model, train_loader, loss_param,model2,ep)

        scheduler.step()

        for key in train_meters: res['train_' + key] = train_meters[key].avg
        train_history.append(res)
        # train_outputs.append(train_out)
        if config['wandb_enable']:
            wandb_train = res.copy()
            wandb_train.pop("ep")
            wandb.log(wandb_train, step=res['ep'])

        modelsd = model.state_dict()
        optimsd = optimizer.state_dict()

        eval_now = (ep + 1) == nepochs or (neval != 0 and (ep + 1) % neval == 0)
        if eval_now:
            res = {'ep': ep + 1}

            test_meters, test_out = test_hashing(model, test_loader, loss_param, True)
            db_meters, db_out = test_hashing(model, db_loader, loss_param, True)

            for key in test_meters: res['test_' + key] = test_meters[key].avg
            for key in db_meters: res['db_' + key] = db_meters[key].avg

            res['mAP'] = calculate_mAP(db_out['codes'], db_out['labels'],
                                       test_out['codes'], test_out['labels'],
                                       loss_param['R'], ep=ep)
            logging.info(f'mAP: {res["mAP"]:.6f}')


            print('mAP : %.6f', res['mAP'])
            if res['mAP'] > Best_map:
                Best_map = res['mAP']


            print(f'Best mAP : {Best_map}')

            curr_metric = res['mAP']
            test_history.append(res)
            # test_outputs.append(outs)

            if config['wandb_enable']:
                wandb_test = res.copy()
                wandb_test.pop("ep")
                wandb.log(wandb_test, step=res['ep'])
            if best < curr_metric:
                best = curr_metric
                io.fast_save(modelsd, f'{logdir}/models/best.pth')
                io.fast_save(optimsd, f'{logdir}/optims/best.pth')
                if config['wandb_enable']:
                    wandb.run.summary["best_map"] = best

        json.dump(train_history, open(f'{logdir}/train_history.json', 'w+'), indent=True, sort_keys=True)
        # io.fast_save(train_outputs, f'{logdir}/outputs/train_last.pth')

        if len(test_history) != 0:
            json.dump(test_history, open(f'{logdir}/test_history.json', 'w+'), indent=True, sort_keys=True)
            # io.fast_save(test_outputs, f'{logdir}/outputs/test_last.pth')

        save_now = config['save_interval'] != 0 and (ep + 1) % config['save_interval'] == 0
        # if save_now:
        #     io.fast_save(modelsd, f'{logdir}/models/ep{ep + 1}.pth')
        #     io.fast_save(optimsd, f'{logdir}/optims/ep{ep + 1}.pth')
        # io.fast_save(train_outputs, f'{logdir}/outputs/train_ep{ep + 1}.pth')

        # 模型参数保存
        # if best < curr_metric:
        #     best = curr_metric
        #     io.fast_save(modelsd, f'{logdir}/models/best.pth')
        #     save_model = '/home/admin01/桌面/CAN/model_para/UCMD/'
        #     os.makedirs(save_model, exist_ok=True)
        #     torch.save(modelsd, os.path.join(save_model, "model-" + str(ep) + ".pth"))
        #

    modelsd = model.state_dict()

    io.fast_save(modelsd, f'{logdir}/models/last.pth')

    # io.fast_save(optimsd, f'{logdir}/optims/last.pth')
    total_time = time.time() - start_time
    io.join_save_queue()
    logging.info(f'Training End at {datetime.today().strftime("%Y-%m-%d %H:%M:%S")}')
    logging.info(f'Total time used: {total_time / (60 * 60):.2f} hours')
    logging.info(f'Best mAP: {best:.6f}')
    print(f'Epoch : {ep}  Best mAP: {best:.6f}')
    logging.info(f'Done: {logdir}')
    logger.info(f'Training End at {datetime.today().strftime("%Y-%m-%d %H:%M:%S")}')
    logger.info(f'Total time used: {total_time / (60 * 60):.2f} hours')
    logger.info(f'Best mAP: {best:.6f}')
    logger.info(f'Done: {logdir}')



    return logdir
