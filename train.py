import argparse
import os
from datautils import PCSOD
from utils import inplace_relu, weights_init, bn_momentum_adjust
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import provider
import numpy as np
import time

import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR

sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='ptsod', help='model name [default: ptsod]')
    parser.add_argument('--data_root', type=str, default='./data/', help='root path of data')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 64]')
    parser.add_argument('--test_batch_size', type=int, default=32, help='Batch Size during evaluation [default: 32]')
    parser.add_argument('--epoch', default=3000, type=int, help='Epoch to run [default: 3000]')
    parser.add_argument('--num_votes', type=int, default=3, help='times of votes during evaluation [default: 3]')
    parser.add_argument('--learning_rate', default=0.0005, type=float, help='Initial learning rate [default: 0.0005]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default='ptsod', help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int, default=4096, help='Point Number [default: 4096]')
    parser.add_argument('--step_size', type=int, default=800, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--eval', action='store_true', default=False, help='whether to eval[default: False]')

    return parser.parse_args()


def main(args):
    def log_string(string):
        logger.info(string)
        print(string)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DTR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = args.data_root
    NUM_POINT = args.npoint
    NUM_VOTE = args.num_votes
    BATCH_SIZE = args.batch_size
    TEST_BATCH_SIZE = args.test_batch_size
    NUM_CLASSES = 2

    print("start loading training data ...")
    TRAIN_DATASET = PCSOD(split='train', data_root=root, num_point=NUM_POINT, transform=None)

    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=1,
                                                  pin_memory=True, drop_last=True,
                                                  worker_init_fn=lambda x: np.random.seed(x + int(time.time())))
    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    weights = torch.Tensor(TRAIN_DATASET.labelweights).cuda()
    if args.eval:
        print("start loading test data ...")
        TEST_DATASET = PCSOD(split='test', data_root=root, num_point=NUM_POINT, transform=None)
        testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=1, shuffle=False, num_workers=10,
                                                     pin_memory=True, drop_last=True)
        log_string("The number of test data is: %d" % len(TEST_DATASET))

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('train.py', str(experiment_dir))
    #classifier = MODEL.get_model(NUM_CLASSES).cuda()
    classifier = MODEL.get_model(num_classes=NUM_CLASSES).cuda()
    criterion = MODEL.get_loss().cuda()
    #criterion = Loss().cuda()
    classifier.apply(inplace_relu)

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrained model...')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    global_epoch = 0
    best_mae = float('inf')
    best_train_mae = float('inf')

    for epoch in range(start_epoch, args.epoch):
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))

        num_batches = len(trainDataLoader)
        loss_sum = 0
        classifier = classifier.train()
        mae = 0

        for i, (points, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            #print('target shape = ', target.shape)
            #print(target[0, :256])
            optimizer.zero_grad()

            points = points.data.numpy()
            points[:, :, :3] = provider.rotate_point_cloud_z(points[:, :, :3])
            points = torch.Tensor(points)
            points, target = points.float().cuda(), target.long().cuda()
            points = points.transpose(2, 1)

            seg_pred, seg_o = classifier(points)
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

            target = target.view(-1, 1)[:, 0]
            loss = criterion(seg_pred, target, weights)
            loss.backward()
            optimizer.step()

            target = target.view(-1, 1)[:, 0]
            seg_o = F.softmax(seg_o, dim=-1)[:, :, 1].view(-1, 1)[:, 0]

            mae += torch.mean(torch.abs(target - seg_o))
            loss_sum += loss
        mae = mae / num_batches
        log_string('Training mean loss: %f' % (loss_sum / num_batches))
        log_string('Training MAE accuracy: %f' % mae)

        if mae < best_train_mae:
            best_train_mae = mae
            savepath = str(checkpoints_dir) + '/best_model.pth'
            state = {
                'epoch': epoch,
                'mae': mae,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
        log_string('Best training MAE accuracy: %f' % best_train_mae)

        if epoch % 100 == 0:
            log_string('Saving model...')
            savepath = str(checkpoints_dir) + '/model_' + str(epoch)+ '.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)

        if args.eval:
            with torch.no_grad():
                num_batches = len(testDataLoader)
                mae = 0

                log_string('---- EPOCH %04d EVALUATION ----' % (global_epoch + 1))

                classifier = classifier.eval()
                for i, (points, target, scene_name) in tqdm(enumerate(testDataLoader), total=len(testDataLoader),
                                                            smoothing=0.9):

                    NUM_POINT_ALL = target.shape[1]
                    NUM_ADD = NUM_POINT - (NUM_POINT_ALL % NUM_POINT) if NUM_POINT - (
                            NUM_POINT_ALL % NUM_POINT) < NUM_POINT else 0
                    NUM_BATCH = (NUM_ADD + NUM_POINT_ALL) // NUM_POINT
                    target = target.cuda()
                    final_pred = torch.zeros(NUM_POINT_ALL).cuda()
                    for _ in range(NUM_VOTE):
                        idx_shuffle = torch.randperm(NUM_POINT_ALL)
                        idx_shuffle = torch.cat([idx_shuffle, torch.arange(NUM_ADD)], axis=0)
                        points_shuffle = points[:, idx_shuffle, :]
                        points_shuffle = points_shuffle.float().cuda()
                        points_shuffle = torch.cat(torch.chunk(points_shuffle, NUM_BATCH, dim=1),
                                                   dim=0)
                        points_shuffle = points_shuffle.transpose(2, 1)
                        NUM_TEST = NUM_BATCH // TEST_BATCH_SIZE if NUM_BATCH % TEST_BATCH_SIZE == 0 else NUM_BATCH // TEST_BATCH_SIZE + 1
                        for i in range(NUM_TEST):
                            _, seg_o = classifier(
                                points_shuffle[i * TEST_BATCH_SIZE: min((i + 1) * TEST_BATCH_SIZE, NUM_BATCH), :, :])
                            if i == 0:
                                seg_pred_shuffle = seg_o
                            else:
                                seg_pred_shuffle = torch.cat([seg_pred_shuffle, seg_o], dim=0)

                        seg_pred_shuffle = torch.cat(torch.chunk(seg_pred_shuffle, seg_pred_shuffle.shape[0], dim=0),
                                                     dim=1).squeeze()
                        seg_pred_shuffle = F.softmax(seg_pred_shuffle, dim=-1)[:, 1].view(-1, 1)[:NUM_POINT_ALL, 0]
                        final_pred[idx_shuffle[:NUM_POINT_ALL]] += seg_pred_shuffle
                    final_pred /= NUM_VOTE
                    mae += torch.mean(torch.abs(final_pred - target))
                mae /= num_batches
                log_string('Eval MAE accuracy: %f' % mae)

                if mae < best_mae:
                    best_mae = mae
                    logger.info('Save model...')
                    savepath = str(checkpoints_dir) + '/best_model.pth'
                    state = {
                        'epoch': epoch,
                        'mae': mae,
                        'model_state_dict': classifier.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
                    torch.save(state, savepath)
                log_string('Best MAE accuracy: %f' % best_mae)

        global_epoch += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)
