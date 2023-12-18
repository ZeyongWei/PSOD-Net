import argparse
import os
from datautils import PCSOD
import torch
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
import numpy as np
import open3d as o3d

import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR

sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--data_root', type=str, default='./data/', help='root path of data')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=4096, help='point number [default: 4096]')
    parser.add_argument('--log_dir', type=str, default='ptsod', help='experiment root')
    parser.add_argument('--num_votes', type=int, default=3,
                        help='aggregate segmentation scores with voting [default: 5]')
    parser.add_argument('--test_batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'log/' + args.log_dir

    visual_dir = experiment_dir + '/visual/'
    visual_dir = Path(visual_dir)
    visual_dir.mkdir(exist_ok=True)

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = args.data_root
    NUM_POINT = args.num_point  #4096
    NUM_CLASSES = 2
    NUM_VOTE = args.num_votes  #3
    TEST_BATCH_SIZE = args.test_batch_size  #32

    print("start loading test data ...")
    TEST_DATASET = PCSOD(split='test', data_root=root, num_point=NUM_POINT, transform=None)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=1, shuffle=False, num_workers=10,
                                                 pin_memory=True, drop_last=True)
    log_string("The number of test data is: %d" % len(TEST_DATASET))


    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()
	
    with torch.no_grad():
        num_batches = len(testDataLoader)
        sum_mae = 0
        print('---- Begin evaluation ----')
        for i, (points, target, scene_name) in tqdm(enumerate(testDataLoader), total=len(testDataLoader),
                                                    smoothing=0.9):
            NUM_POINT_ALL = target.shape[1]
            NUM_ADD = NUM_POINT - (NUM_POINT_ALL % NUM_POINT)
            NUM_ADD = NUM_ADD if NUM_ADD < NUM_POINT else 0
            NUM_BATCH = (NUM_ADD + NUM_POINT_ALL) // NUM_POINT

            target = target.cuda().squeeze()
            final_pred = torch.zeros(NUM_POINT_ALL).cuda()
            for _ in range(NUM_VOTE):
                idx_shuffle = torch.randperm(NUM_POINT_ALL)
                idx_shuffle = torch.cat([idx_shuffle, torch.arange(NUM_ADD)], axis=0)  # NUM_POINT_ALL + NUM_ADD
                points_shuffle = points[:, idx_shuffle, :]
                points_shuffle = points_shuffle.float().cuda()
                points_shuffle = torch.cat(torch.chunk(points_shuffle, points_shuffle.shape[1] // NUM_POINT, dim=1),
                                           dim=0)  # B*NUM_BATCH, NUM_POINT, 3
                points_shuffle = points_shuffle.transpose(2, 1)  # B*NUM_BATCH, 3, NUM_POINT
                NUM_TEST = NUM_BATCH // TEST_BATCH_SIZE if NUM_BATCH % TEST_BATCH_SIZE == 0 else NUM_BATCH // TEST_BATCH_SIZE + 1
                #print(points_shuffle.shape,NUM_BATCH,TEST_BATCH_SIZE)
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

            points = points.squeeze().cuda()
            visual_path = str(visual_dir / scene_name[0])
            pc_vector = o3d.geometry.PointCloud()
            pc_vector_points = points[:, :3].cpu().numpy()
            pc_vector.points = o3d.utility.Vector3dVector(pc_vector_points)

            pc_vector_colors = final_pred.unsqueeze(1).cpu().numpy()
            pc_vector_colors = np.concatenate((pc_vector_colors, np.zeros_like(pc_vector_colors), np.zeros_like(pc_vector_colors)), axis=1)
            pc_vector.colors = o3d.utility.Vector3dVector(pc_vector_colors)

            o3d.io.write_point_cloud(visual_path, pc_vector)

            sum_mae += torch.mean(torch.abs(final_pred - target))
        sum_mae /= num_batches
        log_string('Eval score of mae: %f' % sum_mae)
if __name__ == '__main__':
    args = parse_args()
    main(args)
