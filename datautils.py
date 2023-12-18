import os
import numpy as np
import open3d as o3d
from tqdm import tqdm
from torch.utils.data import Dataset


class PCSOD(Dataset):
    def __init__(self, split='train', data_root='./data/', num_point=4096, transform=None):
        super(PCSOD, self).__init__()
        self.split = split
        self.num_point = num_point
        self.transform = transform
        self.data_root = data_root

        self.scene_coord_min, self.scene_coord_max = [], []
        self.num_point_all = []
        self.scenes_split = []
        labelweights = np.zeros(2)

        print('Computing weights for cross-entropy loss...')
        for split_dir in ['train', 'test']:
            scene_names = os.listdir(os.path.join(self.data_root, split_dir, 'gt'))#[:256]
            #print(scene_names)
            for scene_name in tqdm(scene_names, total=len(scene_names)):
                scene_path = os.path.join(self.data_root, split_dir, 'gt', scene_name)
                scene_data = o3d.io.read_point_cloud(scene_path)
                points, labels = np.array(scene_data.points), np.array(scene_data.colors)[..., 0]
                labels[labels != 0] = 1
                tmp, _ = np.histogram(labels, range(3))
                labelweights += tmp
                if self.split == split_dir:
                    self.scenes_split.append(scene_name)
                    coord_min, coord_max = np.amin(points, axis=0), np.amax(points, axis=0)
                    self.scene_coord_min.append(coord_min), self.scene_coord_max.append(coord_max)
                    self.num_point_all.append(labels.size)
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
        print("Totally {} samples in {} set.".format(len(self.scenes_split), self.split))

    def __getitem__(self, item):
        scene_name = self.scenes_split[item]
        scene_path = os.path.join(self.data_root, self.split, 'point', scene_name)
        label_path = os.path.join(self.data_root, self.split, 'gt', scene_name)
        scene_data = o3d.io.read_point_cloud(scene_path)
        scene_label = o3d.io.read_point_cloud(label_path)
        points, labels = np.concatenate(
            (np.array(scene_data.points, np.float32), np.array(scene_data.colors, np.float32)), axis=1), np.array(
            scene_label.colors, np.float32)[..., 0]

        if self.split == 'train':
            idx_shuffle = np.arange(self.num_point_all[item])
            np.random.shuffle(idx_shuffle)
            points, labels = points[idx_shuffle, :], labels[idx_shuffle]
            points, labels = points[:self.num_point, :], labels[:self.num_point]
            current_points = np.zeros((self.num_point, 9))  # num_point * 9
        else:
            current_points = np.zeros((self.num_point_all[item], 9))
        # normalize
        current_points[:, 6] = (points[:, 0] - self.scene_coord_min[item][0]) / self.scene_coord_max[item][0]
        current_points[:, 7] = (points[:, 1] - self.scene_coord_min[item][1]) / self.scene_coord_max[item][1]
        current_points[:, 8] = (points[:, 2] - self.scene_coord_min[item][2]) / self.scene_coord_max[item][2]

        points[:, 0] = points[:, 0] - self.scene_coord_max[item][0] / 2
        points[:, 1] = points[:, 1] - self.scene_coord_max[item][1] / 2

        current_points[:, 0:6] = points
        labels[labels != 0] = 1

        if self.transform is not None:
            current_points, labels = self.transform(current_points, labels)

        if self.split == 'train':
            return current_points, labels
        else:

            return current_points, labels, scene_name

    def __len__(self):
        return len(self.scenes_split)


if __name__ == '__main__':
    data, label = next(iter(PCSOD()))
    print(data.shape)
    print(label.shape)
