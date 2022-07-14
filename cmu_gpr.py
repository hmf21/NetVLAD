import torch
import torchvision.transforms as transforms
import torch.utils.data as data

from os.path import join, exists, isfile
from scipy.io import loadmat
import numpy as np
from random import randint, random
from collections import namedtuple
from PIL import Image

from sklearn.neighbors import NearestNeighbors
import h5py
import pandas as pd
import glob

# root_dir = 'H:\\Dataset\\CMU_GPR\\UAV_Round2\\Train'
root_dir = '/home/titan/WSc/hmf/CMU_GPR/Round2/'

if not exists(root_dir):
    raise FileNotFoundError('root_dir is hardcoded, please adjust to point to Pittsburth dataset')

train_dir = join(root_dir, 'Train')
val_dir = join(root_dir, 'Val')
save_dir = join(root_dir, 'Test')

dbStruct = namedtuple('dbStruct', ['whichSet', 'dataset',
                                   'dbImage', 'utmDb', 'qImage', 'utmQ', 'numDb', 'numQ',
                                   'posDistThr', 'posDistSqThr', 'nonTrivPosDistSqThr'])


def input_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_whole_training_set(onlyDB=False):
    structFile = join(train_dir, 'gt_matches.csv')
    return WholeDatasetFromStruct(structFile, train_dir,
                                  input_transform=input_transform(),
                                  onlyDB=onlyDB)


def get_whole_val_set():
    structFile = join(val_dir, 'gt_matches.csv')
    return WholeDatasetFromStruct(structFile, val_dir,
                                  input_transform=input_transform())


def get_whole_save_set():
    dataset_dir = save_dir
    return WholeDatasetFromDir(dataset_dir, val_dir,
                                  input_transform=input_transform())


def get_training_query_set(margin=0.1):
    structFile = join(train_dir, 'gt_matches.csv')
    return QueryDatasetFromStruct(structFile, train_dir,
                                  input_transform=input_transform(), margin=margin)


def get_val_query_set():
    structFile = join(val_dir, 'gt_matches.csv')
    return QueryDatasetFromStruct(structFile, val_dir,
                                  input_transform=input_transform())


def parse_dbStruct(csv_dir, is_whole=False):
    whichSet = 'Train'

    dataset = 'CMU_GPR'

    gt_matches_csv = pd.read_csv(csv_dir)

    images = np.array(gt_matches_csv)

    if is_whole:
        dbImage = [join('reference_images', i) for i in images[:, 3]]
        qImage = [join('query_images', i) for i in images[:, 1]]
    else:
        dbImage = []
        for idx, i in enumerate(images[:, 3]):
            if idx % 10 == 0:
                dbImage.append(join('reference_images', i))
        qImage = []
        for idx, i in enumerate(images[:, 1]):
            if idx % 10 == 0:
                qImage.append(join('query_images', i))


    numDb = len(dbImage)    # the number of database is 13783
    numQ = len(qImage)      # same as the number of database

    # utmDb and utmQ represent the actual distance
    # due to the dataset records a UAV flight along with the straight line
    # so we simply use the index use the utm
    route = np.expand_dims(np.linspace(1, numDb, numDb), 1)
    utmDb = np.concatenate((route, route), axis=1)
    utmQ = np.concatenate((route, route), axis=1)

    posDistThr = 1
    posDistSqThr = 2
    nonTrivPosDistSqThr = 2

    return dbStruct(whichSet, dataset, dbImage, utmDb, qImage,
                    utmQ, numDb, numQ, posDistThr,
                    posDistSqThr, nonTrivPosDistSqThr)


def collate_fn(batch):
    """Creates mini-batch tensors from the list of tuples (query, positive, negatives).

    Args:
        data: list of tuple (query, positive, negatives).
            - query: torch tensor of shape (3, h, w).
            - positive: torch tensor of shape (3, h, w).
            - negative: torch tensor of shape (n, 3, h, w).
    Returns:
        query: torch tensor of shape (batch_size, 3, h, w).
        positive: torch tensor of shape (batch_size, 3, h, w).
        negatives: torch tensor of shape (batch_size, n, 3, h, w).
    """

    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0: return None, None, None, None, None

    query, positive, negatives, indices = zip(*batch)

    query = data.dataloader.default_collate(query)
    positive = data.dataloader.default_collate(positive)
    negCounts = data.dataloader.default_collate([x.shape[0] for x in negatives])
    negatives = torch.cat(negatives, 0)
    import itertools
    indices = list(itertools.chain(*indices))

    return query, positive, negatives, negCounts, indices


class WholeDatasetFromStruct(data.Dataset):
    def __init__(self, structFile, parent_dir, input_transform=None, onlyDB=False):
        super().__init__()

        self.input_transform = input_transform

        self.dbStruct = parse_dbStruct(structFile)
        # images of whole dataset
        self.images = [join(parent_dir, dbIm) for dbIm in self.dbStruct.dbImage]
        if not onlyDB:
            self.images += [join(parent_dir, qIm) for qIm in self.dbStruct.qImage]

        self.whichSet = self.dbStruct.whichSet
        self.dataset = self.dbStruct.dataset

        self.positives = None
        self.distances = None

    def __getitem__(self, index):
        img = Image.open(self.images[index])
        img = img.convert('RGB')

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)

    def getPositives(self):
        # positives for evaluation are those within trivial threshold range
        # fit NN to find them, search by radius

        if self.positives is None:
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.dbStruct.utmDb)

            self.distances, self.positives = knn.radius_neighbors(self.dbStruct.utmQ,
                                                                  radius=self.dbStruct.posDistThr)

        return self.positives


class WholeDatasetFromDir(data.Dataset):
    def __init__(self, save_dir, parent_dir, input_transform=None, onlyDB=False):
        super().__init__()

        self.input_transform = input_transform

        self.images = glob.glob(join(save_dir, '*.png'))

        self.whichSet = 'Save'
        self.dataset = 'CMU_GPR'

        self.positives = None
        self.distances = None

    def __getitem__(self, index):
        img = Image.open(self.images[index])
        img = img.convert('RGB')

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)


class QueryDatasetFromStruct(data.Dataset):
    def __init__(self, structFile, parent_dir, nNegSample=1000, nNeg=10, margin=0.1, input_transform=None):
        super().__init__()

        self.parent_dir = parent_dir
        self.input_transform = input_transform
        self.margin = margin

        self.dbStruct = parse_dbStruct(structFile)
        self.whichSet = self.dbStruct.whichSet
        self.dataset = self.dbStruct.dataset
        self.nNegSample = nNegSample  # number of negatives to randomly sample
        self.nNeg = nNeg  # number of negatives used for training

        # potential positives are those within nontrivial threshold range
        # fit NN to find them, search by radius
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.dbStruct.utmDb)

        # TODO use sqeuclidean as metric?
        self.nontrivial_positives = list(knn.radius_neighbors(self.dbStruct.utmQ,
                                                              radius=self.dbStruct.nonTrivPosDistSqThr ** 0.5,
                                                              return_distance=False))
        # radius returns unsorted, sort once now so we dont have to later
        for i, posi in enumerate(self.nontrivial_positives):
            self.nontrivial_positives[i] = np.sort(posi)
        # its possible some queries don't have any non trivial potential positives
        # lets filter those out
        self.queries = np.where(np.array([len(x) for x in self.nontrivial_positives]) > 0)[0]

        # potential negatives are those outside of posDistThr range
        potential_positives = knn.radius_neighbors(self.dbStruct.utmQ,
                                                   radius=self.dbStruct.posDistThr,
                                                   return_distance=False)

        self.potential_negatives = []
        for pos in potential_positives:
            self.potential_negatives.append(np.setdiff1d(np.arange(self.dbStruct.numDb),
                                                         pos, assume_unique=True))

        self.cache = None  # filepath of HDF5 containing feature vectors for images

        self.negCache = [np.empty((0,)) for _ in range(self.dbStruct.numQ)]

    def __getitem__(self, index):
        index = self.queries[index]  # re-map index to match dataset

        interval = int(self.dbStruct.numDb / (self.nNeg+1))

        negIndices = [(index + i * interval) % self.dbStruct.numDb for i in range(1, self.nNeg+1)]
        self.negCache[index] = negIndices

        query = Image.open(join(self.parent_dir, self.dbStruct.qImage[index]))
        query = query.convert('RGB')
        # due to the number of query frames is same as the database, we can use the index
        positive = Image.open(join(self.parent_dir, self.dbStruct.dbImage[index]))
        positive = positive.convert('RGB')

        if self.input_transform:
            query = self.input_transform(query)
            positive = self.input_transform(positive)

        negatives = []
        for negIndex in negIndices:
            negative = Image.open(join(self.parent_dir, self.dbStruct.dbImage[negIndex]))
            negative = negative.convert('RGB')
            if self.input_transform:
                negative = self.input_transform(negative)
            negatives.append(negative)

        negatives = torch.stack(negatives, 0)

        return query, positive, negatives, [index, index] + negIndices

    def __len__(self):
        return len(self.queries)

    def getPositives(self):
        return 0


def main_test():
    train_dataset = WholeDatasetFromStruct(root_dir)


if __name__ == "__main__":
    main_test()
