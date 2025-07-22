from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch
import os.path as osp
import torch.nn as nn


def normalize_features(x):
    rowsum = np.array(x.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_inv = r_inv.reshape((x.shape[0], -1))
    x = x * r_inv
    return x

def normalize_adj(x):
    # rowsum = np.array(x.sum(1))
    # colsum = np.array(x.sum(0))
    # r_inv = np.power(rowsum, -0.5).flatten()
    # r_inv[np.isinf(r_inv)] = 0.
    # c_inv = np.power(colsum, -0.5).flatten()
    # c_inv[np.isinf(c_inv)] = 0.
    # r_inv = r_inv.reshape((x.shape[0], -1))
    # c_inv = c_inv.reshape((-1, x.shape[1]))
    # x = x * r_inv * c_inv

    # rowsum = np.array(x.sum(1))
    # x = x + np.diag(rowsum) - np.eye(x.shape[0])
    # rowsum = np.array(x.sum(1))
    # r_inv = np.power(rowsum, -1.).flatten()
    # r_inv[np.isinf(r_inv)] = 0.
    # r_inv = r_inv.reshape((x.shape[0], -1))
    # x = x * r_inv

    rowsum = np.array(x.sum(1))
    r_inv = np.power(rowsum, -1.).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_inv = r_inv.reshape((x.shape[0], -1))
    x = x * r_inv
    return x

class DHA(Dataset):
    def __init__(self, path):

        data1 = scipy.io.loadmat(path)['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path)['X2'].astype(np.float32)
        labels = scipy.io.loadmat(path)['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
            self.x2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

    def update_view(self, new_data):
        """
        Update the value of self. view
        """
        self.x1 = new_data[0].numpy().astype(np.float32)
        self.x2 = new_data[1].numpy().astype(np.float32)


class Web(Dataset):
    def __init__(self, path):
        print('web')
        x = scipy.io.loadmat(path)
        data1 = scipy.io.loadmat(path)['X'][0,0].astype(np.float32)
        data2 = scipy.io.loadmat(path)['X'][0,1].astype(np.float32)
        labels = scipy.io.loadmat(path)['gnd']
        unique_numbers = len(np.unique(labels))
        self.x1 = data1
        self.x2 = data2
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
            self.x2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

    def update_view(self, new_data):
        """
        Update the value of self. view
        """
        self.x1 = new_data[0].numpy().astype(np.float32)
        self.x2 = new_data[1].numpy().astype(np.float32)


class NGs(Dataset):
    def __init__(self, path):
        print('ngs')
        scaler = MinMaxScaler()
        data1 = scipy.io.loadmat(path)['data'][0,0].astype(np.float32).transpose()
        data2 = scipy.io.loadmat(path)['data'][0,1].astype(np.float32).transpose()
        data3 = scipy.io.loadmat(path)['data'][0,2].astype(np.float32).transpose()
        labels = scipy.io.loadmat(path)['truelabel'][0,0].transpose()
        unique_numbers = len(np.unique(labels))
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
            self.x2[idx]), torch.from_numpy(
            self.x3[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

    def update_view(self, new_data):
        """
        Update the value of self. view
        """
        self.x1 = new_data[0].numpy().astype(np.float32)
        self.x2 = new_data[1].numpy().astype(np.float32)
        self.x3 = new_data[2].numpy().astype(np.float32)



import scipy.sparse as sp


class ACM(Dataset):
    def __init__(self, path):
        print('acm')
        scaler = MinMaxScaler()
        data1 = scipy.io.loadmat(path)['feature'].astype(np.float32)
        data2 = scipy.io.loadmat(path)['feature'].astype(np.float32)

        view1 = scipy.io.loadmat(path)['PAP'].astype(np.float32)
        view2 = scipy.io.loadmat(path)['PLP'].astype(np.float32)

        labels = scipy.io.loadmat(path)['label']
        labels = np.argmax(labels, axis=1).reshape(3025,1)

        self.x1 = data1
        self.x2 = data2
        self.view1 = view1
        self.view2 = view2
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
            self.x2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

    def get_graph(self, idx):
        return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
            self.view2[idx])]

    def get_adj(self, idx):
        return [torch.from_numpy(self.view1[idx,:][:,idx]), torch.from_numpy(
            self.view2[idx,:][:,idx])]

    def update_view(self, new_data):
        """
        Update the value of self. view
        """
        self.x1 = new_data[0].numpy().astype(np.float32)
        self.x2 = new_data[1].numpy().astype(np.float32)
        # self.x3 = new_data[2].numpy().astype(np.float32)

class IMDB(Dataset):
    def __init__(self, path):
        print('imdb')
        scaler = MinMaxScaler()
        data1 = scipy.io.loadmat(path)['feature'].astype(np.float32)
        data2 = scipy.io.loadmat(path)['feature'].astype(np.float32)



        view1 = scipy.io.loadmat(path)['MAM'].astype(np.float32)
        view2 = scipy.io.loadmat(path)['MDM'].astype(np.float32)


        labels = scipy.io.loadmat(path)['label']

        labels = np.argmax(labels, axis=1).reshape(4780, 1)


        self.x1 = data1
        self.x2 = data2

        self.view1 = view1
        self.view2 = view2

        self.y = labels


    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
            self.x2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

    def get_adj(self, idx):
        return [torch.from_numpy(self.view1[idx,:][:,idx]), torch.from_numpy(
            self.view2[idx,:][:,idx])]

    def get_graph(self, idx):
        return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
            self.view2[idx])]

    def update_view(self, new_data):
        """
        Update the value of self. view
        """
        self.x1 = new_data[0].numpy().astype(np.float32)
        self.x2 = new_data[1].numpy().astype(np.float32)


class Texas(Dataset):
    def __init__(self, dataset):
        print('texas')
        scaler = MinMaxScaler()
        path = './data/{}/'.format(dataset)

        f = np.loadtxt(path + '{}.feature'.format(dataset), dtype=float)

        data1 = f.astype(np.float32)
        data2 = f.astype(np.float32)
        l = np.loadtxt(path + '{}.label'.format(dataset), dtype=int)
        labels = l.reshape(183,1)

        struct_edges = np.genfromtxt(path + '{}.edge'.format(dataset), dtype=np.int32)
        sedges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape)
        sadj = sp.coo_matrix((np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])),
                             shape=(183, 183), dtype=np.float32)
        sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
        sadj = sadj + 1 * sp.eye(sadj.shape[0])
        adjs_labels = sadj.todense()
        view1 = adjs_labels.A.astype(np.float32)
        view2 = adjs_labels.A.astype(np.float32)



        self.x1 = data1
        self.x2 = data2


        self.view1 = view1
        self.view2 = view2

        self.y = labels


    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
            self.x2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()


    def get_adj(self, idx):
        return [torch.from_numpy(self.view1[idx,:][:,idx]), torch.from_numpy(
            self.view2[idx,:][:,idx])]

    def get_graph(self, idx):
        return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
            self.view2[idx])]

    def update_view(self, new_data):
        """
        Update the value of self. view
        """
        self.x1 = new_data[0].numpy().astype(np.float32)
        self.x2 = new_data[1].numpy().astype(np.float32)


class Chameleon(Dataset):
    def __init__(self, dataset):
        print('chameleon')
        scaler = MinMaxScaler()
        path = './data/{}/'.format(dataset)

        f = np.loadtxt(path + '{}.feature'.format(dataset), dtype=float)

        data1 = f.astype(np.float32)
        data2 = f.astype(np.float32)
        l = np.loadtxt(path + '{}.label'.format(dataset), dtype=int)
        labels = l.reshape(2277,1)

        struct_edges = np.genfromtxt(path + '{}.edge'.format(dataset), dtype=np.int32)
        sedges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape)
        sadj = sp.coo_matrix((np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])),
                             shape=(2277, 2277), dtype=np.float32)
        sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
        sadj = sadj + 1 * sp.eye(sadj.shape[0])
        adjs_labels = sadj.todense()
        view1 = adjs_labels.A.astype(np.float32)
        view2 = adjs_labels.A.astype(np.float32)



        self.x1 = data1
        self.x2 = data2


        self.view1 = view1
        self.view2 = view2

        self.y = labels


    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
            self.x2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()


    def get_adj(self, idx):
        return [torch.from_numpy(self.view1[idx,:][:,idx]), torch.from_numpy(
            self.view2[idx,:][:,idx])]

    def get_graph(self, idx):
        return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
            self.view2[idx])]

    def update_view(self, new_data):
        """
        Update the value of self. view
        """
        self.x1 = new_data[0].numpy().astype(np.float32)
        self.x2 = new_data[1].numpy().astype(np.float32)

class Caltech6(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path)
        scaler = MinMaxScaler()
        self.view1 = scaler.fit_transform(data['X1'].astype(np.float32))
        self.view2 = scaler.fit_transform(data['X2'].astype(np.float32))
        self.view3 = scaler.fit_transform(data['X3'].astype(np.float32))
        self.view4 = scaler.fit_transform(data['X4'].astype(np.float32))
        self.view5 = scaler.fit_transform(data['X5'].astype(np.float32))
        self.view6 = scaler.fit_transform(data['X6'].astype(np.float32))
        self.labels = scipy.io.loadmat(path)['Y'].transpose()
        unique_values = np.unique(self.labels)
        # self.labels[self.labels == 95] = 7
        a = 1


    def __len__(self):
        return 1400

    def __getitem__(self, idx):
        return ([torch.from_numpy(self.view1[idx]), torch.from_numpy(
            self.view2[idx]), torch.from_numpy(self.view3[idx]), torch.from_numpy(
            self.view6[idx]), torch.from_numpy(self.view5[idx]),  torch.from_numpy(self.view4[idx])],
                torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long())


    def update_view(self, new_data):
        """
        Update the value of self. view
        """
        self.view1 = new_data[0].numpy().astype(np.float32)
        self.view2 = new_data[1].numpy().astype(np.float32)
        self.view3 = new_data[2].numpy().astype(np.float32)
        self.view6 = new_data[3].numpy().astype(np.float32)
        self.view5 = new_data[4].numpy().astype(np.float32)
        self.view4 = new_data[5].numpy().astype(np.float32)



def load_data(dataset):
    if dataset == "DHA":
        dataset = DHA('data/DHA.mat')
        dims = [110, 6144]
        view = 2
        data_size = 483
        class_num = 23
    elif dataset == "Web":
        dataset = Web('data/WebKB.mat')
        dims = [2949, 334]
        view = 2
        data_size = 1051
        class_num = 2
    elif dataset == "NGs":
        dataset = NGs('data/NGs.mat')
        dims = [2000,2000,2000]
        view = 3
        data_size = 500
        class_num = 5
    elif dataset == "acm":
        dataset = ACM('data/acm.mat')
        dims = [1870,1870]
        view = 2
        data_size = 3025
        class_num = 3
    elif dataset == 'imdb':
        dataset = IMDB('data/imdb5k.mat')
        dims = [1232, 1232]
        view = 2
        data_size = 4780
        class_num = 3
    elif dataset == 'texas':
        dataset = Texas('texas')
        dims = [1703, 1703]
        view = 2
        data_size = 183
        class_num = 5
    elif dataset == 'chameleon':
        dataset = Chameleon('chameleon')
        dims = [2325, 2325]
        view = 2
        data_size = 2277
        class_num = 5
    elif dataset == 'Caltech6':
        dataset = Caltech6('data/Caltech.mat')
        dims = [48, 40, 254, 928, 512, 1984]
        view = 6
        data_size = 1400
        class_num = 7
    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num