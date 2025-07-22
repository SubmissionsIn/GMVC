import torch
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, accuracy_score
from scipy.optimize import linear_sum_assignment
import numpy as np
from torch.utils.data import DataLoader



def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    u = linear_sum_assignment(w.max() - w)
    ind = np.concatenate([u[0].reshape(u[0].shape[0], 1), u[1].reshape([u[0].shape[0], 1])], axis=1)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def purity(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels) + 1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    return accuracy_score(y_true, y_voted_labels)


def evaluate(label, pred):
    nmi = normalized_mutual_info_score(label, pred)
    ari = adjusted_rand_score(label, pred)
    acc = cluster_acc(label, pred)
    pur = purity(label, pred)
    return acc, nmi, ari, pur


def inference(dataset, loader, model, view, data_size, device):
    """
    :return:
    total_pred: prediction among all modalities
    pred_vectors: predictions of each modality, list
    labels_vector: true label
    Hs: high-level features
    Zs: low-level features
    """
    model.eval()
    Hs = []
    Zs = []
    soft_vector = []
    pred_vectors = []
    for v in range(view):
        Hs.append([])
        Zs.append([])
        pred_vectors.append([])
    labels_vector = []
    for step, (xs, y, _) in enumerate(loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        with torch.no_grad():
            qs, preds = model.forward_cluster(xs)
            _, zs, hs, _ = model.forward(xs)
            q = sum(qs) / view
        for v in range(view):
            hs[v] = hs[v].detach()
            zs[v] = zs[v].detach()
            preds[v] = preds[v].detach()
            pred_vectors[v].extend(preds[v].cpu().detach().numpy())
            Hs[v].extend(hs[v].cpu().detach().numpy())
            Zs[v].extend(zs[v].cpu().detach().numpy())
        q = q.detach()
        soft_vector.extend(q.cpu().detach().numpy())
        labels_vector.extend(y.numpy())
    total_pred = np.argmax(np.array(soft_vector), axis=1)
    labels_vector = np.array(labels_vector).reshape(data_size)


    for v in range(view):
        Hs[v] = np.array(Hs[v])
        Zs[v] = np.array(Zs[v])
        pred_vectors[v] = np.array(pred_vectors[v])

    return Hs, Zs, total_pred, pred_vectors, labels_vector


def clustering(model, dataset, view, data_size, cluster_num, device):
    test_loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
    )
    high_level_vectors, low_level_vectors, total_label, pred_label, label_vector = inference(dataset, test_loader, model,
                                                                                         view, data_size, device)



    acc2, nmi1, ari1, pur1 = evaluate(label_vector, total_label)
    print('ACC{} = {:.4f} NMI{} = {:.4f} ARI{} = {:.4f} pur{}={:.4f}'.format(1, acc2,
                                                                             1, nmi1,
                                                                             1, ari1,
                                                                             1, pur1))



    return acc2, nmi1, ari1, pur1
