from models import Network
from eva import clustering
from torch.utils.data import Dataset
import argparse
from lossfunction import Loss
from Load_data import load_data
import os
from tqdm import tqdm
import time
from utils import *
import torch.nn.functional as F

start_time = time.time()
Dataname = 'DHA'
# Dataname = 'acm'
# Dataname = 'imdb'
# Dataname = 'texas'
# Dataname = 'chameleon'
# Dataname = 'Caltech6'
# Dataname = 'Web'
# Dataname = 'NGs'
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--temperature_f", default=0.5)
parser.add_argument("--temperature_l", default=1.0)
parser.add_argument("--learning_rate", default=0.0003)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--workers", default=8)
parser.add_argument("--mse_epochs", default=200)
parser.add_argument("--con_epochs", default=50)
parser.add_argument("--feature_dim", default=512)
parser.add_argument("--high_feature_dim", default=128)
parser.add_argument("--top_k", type=int, default=6, help="The number of neighbors to search")
parser.add_argument("--num_kmeans", type=int, default=5, help="The number of K-means Clustering for being robust to randomness")
parser.add_argument("--alpha", default=1,
                    help="Reconstruction error coefficient", type=float)
parser.add_argument("--beta", default=0.1,
                    help="Independence constraint coefficient", type=float)
parser.add_argument('--attn_bias_dim', type=int, default=6)
parser.add_argument('--attention_dropout_rate', type=float, default=0.5)
parser.add_argument('--num_heads', type=int, default=8)
parser.add_argument('--K', type=int, default=30)
parser.add_argument('--reg_zero', default=1e10)
parser.add_argument('--gamma', default=0.001)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(1)


if args.dataset == "NGs":
    args.learning_rate = 0.0003
    args.gamma = 0.001
    args.K = 30
    args.top_k = 4
    args.con_epochs = 50
    seed = 10
if args.dataset == "Web":
    args.learning_rate = 0.0003
    args.gamma = 0.1
    args.K = 33
    args.top_k = 4
    args.con_epochs = 200
    seed = 10
if args.dataset == "DHA":
    args.learning_rate = 0.0003
    args.K = 13
    args.top_k = 4
    args.gamma = 0.001
    args.con_epochs = 200
    seed = 10
if args.dataset == 'Caltech6':
    args.learning_rate = 0.0004
    args.gamma = 0.1
    args.K = 5
    args.top_k = 4
    args.con_epochs = 100
    seed = 5
if args.dataset == "acm":
    args.learning_rate = 0.001
    args.gamma = 0.001
    args.K = 1
    args.top_k = 6
    args.con_epochs = 50
    seed = 10
if args.dataset == 'imdb':
    args.learning_rate = 0.0006
    args.gamma = 0.01
    args.K = 1
    args.top_k = 6
    args.con_epochs = 50
    seed = 10
if args.dataset == 'texas':
    args.learning_rate = 0.0008
    args.gamma = 0.01
    args.K = 24
    args.top_k = 6
    args.con_epochs = 5
    seed = 10
if args.dataset == 'chameleon':
    args.learning_rate = 0.0002
    args.gamma = 0.01
    args.K = 25
    args.top_k = 6
    args.con_epochs = 50
    seed = 10

"""
view-specific pre-train module
"""
def pretrain(epoch):
    tot_loss = 0.
    mse = torch.nn.MSELoss()
    for batch_idx, (xs, _, idx) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        xrs, _, _, _ = model(xs)
        loss_list = []
        for v in range(view):
            loss_list.append(mse(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))


"""
View-cross contrastive learning module
"""
def feature_contrastive_train(epoch):
    tot_loss = 0.
    count = 0
    mes = torch.nn.MSELoss()
    for batch_idx, (xs, ys, idx) in enumerate(data_loader):
        count += 1
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        xrs, zs, hs, ls = model(xs)

        loss_list = []

        hs_tensor = torch.tensor([]).cuda()
        for v in range(view):
            hs_tensor = torch.cat((hs_tensor, torch.mean(hs[v], 1).unsqueeze(1)), 1)  # mean by feature dimension and connect, (n * v)
        # transpose
        hs_tensor = hs_tensor.t()  # (v, n)
        # process by the attention
        hs_atten = model.attention_net(hs_tensor, hs_tensor, hs_tensor)  # (v * 1)
        s_p = torch.nn.Softmax(dim=0)
        r = s_p(hs_atten)  # Calculate view weight vector b^v

        adaptive_weight = r
        feature_fuse = torch.zeros([hs[0].shape[0], hs[0].shape[1]]).cuda()  # Initialize H
        for v in range(view):
            feature_fuse = feature_fuse + adaptive_weight[v].item() * hs[v] # obtain global feature H by Eq.(12)
        # Calculate the closed form solution of global graph A
        D = pairwise_distance(feature_fuse)  # Calculate distance matrix
        D = normalize(D)
        res = torch.mm(feature_fuse, torch.transpose(feature_fuse, 0, 1))  # Calculate H*H.T
        inv = torch.inverse(res + args.beta * torch.eye(feature_fuse.shape[0]).to(device))  # Inverse matrix in A
        front = res - args.alpha/2 * D  # The first part of A
        S = torch.mm(front, inv)  # Calculate global graph A

        S = (S + S.t()) / 2
        S.fill_diagonal_(float('-inf'))  # to ensure that positive samples do not choose themselves repeatedly

        _, I_knn = S.topk(k=args.top_k, dim=1, largest=True, sorted=True)  # takes the first k largest values and returns the index
        knn_neighbor = create_sparse(I_knn)  # Construct KNN matrix
        knn_neighbor = knn_neighbor.to_dense().to(dtype=torch.float32)

        for v in range(view):
            loss_list.append(mes(xs[v], xrs[v]))
            for w in range(v+1, view):
                loss_list.append(criterion.forward_label(ls[v], ls[w]))  # CGC loss
                loss_list.append(criterion.feature_contrastive_loss(hs[v], hs[w], knn_neighbor, args.gamma))  # GGC loss

        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss/len(data_loader)))

# if not os.path.exists('./models'):
#     os.makedirs('./models')


accs = []
nmis = []
aris = []
purs = []
best_acc1 = 0
best_acc2 = 0
best_nmi2 = 0
best_ari2 = 0
best_pur2 = 0
best_epoch = 0


T = 1
acc_l = []
nmi_l = []
ari_l = []
loss_l = []

for i in range(T):
    dataset, dims, view, data_size, class_num = load_data(args.dataset)

    print(dataset, dims, view, data_size, class_num)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    print("ROUND:{}".format(i+1))
    print(args.K)

    setup_seed(seed+i)

    if Dataname in ['DHA', 'NGs', 'Web', 'Caltech6']:
        node_filter_adaptive_i(dataset, view, args.K, args.reg_zero, args.beta, args.alpha, verbose=False)  # AGC
    elif Dataname in ['acm', 'imdb', 'texas', 'chameleon']:
        node_filter_adaptive_g(dataset, view, args.K, args.reg_zero, verbose=False)  # AGC

    model = Network(view, dims, args.feature_dim, args.high_feature_dim, class_num, args.batch_size,
                    args.attention_dropout_rate, args.num_heads, args.attn_bias_dim, device)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = Loss(args.batch_size, class_num, args.temperature_f, args.temperature_l, device).to(device)

    # pre-train
    for epoch in range(1, args.mse_epochs+1):
        pretrain(epoch)


    accmax=0
    nmimax=0
    arimax=0
    purmax=0
    for epoch in range(args.mse_epochs+1, args.mse_epochs+args.con_epochs+1):
        feature_contrastive_train(epoch)

        acc, nmi, ari, pur = clustering(model, dataset, view, data_size, class_num, device)


        if epoch == args.mse_epochs+args.con_epochs:
            accmax = acc
            nmimax = nmi
            arimax = ari
            purmax = pur
            best_epoch = epoch

    accmax = round(accmax, 4)
    nmimax = round(nmimax, 4)
    arimax = round(arimax, 4)
    accs.append(accmax)
    nmis.append(nmimax)
    aris.append(arimax)
    purs.append(purmax)

    # Save model
    # state = model.state_dict()
    # torch.save(state, './models/' + args.dataset + 'result' + '.pth')



print(Dataname)
print('acc:',accs, np.mean(accs), np.std(accs))
print('nmi:',nmis, np.mean(nmis), np.std(nmis))
print('ari:',aris, np.mean(aris), np.std(aris))

print('best_epoch:',best_epoch)
end_time = time.time()  # 记录结束时间
elapsed_time = end_time - start_time  # 计算运行时间

print(f"代码运行时间：{elapsed_time:.4f} 秒")

