from models import Network
from eva import clustering
import argparse
from Load_data import load_data
from utils import *



# Dataname = 'DHA'
# Dataname = 'Caltech6'
Dataname = 'Web'
# Dataname = 'NGs'
# Dataname = 'imdb'
# Dataname = 'texas'
# Dataname = 'acm'
# Dataname = 'chameleon'
parser = argparse.ArgumentParser(description='test')
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
dataset, dims, view, data_size, class_num = load_data(args.dataset)
model = Network(view, dims, args.feature_dim, args.high_feature_dim, class_num, args.batch_size,
                args.attention_dropout_rate, args.num_heads, args.attn_bias_dim, device)

model = model.to(device)
checkpoint = torch.load('./models/' + args.dataset + 'result' + '.pth')
model.load_state_dict(checkpoint)
print('test')
acc, nmi, ari, pur = clustering(model, dataset, view, data_size, class_num, device)