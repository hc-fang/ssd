import torch
import torch.nn
import torch.nn.functional
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
from tensorloader import TensorLoader


def knn_evaluate(
    train_z: torch.Tensor, 
    train_y: torch.Tensor,
    test_z: torch.Tensor,
    test_y: torch.Tensor,
    device: torch.device,
    n_neighbors: int = 200,
    batch_size: int = 128,
):
    pred = knn_fit_predict(train_z, train_y, test_z, n_neighbors, device, batch_size)
    acc = accuracy_score(test_y, pred)
    return float(acc)


def knn_fit_predict(
    train_z: torch.Tensor,
    train_y: torch.Tensor,
    test_z: torch.Tensor,
    n_neighbors: int,
    device: torch.device,
    batch_size: int = 128
) -> torch.Tensor:
    train_z = train_z.to(device).t().contiguous()
    train_y = train_y.to(device)
    classes = len(train_y.unique())
    pred_list = []
    for z in TensorLoader(test_z, batch_size=batch_size):
        pred = knn_predict(z.to(device), train_z, train_y, classes, n_neighbors, 0.1)
        pred = pred[:, 0]
        pred_list.append(pred.cpu())
    return torch.concat(pred_list)


# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(
    feature: torch.Tensor,
    feature_bank: torch.Tensor,
    feature_labels: torch.Tensor,
    classes: int,
    knn_k: int,
    knn_t: float
):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)
    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels


# uniform loss  (https://github.com/ssnl/align_uniform)
def uniform_loss(x, t=2):
    return float(torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log())

def calc_uniform(train_feats, test_feat):
    
    # calculate uniformity for each client
    uni_train_client = [uniform_loss(client_feat) for client_feat in train_feats]

    train_feats = torch.concat(train_feats)

    # calculate uniformity for the whole training set (aggregation of client data)
    uni_train_all = uniform_loss(train_feats)

    # calculate uniformity for the test set
    uni_test = uniform_loss(test_feat)

    return uni_train_client, uni_train_all, uni_test
