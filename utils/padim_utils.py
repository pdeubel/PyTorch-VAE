from matplotlib import pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import mahalanobis
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
import torch
import torch.nn.functional as F


def get_roc_plot_and_threshold(scores, gt_list):
    # calculate image-level ROC AUC score
    img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
    gt_list = np.asarray(gt_list)
    fpr, tpr, thresholds = roc_curve(gt_list, img_scores)
    img_roc_auc = roc_auc_score(gt_list, img_scores)

    fig, ax = plt.subplots(1, 1)
    fig_img_rocauc = ax

    fig_img_rocauc.plot(fpr, tpr, label="ROC Curve (area = {:.2f})".format(img_roc_auc))
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title('Receiver operating characteristic')
    ax.legend(loc="lower right")

    precision, recall, thresholds = precision_recall_curve(gt_list, img_scores)
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    best_threshold = thresholds[np.argmax(f1)]

    return (fig, ax), best_threshold


def _embedding_concat(x, y):
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z


def get_embedding(outputs_layer1, outputs_layer2, outputs_layer3, embedding_ids, device):
    embedding = torch.cat(outputs_layer1, dim=0).to(device)
    embedding = _embedding_concat(embedding, torch.cat(outputs_layer2, dim=0)).to(device)
    embedding = _embedding_concat(embedding, torch.cat(outputs_layer3, dim=0)).to(device)

    embedding = torch.index_select(embedding, dim=1, index=embedding_ids.to(device))

    B, C, H, W = embedding.size()

    return embedding.view(B, C, H * W), B, C, H, W


def calculate_score_map(dist_list, crop_size, min_max_norm: bool):
    # Upsample
    dist_list = torch.tensor(dist_list)
    score_map = F.interpolate(dist_list.unsqueeze(1), size=crop_size, mode='bilinear',
                              align_corners=False).squeeze().numpy()

    # Apply gaussian smoothing on the score map
    for i in range(score_map.shape[0]):
        score_map[i] = gaussian_filter(score_map[i], sigma=4)

    # Normalization
    if min_max_norm:
        max_score = score_map.max()
        min_score = score_map.min()
        scores = (score_map - min_score) / (max_score - min_score)
    else:
        scores = score_map

    return scores
