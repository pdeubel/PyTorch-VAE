from typing import Any, List

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from skimage import morphology
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from scipy.spatial.distance import mahalanobis
from skimage.segmentation import mark_boundaries
from scipy.ndimage import gaussian_filter
import torch
import torch.nn.functional as F
from torchvision.models import resnet18, wide_resnet50_2
from torchvision import transforms

from models import BaseVAE
from models.types_ import Tensor


class PaDiM(BaseVAE):
    """
    Patch Distribution Modeling (PaDiM) architecture. This is not a VAE but uses a pretrained CNN to get
    embedded features of the image to detect anomalies.

    Code is heavily based on this implementation:
    https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master

    """

    def __init__(self,
                 params: dict,
                 **kwargs) -> None:

        super().__init__(params)

        self.crop_size = self.params["crop_size"]
        self.number_train_batches = self.params["number_train_batches"]
        self.number_val_batches = self.params["number_val_batches"]
        self.calculated_train_batches = 0
        self.calculated_val_batches = 0

        if self.params["arch"] == "resnet":
            self.model = resnet18(pretrained=True)
        elif self.params["arch"] == "resnet_wide":
            self.model = wide_resnet50_2(pretrained=True)
        else:
            raise RuntimeError("'{}' is not supported for the 'arch' config parameter".format(self.params["arch"]))

        self.model.eval()
        self.model.to(self.curr_device)

        self.outputs_layer1 = []
        self.outputs_layer2 = []
        self.outputs_layer3 = []

        def hook_1(module, input, output):
            self.outputs_layer1.append(output)

        def hook_2(module, input, output):
            self.outputs_layer2.append(output)

        def hook_3(module, input, output):
            self.outputs_layer3.append(output)

        # Simply save the outputs of the layers 1 to 3 after each forward pass
        self.model.layer1[-1].register_forward_hook(hook_1)
        self.model.layer2[-1].register_forward_hook(hook_2)
        self.model.layer3[-1].register_forward_hook(hook_3)

        self.learned_mean, self.learned_cov = None, None

        self.means, self.covs = None, None
        self.N = 0
        self.num_patches = 0
        self.num_embeddings = 0

        self.anomaly_dataloader = self.get_dataloader(train_split=False, abnormal_data=True)[0]
        self.anomaly_data_iterator = iter(self.anomaly_dataloader)

        self.val_images = []
        self.gt_list = []

    @staticmethod
    def embedding_concat(x, y):
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

    def forward(self, inputs: Tensor) -> Tensor:
        _ = self.model(inputs)

        return None

    def encode(self, input: Tensor) -> List[Tensor]:
        pass

    def decode(self, input: Tensor) -> Any:
        pass

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        pass

    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass

    def get_embedding(self):
        embedding = torch.cat(self.outputs_layer1, dim=0)
        embedding = self.embedding_concat(embedding, torch.cat(self.outputs_layer2, dim=0))
        embedding = self.embedding_concat(embedding, torch.cat(self.outputs_layer3, dim=0))

        # randomly select d dimension
        # embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
        # calculate multivariate Gaussian distribution
        B, C, H, W = embedding.size()

        return embedding.view(B, C, H * W), B, C, H, W

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        if self.calculated_train_batches < self.number_train_batches:
            x, labels = batch
            with torch.no_grad():
                _ = self.forward(x)

            embeddings, B, C, H, W = self.get_embedding()

            if self.means is None and self.covs is None and self.num_patches == 0 and self.num_embeddings == 0:
                self.means = torch.zeros((W * H, C))
                self.covs = torch.zeros((W * H, C, C))
                self.num_patches = W * H
                self.num_embeddings = C

            for i in range(H * W):
                patch_embeddings = embeddings[:, :, i]  # b * c
                for j in range(B):
                    self.covs[i, :, :] += torch.outer(
                        patch_embeddings[j, :],
                        patch_embeddings[j, :])  # c * c
                self.means[i, :] += patch_embeddings.sum(dim=0)  # c
            self.N += B  # number of images

            self.outputs_layer1 = []
            self.outputs_layer2 = []
            self.outputs_layer3 = []

            self.calculated_train_batches += 1

        return None

    def training_epoch_end(self, outputs) -> None:
        means = self.means.detach().clone()
        covs = self.covs.detach().clone()

        epsilon = 0.01

        identity = torch.eye(self.num_embeddings).to(self.device)
        means /= self.N
        for i in range(self.num_patches):
            covs[i, :, :] -= self.N * torch.outer(means[i, :], means[i, :])
            covs[i, :, :] /= self.N - 1  # corrected covariance
            covs[i, :, :] += epsilon * identity  # constant term

        self.learned_mean = means
        self.learned_cov = covs

    @staticmethod
    def denormalization(x):
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)

        return x

    def data_transforms(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        return transforms.Compose([transforms.CenterCrop(self.crop_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=mean,
                                                        std=std)])

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        if self.learned_mean is None or self.learned_cov is None:
            return

        if self.calculated_val_batches < self.number_val_batches:
            normal_x, normal_label = batch

            try:
                abnormal_x, abnormal_label = next(self.anomaly_data_iterator)
            except StopIteration:
                self.anomaly_data_iterator = iter(self.anomaly_dataloader)
                abnormal_x, abnormal_label = next(self.anomaly_data_iterator)

            with torch.no_grad():
                _ = self.forward(normal_x)
                _ = self.forward(abnormal_x)

            self.val_images.extend(normal_x.numpy())
            self.val_images.extend(abnormal_x.numpy())

            self.gt_list.extend(normal_label.numpy())
            self.gt_list.extend(abnormal_label.numpy())

            self.calculated_val_batches += 1

    def validation_epoch_end(self, outputs):
        if self.learned_mean is None or self.learned_cov is None:
            return

        embedding, B, C, H, W = self.get_embedding()

        dist_list = []
        for i in range(H * W):
            mean = self.learned_mean[i, :]
            conv_inv = np.linalg.inv(self.learned_cov[i, :, :])
            dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding]
            dist_list.append(dist)

        dist_list = np.array(dist_list).transpose((1, 0)).reshape((B, H, W))

        # upsample
        dist_list = torch.tensor(dist_list)
        score_map = F.interpolate(dist_list.unsqueeze(1), size=self.crop_size, mode='bilinear',
                                  align_corners=False).squeeze().numpy()

        # apply gaussian smoothing on the score map
        for i in range(score_map.shape[0]):
            score_map[i] = gaussian_filter(score_map[i], sigma=4)

        # Normalization
        max_score = score_map.max()
        min_score = score_map.min()
        scores = (score_map - min_score) / (max_score - min_score)

        # calculate image-level ROC AUC score
        img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
        gt_list = np.asarray(self.gt_list)
        fpr, tpr, thresholds = roc_curve(gt_list, img_scores)
        img_roc_auc = roc_auc_score(gt_list, img_scores)

        fig, ax = plt.subplots(1, 1)
        fig_img_rocauc = ax

        fig_img_rocauc.plot(fpr, tpr, label="ROC Curve (area = {:.2f})".format(img_roc_auc))
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title('Receiver operating characteristic')
        ax.legend(loc="lower right")

        self.logger.experiment.add_figure("ROC Curve",
                                          fig,
                                          global_step=self.current_epoch)

        precision, recall, thresholds = precision_recall_curve(gt_list, img_scores)
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        best_threshold = thresholds[np.argmax(f1)]

        self.plot_fig(scores, best_threshold)

        self.gt_list = []
        self.val_images = []

    def plot_fig(self, scores, threshold):
        num = len(scores)
        vmax = scores.max() * 255.
        vmin = scores.min() * 255.

        for i in range(num):
            img = self.val_images[i]
            img = self.denormalization(img)
            # gt = gts[i].transpose(1, 2, 0).squeeze()
            heat_map = scores[i] * 255
            mask = scores[i]
            mask[mask > threshold] = 1
            mask[mask <= threshold] = 0
            kernel = morphology.disk(4)
            mask = morphology.opening(mask, kernel)
            mask *= 255
            vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
            fig_img, ax_img = plt.subplots(1, 4, figsize=(12, 3))
            fig_img.subplots_adjust(right=0.9)
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            for ax_i in ax_img:
                ax_i.axes.xaxis.set_visible(False)
                ax_i.axes.yaxis.set_visible(False)
            ax_img[0].imshow(img)
            ax_img[0].title.set_text('Image')
            ax = ax_img[1].imshow(heat_map, cmap='jet', norm=norm)
            ax_img[1].imshow(img, cmap='gray', interpolation='none')
            ax_img[1].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
            ax_img[1].title.set_text('Predicted heat map')
            ax_img[2].imshow(mask, cmap='gray')
            ax_img[2].title.set_text('Predicted mask')
            ax_img[3].imshow(vis_img)
            ax_img[3].title.set_text('Segmentation result')
            left = 0.92
            bottom = 0.15
            width = 0.015
            height = 1 - 2 * bottom
            rect = [left, bottom, width, height]
            cbar_ax = fig_img.add_axes(rect)
            cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
            cb.ax.tick_params(labelsize=8)
            font = {
                'family': 'serif',
                'color': 'black',
                'weight': 'normal',
                'size': 8,
            }
            cb.set_label('Anomaly Score', fontdict=font)

            if self.gt_list[i] == 0:
                type_of_img = "Normal"
            else:
                type_of_img = "Abnormal"

            self.logger.experiment.add_figure("Validation Image {} - {}".format(i, type_of_img),
                                              fig_img,
                                              global_step=self.current_epoch)

        plt.close()

    def configure_optimizers(self):
        pass
