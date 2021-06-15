from typing import Any, List

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from skimage import morphology

from scipy.spatial.distance import mahalanobis
from skimage.segmentation import mark_boundaries
from scipy.ndimage import gaussian_filter
import torch
import torch.nn.functional as F
from torchvision.models import resnet18, wide_resnet50_2
from torchvision import transforms

from models import BaseVAE
from models.types_ import Tensor
import utils.padim_utils as padim_utils


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

        try:
            self.number_train_batches = self.params["number_train_batches"]
        except KeyError:
            self.number_train_batches = len(self.get_dataloader(train_split=True, abnormal_data=False)[0])

        try:
            self.number_val_batches = self.params["number_val_batches"]
        except KeyError:
            self.number_val_batches = len(self.get_dataloader(train_split=False, abnormal_data=False)[0])

        self.calculated_train_batches = 0
        self.calculated_val_batches = 0

        if self.params["arch"] == "resnet":
            self.model = resnet18(pretrained=True)
            # max_embedding_size is the actual amount of embeddings that would result when using the net, but the paper
            # suggests randomly selecting like a 100 dimensions of this. So we save this here and select a
            # random amount of indices later in the constructor. These are then used in the get_embedding() method.
            self.max_embedding_size = 448
        elif self.params["arch"] == "resnet_wide":
            self.model = wide_resnet50_2(pretrained=True)
            self.max_embedding_size = 1792
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

        # Run one sample of a batch to get the number of patches. This equals the width and height of the first feature
        # map
        temporary_batch = next(iter(self.get_dataloader(train_split=True, abnormal_data=False)[0]))[0][0].unsqueeze(0)
        _ = self.model(temporary_batch)

        self.N = 0
        self.num_embeddings = self.params["number_of_embeddings"]
        _, C, H, W = self.outputs_layer1[0].size()
        self.num_patches = H * W

        # Empty the lists again
        self.outputs_layer1 = []
        self.outputs_layer2 = []
        self.outputs_layer3 = []

        # These indices will be used in the get_embedding() method to then have only num_embeddings instead of
        # max_embedding_size. The paper states that this saves calculation time and is not critical to the model
        # performance.
        self.embedding_ids = torch.randperm(self.max_embedding_size)[:self.num_embeddings].to(self.device)

        self.means = torch.zeros((self.num_patches, self.num_embeddings))
        self.covs = torch.zeros((self.num_patches, self.num_embeddings, self.num_embeddings))

        # Transform means and covs into a Parameter, this way the values inside them get saved when checkpoints are
        # created by PyTorch Lightning. Also since we do not do backpropagation in the PaDiM architecture, disable the
        # gradient. Otherwise there are some accessing errors which are caused by this casting to a Parameter object.
        self.means = torch.nn.Parameter(self.means, requires_grad=False).to(self.device)
        self.covs = torch.nn.Parameter(self.covs, requires_grad=False).to(self.device)

        self.anomaly_dataloader = self.get_dataloader(train_split=False, abnormal_data=True)[0]
        self.anomaly_data_iterator = iter(self.anomaly_dataloader)

        self.val_images = []
        self.gt_list = []

        self.save_hyperparameters()

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

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        if self.calculated_train_batches < self.number_train_batches:
            x, labels = batch
            with torch.no_grad():
                _ = self.forward(x)

            embeddings, B, C, H, W = padim_utils.get_embedding(self.outputs_layer1, self.outputs_layer2, self.outputs_layer3,
                                                   self.embedding_ids)

            # Empty the lists for the next batch
            self.outputs_layer1 = []
            self.outputs_layer2 = []
            self.outputs_layer3 = []

            for i in range(H * W):
                patch_embeddings = embeddings[:, :, i]  # b * c
                for j in range(B):
                    self.covs[i, :, :] += torch.outer(
                        patch_embeddings[j, :],
                        patch_embeddings[j, :])  # c * c
                self.means[i, :] += patch_embeddings.sum(dim=0)  # c
            self.N += B  # number of images

            self.calculated_train_batches += 1

        return None

    def training_epoch_end(self, outputs) -> None:
        means = self.means.clone()
        covs = self.covs.clone()

        epsilon = 0.01

        identity = torch.eye(self.num_embeddings).to(self.device)
        means /= self.N
        for i in range(self.num_patches):
            covs[i, :, :] -= self.N * torch.outer(means[i, :], means[i, :])
            covs[i, :, :] /= self.N - 1  # corrected covariance
            covs[i, :, :] += epsilon * identity  # constant term

        self.means = torch.nn.Parameter(means, requires_grad=False)
        self.covs = torch.nn.Parameter(covs, requires_grad=False)

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
        if self.calculated_train_batches == 0:
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

    def validation_epoch_end(self, outputs, min_max_norm=True):
        if self.calculated_train_batches == 0:
            return

        embedding, B, C, H, W = padim_utils.get_embedding(self.outputs_layer1, self.outputs_layer2, self.outputs_layer3,
                                              self.embedding_ids)

        # Empty the lists for the next batch
        self.outputs_layer1 = []
        self.outputs_layer2 = []
        self.outputs_layer3 = []

        scores = padim_utils.calculate_score_map(embedding, (B, C, H, W), self.means, self.covs, self.crop_size,
                                                 min_max_norm=min_max_norm)

        (fig, _), best_threshold = padim_utils.get_roc_plot_and_threshold(scores, self.gt_list)

        self.logger.experiment.add_figure("ROC Curve",
                                          fig,
                                          global_step=self.current_epoch)

        figures = self.get_plot_fig(scores, best_threshold)

        for (i, _fig) in enumerate(figures):
            fig_img, type_of_img = _fig
            self.logger.experiment.add_figure("Validation Image {} - {}".format(i, type_of_img),
                                              fig_img,
                                              global_step=self.current_epoch)

        self.gt_list = []
        self.val_images = []

        self.calculated_train_batches = 0
        self.calculated_val_batches = 0


    def create_mask(self, img_score: np.ndarray, threshold):
        idx_above_threshold = img_score > threshold
        idx_below_threshold = img_score <= threshold

        mask = img_score
        mask[idx_above_threshold] = 1
        mask[idx_below_threshold] = 0

        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        # mask *= 255

        return mask

    def create_img_subplot(self, img, img_score, threshold, vmin, vmax):
        img = self.denormalization(img)
        # gt = gts[i].transpose(1, 2, 0).squeeze()
        # heat_map = scores[i] * 255
        heat_map = np.copy(img_score)

        mask = self.create_mask(img_score, threshold)

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

        return fig_img, ax_img

    def get_plot_fig(self, scores, threshold):
        figures = []
        num = len(scores)
        # vmax = scores.max() * 255.
        # vmin = scores.min() * 255.
        vmax = scores.max()
        vmin = scores.min()

        for i in range(num):
            classified_as = scores[i].max() > threshold

            fig_img, ax_img = self.create_img_subplot(self.val_images[i], scores[i], threshold=threshold, vmin=vmin,
                                                      vmax=vmax)

            figures.append((classified_as, fig_img))

        return figures

    def configure_optimizers(self):
        pass
