from typing import Any, List

import numpy as np
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
                 crop_size: int = 224) -> None:

        super().__init__(params)

        self.crop_size = crop_size

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

        # Simply save the outputs of the layers 1 to 3 after the forward pass
        self.model.layer1[-1].register_forward_hook(hook_1)
        self.model.layer2[-1].register_forward_hook(hook_2)
        self.model.layer3[-1].register_forward_hook(hook_3)

        self.learned_mean, self.learned_cov = None, None

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


    def encode(self, input: Tensor) -> List[Tensor]:
        pass

    def decode(self, input: Tensor) -> Any:
        pass

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        pass

    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        x, labels = batch

        self.forward(x)

        return None

    def training_epoch_end(self, outputs) -> None:
        embedding = torch.cat(self.outputs_layer1, dim=0)
        embedding = self.embedding_concat(embedding, torch.cat(self.outputs_layer2, dim=0))
        embedding = self.embedding_concat(embedding, torch.cat(self.outputs_layer3, dim=0))

        # randomly select d dimension
        # embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
        # calculate multivariate Gaussian distribution
        B, C, H, W = embedding.size()
        embedding = embedding.view(B, C, H * W)
        mean = torch.mean(embedding, dim=0).numpy()
        cov = torch.zeros(C, C, H * W).numpy()
        I = np.identity(C)
        for i in range(H * W):
            # cov[:, :, i] = LedoitWolf().fit(embedding_vectors[:, :, i].numpy()).covariance_
            cov[:, :, i] = np.cov(embedding[:, :, i].numpy(), rowvar=False) + 0.01 * I
        # save learned distribution

        self.learned_mean = mean
        self.learned_cov = cov

    def data_transforms(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.denormalize = lambda x: (x * std) + mean

        return transforms.Compose([transforms.CenterCrop(self.crop_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=mean,
                                                        std=std)])

