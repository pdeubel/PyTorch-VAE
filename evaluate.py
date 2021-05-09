"""
Idea of this script:

1. Load Model -> model
2. Val Normal data
3. Val Abnormal data
4. Take random subset of normal and abnormal data + their labels and mix them (of course keep labels in right order)
5. Feed each image through model
6. Take pixel wise difference between original image and reconstruction
7. Then y_true and y_score are known -> plot ROC Curve
8. ROC curve gives a good threshold
9. Show the reconstructed images and how they have been classified
"""

import argparse

import numpy as np
from sklearn.metrics import roc_curve, auc
import yaml

from models import *

parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config', '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file',
                    default='configs/vae.yaml')
parser.add_argument('--load', '-l',
                    dest='checkpoint_file',
                    metavar='FILE',
                    help='Path to checkpoint to load experiment from')

args = parser.parse_args()

with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

model = vae_models[config['model_params']['name']](config['exp_params'], **config['model_params'])
model = model.load_from_checkpoint(args.checkpoint_file)
model.eval()

transform = model.data_transforms()

dataset_normal = ConcreteCracksDataset(root_dir=config['exp_params']['data_path'],
                                       split="val",
                                       abnormal_data=False,
                                       transform=transform)

dataset_abnormal = ConcreteCracksDataset(root_dir=config['exp_params']['data_path'],
                                         split="val",
                                         abnormal_data=True,
                                         transform=transform)

dataloader_normal = DataLoader(dataset_normal,
                               batch_size=150,
                               shuffle=True,
                               drop_last=True)

dataloader_abnormal = DataLoader(dataset_abnormal,
                                 batch_size=150,
                                 shuffle=True,
                                 drop_last=True)

batch_normal, labels_normal = next(iter(dataloader_normal))
batch_abnormal, labels_abnormal = next(iter(dataloader_abnormal))

random_indices_normal = np.random.randint(0, 149, size=50)
random_indices_abnormal = np.random.randint(0, 149, size=50)

batch_normal = batch_normal[random_indices_normal]
labels_normal = labels_normal[random_indices_normal]

batch_abnormal = batch_abnormal[random_indices_abnormal]
labels_abnormal = labels_abnormal[random_indices_abnormal]

batch = np.concatenate([batch_normal, batch_abnormal])
labels = np.concatenate([labels_normal, labels_abnormal])

assert batch.shape[0] == labels.shape[0]

indices = np.arange(batch.shape[0])
# In-place shuffling
np.random.shuffle(indices)

batch = batch[indices]
labels = labels[indices]

batch = torch.from_numpy(batch)

predictions = model.generate(batch)

pixel_wise_diff = torch.mean(torch.abs(predictions - batch), dim=(1, 2, 3)).detach().numpy()

fpr, tpr, thresholds = roc_curve(labels, pixel_wise_diff)
roc_auc = auc(fpr, tpr)

best_threshold = thresholds[np.argmax(tpr - fpr)]

lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")

print("Best Threshold: {}".format(best_threshold))

labels_predicted = pixel_wise_diff > best_threshold

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)

ax1.imshow(vutils.make_grid(predictions[labels_predicted == 0], normalize=True, nrow=5).permute(2, 1, 0).numpy())
ax1.set_title("Reconstructed - Predicted Negative")

ax2.imshow(vutils.make_grid(predictions[labels_predicted == 1], normalize=True, nrow=5).permute(2, 1, 0).numpy())
ax2.set_title("Reconstructed - Predicted Positive")

plt.tight_layout()
plt.show()

print("Done")
