import argparse
import os
import shutil

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml
from pytorch_lightning.loggers import TestTubeLogger

from models import PaDiM

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

tt_logger = TestTubeLogger(
    save_dir=config['logging_params']['save_dir'],
    name=config['logging_params']['name'],
    debug=False,
    create_git_tag=False,
)

# For reproducibility
torch.manual_seed(config['logging_params']['manual_seed'])
np.random.seed(config['logging_params']['manual_seed'])
cudnn.deterministic = True
cudnn.benchmark = False

model = PaDiM(config['exp_params'], **config['model_params'])
params = config['exp_params']
params["dataloader_workers"] = 0
# params["batch_size"] = 4
model = model.load_from_checkpoint(args.checkpoint_file, params=params)

model.anomaly_dataloader = model.get_dataloader(train_split=False, abnormal_data=True)[0]
model.anomaly_data_iterator = iter(model.anomaly_dataloader)

# For debugging
# model.params["data_path"] = "/home/pdeubel/PycharmProjects/data/Concrete-Crack-Images"

normal_val_dataloader_iter, _ = model.get_dataloader(train_split=False, abnormal_data=False)
abnormal_val_dataloader_iter, _ = model.get_dataloader(train_split=False, abnormal_data=True)

model.calculated_train_batches = 1

for i, normal_batch in enumerate(normal_val_dataloader_iter):
    model.validation_step(batch=normal_batch, batch_idx=i, number_of_batches=len(normal_val_dataloader_iter))

for i, abnormal_batch in enumerate(abnormal_val_dataloader_iter):
    model.validation_step(batch=abnormal_batch, batch_idx=i, number_of_batches=len(abnormal_val_dataloader_iter))

model.val_images_predicted = np.array(model.val_images_predicted).flatten()

(roc_auc_fig, _), best_threshold = model.get_roc_plot_and_threshold()

figures = model.save_plot_figs(model.val_images_scores_visualize, best_threshold)

experiment_dir = args.checkpoint_file.split("checkpoints")[0]
dir_path = os.path.join(experiment_dir, "evaluation")

try:
    shutil.rmtree(dir_path)
except FileNotFoundError:
    # Has not been created
    pass

os.makedirs(dir_path, exist_ok=True)

roc_auc_fig.savefig(os.path.join(dir_path, "roc_auc.png"))

for i, (classified_as, _fig) in enumerate(figures):
    if classified_as:
        classified_as_str = "Anomaly"
    else:
        classified_as_str = "Normal"

    _fig.savefig(os.path.join(dir_path, "classified_{}_{}.png".format(classified_as_str, i)))

print("Saved ROC and some validation images to {}".format(dir_path))
