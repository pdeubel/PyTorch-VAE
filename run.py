import argparse
import os

import torch.backends.cudnn as cudnn
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TestTubeLogger
from torchsummary import summary

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
                    help='Path to checkpoint to load experiment from',
                    default=None)
parser.add_argument('--sample', '-s',
                    action='store_true',
                    help='Reconstruct image and plot it')

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

model = vae_models[config['model_params']['name']](config['exp_params'], **config['model_params'])

if args.checkpoint_file is not None:
    model = model.load_from_checkpoint(args.checkpoint_file)

    # For debugging
    # model.params["data_path"] = "/home/pdeubel/PycharmProjects/data/Concrete-Crack-Images"

# TODO Maybe delete this if not needed anymore. was used to create own ModelCheckpoint callback
# Call this explicitly otherwise the version parameter is not set of the logger which is required for the file path
# tt_logger.save()
# checkpoint_file_path = os.path.join(tt_logger.save_dir,
#                                     tt_logger.name,
#                                     f'version_{tt_logger.version}',
#                                     "checkpoints")
# callbacks=ModelCheckpoint(dirpath=checkpoint_file_path, save_top_k=-1, period=1),

if args.sample:
    # Initialize dataloader from which is sampled
    model.val_dataloader()
    model.eval()
    model.sample_images(save=False, display=True)

    print("Sampled a batch of images and plotted them.")
else:
    runner = Trainer(default_root_dir=f"{tt_logger.save_dir}",
                     min_epochs=1,
                     logger=tt_logger,
                     limit_train_batches=1.0,
                     limit_val_batches=1.0,
                     num_sanity_val_steps=5,
                     **config['trainer_params'])

    print(f"======= Training {config['model_params']['name']} =======")
    runner.fit(model)
