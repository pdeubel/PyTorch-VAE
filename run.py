import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
import yaml
from PIL import Image
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TestTubeLogger

from models import vae_models

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
parser.add_argument('--reconstruct', '-r',
                    metavar='FILE',
                    help='Provide a file that the model shall reconstruct and plot',
                    default=None)

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

if args.sample:
    # Initialize dataloader from which is sampled
    model.val_dataloader()
    model.eval()
    model.sample_images()

    print("Sampled a batch of images and plotted them.")
elif args.reconstruct is not None:
    model.eval()

    original_img = Image.open(args.reconstruct)
    original_img: torch.Tensor = model.data_transforms()(original_img).unsqueeze(0)

    reconstructed_img = model.generate(original_img)

    pixelwise_difference = torch.abs(reconstructed_img - original_img)

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=4)

    ax1.imshow(vutils.make_grid(original_img, normalize=True, nrow=1).permute(2, 1, 0).numpy())
    ax1.set_title("Original Image")

    ax2.imshow(vutils.make_grid(reconstructed_img, normalize=True, nrow=1).permute(2, 1, 0).numpy())
    ax2.set_title("Reconstructed Image")

    ax3.imshow(vutils.make_grid(pixelwise_difference, normalize=True, nrow=1).permute(2, 1, 0).numpy())
    ax3.set_title("Pixelwise Difference")

    plt.show()

    print("Done")
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
