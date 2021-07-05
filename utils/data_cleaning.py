import torch
import yaml
import argparse
import os
from shutil import copyfile
from torchvision import transforms
import numpy as np
from PIL import Image

from models import vae_models

parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config', '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file',
                    default='/configs/vae_vanilla_cracks_second.yaml')
parser.add_argument('--load', '-l',
                    dest='checkpoint_file',
                    metavar='FILE',
                    default='/logs/VanillaVAE_second_concrete_crack/version_2/checkpoints/epoch=29-step=6089.ckpt',
                    help='Path to checkpoint to load experiment from')
parser.add_argument('--save_plots',
                    type=str,
                    default=False,
                    help="Specify a suffix which is used to save the plots to roc_curve_suffix.svg and reconstructed "
                         "classified_suffix.svg")

args = parser.parse_args()

with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

filename_roc_curve = None
filename_reconstructed_classified = None

if args.save_plots:
    filename_roc_curve = "roc_curve_" + args.save_plots + ".pdf"
    filename_reconstructed_classified = "reconstructed_classified_" + args.save_plots + ".pdf"

model = vae_models[config['model_params']['name']](config['exp_params'], **config['model_params'])
model = model.load_from_checkpoint(args.checkpoint_file)
model.eval()

root_dir = "/home/lei/Desktop/project-cvhci/data/SDNET2018"
clean_root_dir = "/home/lei/Desktop/project-cvhci/data/SDNET2018_clean"
dirty_root_dir = "/home/lei/Desktop/project-cvhci/data/SDNET2018_dirty"

if not os.path.exists(clean_root_dir):
    os.makedirs(clean_root_dir)
if not os.path.exists(dirty_root_dir):
    os.makedirs(dirty_root_dir)

classes = ["D", "P", "W"]

for _class in classes:
    clean_data_path = os.path.join(clean_root_dir, _class, "U" + _class)
    if not os.path.exists(clean_data_path):
        os.makedirs(clean_data_path)

    data_path = os.path.join(root_dir, _class, "U" + _class)
    data = np.array([os.path.join(data_path, img_file) for img_file in os.listdir(data_path)])

    all_pixel_wise_difference_means = []
    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    transform = transforms.Compose([transforms.Resize((config['exp_params']['img_size'], config['exp_params']['img_size'])),
                                    transforms.ToTensor(),
                                    SetRange])

    for img_path in data:
        img = Image.open(img_path)
        img = transform(img)
        batch = img.unsqueeze(0)
        with torch.no_grad():
            predictions = model.generate(batch)
        pixel_wise_diff_mean = torch.mean(torch.abs(predictions - batch), dim=(1, 2, 3)).numpy()
        all_pixel_wise_difference_means = np.concatenate([all_pixel_wise_difference_means, pixel_wise_diff_mean],axis=0)

    idx = np.argsort(all_pixel_wise_difference_means)
    idx_clean = idx[:int(0.95*len(idx))]
    idx_dirty = idx[int(0.95 * len(idx)):]
    clean_data = data[idx_clean]
    dirty_data = data[idx_dirty]

    assert np.count_nonzero(clean_data) + np.count_nonzero(dirty_data) == data.shape[0]

    for clean_file in clean_data:
        copyfile(clean_file, os.path.join(clean_data_path, os.path.basename(clean_file)))

    dirty_data_path = os.path.join(dirty_root_dir, _class, "U" + _class)
    if not os.path.exists(dirty_data_path):
        os.makedirs(dirty_data_path)

    for dirty_file in dirty_data:
        copyfile(dirty_file, os.path.join(dirty_data_path, os.path.basename(dirty_file)))

    print("Copied clean data for category '{}' to {}".format(_class, clean_data_path))

# Copy images with cracks without changing them
for _class in classes:
    new_data_path = os.path.join(clean_root_dir, _class, "C" + _class)
    if not os.path.exists(new_data_path):
        os.makedirs(new_data_path)

    data_path_anomalies = os.path.join(root_dir, _class, "C" + _class)
    data_anomalies = [os.path.join(data_path_anomalies, img_file) for img_file in os.listdir(data_path_anomalies)]

    for anomaly_img in data_anomalies:
        copyfile(anomaly_img, os.path.join(new_data_path, os.path.basename(anomaly_img)))

    print("Copied anomaly data without changing them for category '{}' to {}".format(_class, new_data_path))

print("Done")