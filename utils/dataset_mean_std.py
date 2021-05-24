from torchvision import transforms
from datasets.sdnet2018 import SDNet2018
from torch.utils.data import DataLoader

transformation_functions = transforms.Compose([transforms.ToTensor()])

dataset_normal = SDNet2018(root_dir="/home/pdeubel/PycharmProjects/data/SDNET2018",
                           split="train",
                           abnormal_data=False,
                           transform=transformation_functions)

normal_dataloader = DataLoader(dataset_normal,
                               batch_size=8,
                               shuffle=True,
                               drop_last=False)

mean = 0.
std = 0.
nb_samples = 0.
for (data, _) in normal_dataloader:
    batch_samples = data.size(0)
    data = data.view(batch_samples, data.size(1), -1)
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += batch_samples

mean /= nb_samples
std /= nb_samples

print("SDNet Train Normal Mean: {}".format(mean))
print("SDNet Train Normal Std: {}".format(std))
