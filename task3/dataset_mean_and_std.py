import numpy as np
import torch
from torchvision import transforms
from task3.utils.data_utils import load_zipped_pickle

samples = load_zipped_pickle('data/train.pkl') + load_zipped_pickle('data/test.pkl')

sample_means = []
sample_stds = []

for sample in samples:
    video = sample['video']
    sample_means.append(np.mean(video))
    sample_stds.append(np.std(video))

mean = sum(sample_means) / len(sample_means)
std = sum(sample_stds) / len(sample_stds)

print('===========')
print(mean, std)
print(mean/255, std/255)

transformations = transforms.Compose([transforms.ToTensor()])
sample_means = []
sample_stds = []
for sample in samples:
    video = transformations(sample['video'])
    sample_means.append(video.mean())
    sample_stds.append(video.std())

mean = sum(sample_means) / len(sample_means)
std = sum(sample_stds) / len(sample_stds)
print('===========')
print(mean, std)
