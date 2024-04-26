import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from utils.model import *

transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                   (0.5,),(0.5,))
               ])

dataset= datasets.ImageFolder(root='data/digits/assets', transform=transform)
validation_size= 0.2

train_size = int((1 - validation_size) * len(dataset))
test_size = len(dataset) - train_size
trainset, testset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
test_loader = DataLoader(testset, batch_size=64)
device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model= Net()
my_trainer= trainer(model, train_loader, test_loader, device)
my_trainer.train(1)
