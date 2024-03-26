import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision import models, transforms
import torchvision
from sklearn.manifold import LocallyLinearEmbedding, Isomap, TSNE
from sklearn.decomposition import PCA, KernelPCA
from PIL import Image
import numpy as np

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
class AddNoise(object):
    def __init__(self, prob=0.05, noise_value=255):
        self.prob = prob
        self.noise_value = noise_value

    def __call__(self, tensor):
        # Add black or white noise with the specified probability
        noise_mask = torch.rand_like(tensor) < self.prob
        tensor[noise_mask] = self.noise_value
        return tensor

class Binarize(object):
    def __call__(self, image):
        # Convert PIL Image to numpy array
        image = np.array(image)
        # Binarize the image
        _, image = cv2.threshold(image, 15, 255, cv2.THRESH_BINARY)
        # Convert back to PIL Image and return
        return Image.fromarray(image)

# Create a transform function that first converts the images to grayscale,
# then binarizes them, and finally converts them to tensors
transform = transforms.Compose([
    transforms.Grayscale(),
    Binarize(),
    transforms.ToTensor(),
    AddGaussianNoise(0., 0.1)
])

batch_size = 4
data_dir = '/data/home/tal.dugma/Monitor-Recognition/MNEW/MeasurementsRecognition/data'
image_datasets = {x: ImageFolder(os.path.join(data_dir, x), transform) for x in ['train', 'val']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size,
                             shuffle=True) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

def imshow(inp, title=None, save_path=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))

    # Convert to 8-bit unsigned integers (0-255)
    inp = (inp * 255).astype(np.uint8)
 
    cv2.imwrite(save_path, cv2.cvtColor(inp, cv2.COLOR_RGB2BGR))

# Let’s visualize a few training images so as to understand the data augmentations.
# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

# You can specify the save path here
save_path = "output_image.png"

# Call the modified function with the save_path argument
imshow(out, title=[class_names[x] for x in classes], save_path=save_path)

#create convolutional neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)

        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 61 * 61, 120)  # Adjusted input size

        # self.fc2 = nn.Linear(120, 32)

        self.fc3 = nn.Linear(120, 11)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 61 * 61)  # Adjusted size
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x
    
# Create an instance of the model
model = Net()
model = model.to("cpu")
# Define the loss function
criterion = nn.CrossEntropyLoss()
# Define the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

# Define the threshold
threshold = 0.7
# Train the model
num_epochs = 10

for epoch in range(num_epochs):
    print(f'Epoch {epoch}/{num_epochs - 1}')
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in dataloaders[phase]:
            inputs,labels = inputs.to("cpu"),labels.to("cpu")
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

print('Finished Training')
#save model
torch.save(model.state_dict(), "conv_model.pth")


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))
#show some images and their predictions
outputs = model(inputs)
_, preds = torch.max(outputs, 1)
for i in range(len(inputs)):
    inp = inputs[i].numpy().transpose((1, 2, 0))
    inp = (inp * 255).astype(np.uint8)
    cv2.imwrite(f"{preds[i]}.png", cv2.cvtColor(inp, cv2.COLOR_RGB2BGR))

