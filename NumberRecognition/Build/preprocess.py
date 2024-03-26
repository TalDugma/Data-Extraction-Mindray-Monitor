#file preprocessing the data- saves it as grayscale images, and shows a couple of examples

import numpy as np
from torchvision.datasets import ImageFolder
from torchvision import transforms
import cv2

raw_data_dir = 'insert'
image_datasets = ImageFolder(raw_data_dir, transforms.ToTensor())
print(image_datasets.classes)
new_data_dir = 'insert'
#make every datasample grayscale
for i in range(len(image_datasets)):
    image, label = image_datasets[i]
    image = cv2.cvtColor(image.numpy().transpose((1, 2, 0)), cv2.COLOR_BGR2GRAY)
    # Scale pixel values back to [0, 255]
    image = (image * 255).astype(np.uint8)
    cv2.imwrite(new_data_dir + "/bin_" + str(i) + ".png", image)
    print(image_datasets.classes[label], i)