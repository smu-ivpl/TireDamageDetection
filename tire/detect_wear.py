''' classification tire wear using resnet101 model'''

import os
import sys
import torch
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Dataset(torch.utils.data.Dataset):
    def __init__(self, image):
        self.image = image
        self.len = 1
        self.transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        pil_image=Image.fromarray(self.image)
        return self.transform(pil_image)


def wearModel(image):

    image=image

    # data load
    dataset = Dataset(image)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=int(os.cpu_count()/2), pin_memory=True
    )


    # -- load model
    model = torchvision.models.resnet101()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    # load weignt
    # change sys.path to parse_config.py module path
    sys.path.insert(0, './tire')
    checkpoint = torch.load("tire/model/wear_safeWarning_model.pth")
    sys.path.remove(sys.path[0])

    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    # test model
    label = ['안전', '주의']



    for index, data in enumerate(dataloader):
        data = data.to(device)
        output = model(data)

        pred = torch.argmax(output, dim=1)
        print('*** 타이어 마모 상태 '+label[pred]+" ***\n")


def predict(image):

    image=image

    # data load
    dataset = Dataset(image)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=int(os.cpu_count()/2), pin_memory=True
    )


    # -- load model
    model = torchvision.models.resnet101()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    # load weignt
    # change sys.path to parse_config.py module path
    sys.path.insert(0, './tire')
    checkpoint = torch.load("tire/model/wear_safeWarning_model.pth")
    sys.path.remove(sys.path[0])

    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    # test model
    label = ['안전', '주의']



    for index, data in enumerate(dataloader):
        data = data.to(device)
        output = model(data)

        return torch.argmax(output, dim=1)
