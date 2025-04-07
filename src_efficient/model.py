# https://www.kaggle.com/code/shnakazawa/image-classification-with-pytorch-and-efficientnet
# https://debuggercafe.com/transfer-learning-using-efficientnet-pytorch/

import torchvision.models as models
import torch.nn as nn

def build_model(pretrained=True, fine_tune=True, num_classes=5): # 5 classes for ME, DR and 3 for glaucoma
    if pretrained:
        print('[INFO]: Loading pre-trained weights')
    else:
        print('[INFO]: Not loading pre-trained weights')
    model = models.efficientnet_b0(pretrained=pretrained)
    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    elif not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False
    model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)
    return model

