import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import time
from tqdm.auto import tqdm
from model import build_model
from datasets import split_datasets, get_data_loaders, SLODataset, BATCH_SIZE
from utils import save_model_onnx, save_plots, save_feature_model_onnx
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

# make tensorboard work
# once it works, can remove the save plots feature 
writer = SummaryWriter()


parser = argparse.ArgumentParser()
parser.add_argument(
    '-e', '--epochs', type=int, default=1,
    help='Number of epochs to train our network for'
)
parser.add_argument(
    '-pt', '--pretrained', action='store_true',
    help='Whether to use pretrained weights or not'
)
parser.add_argument(
    '-lr', '--learning-rate', type=float,
    dest='learning_rate', default=0.0001,
    help='Learning rate for training the model'
)
args = vars(parser.parse_args())

def train(model, trainloader, optimizer, criterion_binary, criterion_multiclass, alpha=1, beta=1):
    # alpha is a scaling factor for the relative importance of multiclass loss vs binary loss
    model.train()
    print('Training')
    train_running_loss = 0.0
    correct_me = 0
    correct_dr = 0
    correct_glaucoma = 0
    counter = 0
    
    all_preds_me = []
    all_labels_me = []
    all_preds_dr = []
    all_labels_dr = []
    all_preds_glaucoma = []
    all_labels_glaucoma = []

    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        # load images and labels and send to device
        image, label_me, label_dr, label_glaucoma = data
        image = image.to(device)
        label_me = label_me.to(device)
        label_dr = label_dr.to(device)
        label_glaucoma = label_glaucoma.to(device)
        
        # optimizer set
        optimizer.zero_grad()
        outputs = model(image)

        # loss from binary
        binary_outputs = outputs[:,:2]
        binary_labels = torch.stack((label_me, label_dr), dim=1).float().to(device)
        loss_binary = criterion_binary(binary_outputs, binary_labels.float())

        # loss from multiclass
        multiclass_outputs = outputs[:, 2:]
        loss_multiclass = criterion_multiclass(multiclass_outputs, label_glaucoma)

        # combine losses
        loss = alpha * loss_binary + beta * loss_multiclass
        train_running_loss += loss.item()
        
        # change preds to be label-wise and combine binary + multiclass
        preds_binary = (torch.sigmoid(binary_outputs) > 0.5).float()
        pred_multiclass = torch.argmax(multiclass_outputs, dim=1)
        correct_me += (preds_binary[:, 0] == label_me).sum().item()
        correct_dr += (preds_binary[:, 1] == label_dr).sum().item()
        correct_glaucoma += (pred_multiclass == label_glaucoma).sum().item()

        all_preds_me.extend(preds_binary[:, 0].detach().cpu().numpy())
        all_labels_me.extend(label_me.detach().cpu().numpy())

        all_preds_dr.extend(preds_binary[:, 1].detach().cpu().numpy())
        all_labels_dr.extend(label_dr.detach().cpu().numpy())

        all_preds_glaucoma.extend(pred_multiclass.detach().cpu().numpy())
        all_labels_glaucoma.extend(label_glaucoma.detach().cpu().numpy())

        loss.backward()
        optimizer.step()
    
    epoch_loss = train_running_loss / counter
    epoch_acc_me = correct_me / len(trainloader.dataset)
    epoch_acc_dr = correct_dr / len(trainloader.dataset)
    epoch_acc_glaucoma = correct_glaucoma / len(trainloader.dataset)

    f1_me = f1_score(all_labels_me, all_preds_me, average='binary', zero_division=0)
    f1_dr = f1_score(all_labels_dr, all_preds_dr, average='binary', zero_division=0)
    f1_glaucoma = f1_score(all_labels_glaucoma, all_preds_glaucoma, average='macro', zero_division=0)

    return epoch_loss , epoch_acc_me, epoch_acc_dr, epoch_acc_glaucoma, f1_me, f1_dr, f1_glaucoma

def validate(model, validloader, criterion_binary, criterion_multiclass, alpha=1, beta=1):
    model.eval()
    print('Validation')

    valid_running_loss = 0.0
    correct_me = 0
    correct_dr = 0
    correct_glaucoma = 0
    counter = 0

    # test
    all_preds_me = []
    all_labels_me = []
    all_preds_dr = []
    all_labels_dr = []
    all_preds_glaucoma = []
    all_labels_glaucoma = []

    with torch.no_grad():
        for i, data in tqdm(enumerate(validloader), total=len(validloader)):
            counter += 1
            
            image, label_me, label_dr, label_glaucoma = data
            image = image.to(device)
            label_me = label_me.to(device)
            label_dr = label_dr.to(device)
            label_glaucoma = label_glaucoma.to(device)

            outputs = model(image)

            # loss from binary
            binary_outputs = outputs[:,:2]
            binary_labels = torch.stack((label_me, label_dr), dim=1).float().to(device)
            loss_binary = criterion_binary(binary_outputs, binary_labels.float())

            # loss from multiclass
            multiclass_outputs = outputs[:, 2:]
            loss_multiclass = criterion_multiclass(multiclass_outputs, label_glaucoma)

            # combine losses
            loss = alpha * loss_binary + beta * loss_multiclass
            valid_running_loss += loss.item()

            # change preds to be label-wise and combine binary + multiclass
            preds_binary = (torch.sigmoid(binary_outputs) > 0.5).float()
            pred_multiclass = torch.argmax(multiclass_outputs, dim=1)
            correct_me += (preds_binary[:, 0] == label_me).sum().item()
            correct_dr += (preds_binary[:, 1] == label_dr).sum().item()
            correct_glaucoma += (pred_multiclass == label_glaucoma).sum().item()

            # testing how to create longer preds arrays to then compare at the end to calculate F1 and accuracy
            all_preds_me.extend(preds_binary[:, 0].detach().cpu().numpy())
            all_labels_me.extend(label_me.detach().cpu().numpy())

            all_preds_dr.extend(preds_binary[:, 1].detach().cpu().numpy())
            all_labels_dr.extend(label_dr.detach().cpu().numpy())

            all_preds_glaucoma.extend(pred_multiclass.detach().cpu().numpy())
            all_labels_glaucoma.extend(label_glaucoma.detach().cpu().numpy())

        
    epoch_loss = valid_running_loss / counter
    epoch_acc_me = correct_me / len(validloader.dataset)
    epoch_acc_dr = correct_dr / len(validloader.dataset)
    epoch_acc_glaucoma = correct_glaucoma / len(validloader.dataset)

    # f1 score
    f1_me = f1_score(all_labels_me, all_preds_me, average='binary', zero_division=0)
    f1_dr = f1_score(all_labels_dr, all_preds_dr, average='binary', zero_division=0)
    f1_glaucoma = f1_score(all_labels_glaucoma, all_preds_glaucoma, average='macro', zero_division=0)

    return epoch_loss, epoch_acc_me, epoch_acc_dr, epoch_acc_glaucoma, f1_me, f1_dr, f1_glaucoma

if __name__ == '__main__':

    df = pd.read_csv("Images/Retina-SLO_dataset/Retina-SLO_labels/Retina-SLO.txt",
                    sep=":|,", engine='python')
    df.rename(columns={'img_path ': 'img_path', ' ME_GT': 'ME', ' DR_GT': 'DR', ' glaucoma_GT': 'glaucoma'}, inplace=True)
    df = df[:20]
    df['glaucoma'] = df['glaucoma'].str.strip("; ")
    df['img_path'] = df['img_path'].str.strip()
    df = df[df['img_path'].str.contains('study1')]
    df.reset_index(drop=True, inplace=True)

    dataset_train, dataset_valid, dataset_test = split_datasets(df, args['pretrained'], root_dir="Images/Retina-SLO_dataset/")
    print(f"[INFO]: Number of training images: {len(dataset_train)}")
    print(f"[INFO]: Number of validation images: {len(dataset_valid)}")
    

    train_loader, valid_loader, test_loader = get_data_loaders(dataset_train, dataset_valid, dataset_test)
    

    lr = args['learning_rate']
    epochs = args['epochs']
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Computation device: {device}")
    print(f"Learning rate: {lr}")
    print(f"Epochs to train for: {epochs}\n")
    
    model = build_model(
        pretrained=args['pretrained'], 
        fine_tune=True
    ).to(device)
    
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_binary = nn.BCEWithLogitsLoss()
    criterion_multiclass = nn.CrossEntropyLoss()

    # early stopping parameters
    min_loss = float('inf')
    epoch_no_improvement = 0
    patience = 3
    best_model_path = f"best_model_pt_{args['pretrained']}.pth"


    train_loss, valid_loss = [], []
    train_acc_me, valid_acc_me = [], []
    train_acc_dr, valid_acc_dr = [], []
    train_acc_glaucoma, valid_acc_glaucoma = [], []

    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_acc_me, train_epoch_acc_dr, train_epoch_acc_glaucoma, train_epoch_f1_me, train_epoch_f1_dr, train_epoch_f1_glaucoma = train(model, train_loader, 
                                                optimizer, criterion_binary, criterion_multiclass)
        valid_epoch_loss, valid_epoch_acc_me, valid_epoch_acc_dr, valid_epoch_acc_glaucoma, valid_epoch_f1_me, valid_epoch_f1_dr, valid_epoch_f1_glaucoma  = validate(model, valid_loader,  
                                                    criterion_binary, criterion_multiclass)
        # for tensorboard
        writer.add_scalar("Loss/train", train_epoch_loss, epoch)
        writer.add_scalar('Loss/validation', valid_epoch_loss, epoch)
        
        writer.add_scalar('Accuracy/ME/train', train_epoch_acc_me, epoch)
        writer.add_scalar('Accuracy/ME/validation', valid_epoch_acc_me, epoch)
        writer.add_scalar('Accuracy/DR/train', train_epoch_acc_dr, epoch)
        writer.add_scalar('Accuracy/DR/validation', valid_epoch_acc_dr, epoch)
        writer.add_scalar('Accuracy/glaucoma/train', train_epoch_acc_glaucoma, epoch)
        writer.add_scalar('Accuracy/glaucoma/validation', valid_epoch_acc_glaucoma, epoch)

        # for own plots
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)

        train_acc_me.append(train_epoch_acc_me)
        valid_acc_me.append(valid_epoch_acc_me)
        train_acc_dr.append(train_epoch_acc_dr)
        valid_acc_dr.append(valid_epoch_acc_dr)
        train_acc_glaucoma.append(train_epoch_acc_glaucoma)
        valid_acc_glaucoma.append(valid_epoch_acc_glaucoma)

        print(f"Training loss: {train_epoch_loss:.3f}, training acc me: {train_epoch_acc_me:.3f}, training acc dr: {train_epoch_acc_dr:.3f}, training acc glaucoma: {train_epoch_acc_glaucoma:.3f}, training f1 me: {train_epoch_f1_me:.3f}, training f1 dr: {train_epoch_f1_dr:.3f}, training f1 glaucoma: {train_epoch_f1_glaucoma:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc me: {valid_epoch_acc_me:.3f}, validation acc dr: {valid_epoch_acc_dr:.3f}, validation acc glaucoma: {valid_epoch_acc_glaucoma:.3f}, validation f1 me: {valid_epoch_f1_me:.3f}, validation f1 dr: {valid_epoch_f1_dr:.3f}, validation f1 glaucoma: {valid_epoch_f1_glaucoma:.3f} ")
        print('-'*50)

        # early stopping
        if valid_epoch_loss < min_loss:
            min_loss = valid_epoch_loss
            epoch_no_improvement = 0
            torch.save(model.state_dict(), best_model_path)   # save the model in case it is best
        else:
            epoch_no_improvement += 1
        if epoch_no_improvement >= patience:
            break
    
    writer.flush()
    model.load_state_dict(torch.load(best_model_path))
        
    save_model_onnx(model, (3,224,224) ,args['pretrained'])
    save_feature_model_onnx(model, (3,224,224) ,args['pretrained'])
    save_plots(train_acc_me, valid_acc_me, train_acc_dr, valid_acc_dr, train_acc_glaucoma, valid_acc_glaucoma, train_loss, valid_loss, args['pretrained'])
