import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import time
from tqdm.auto import tqdm
from model import build_model
from datasets import split_datasets, get_data_loaders, SLODataset, BATCH_SIZE, IMAGE_SIZE
from utils import save_model_onnx, save_plots, save_feature_model_onnx
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from train import train, validate
import optuna
from optuna import Trial
from device import device

random_seed = 1
np.random.seed(random_seed)
torch.manual_seed(1)


def optuna_hp_space(trial, lr=[1e-6, 1e-4], bs = [8,16,32,64]):
    return {
        "learning_rate": trial.suggest_float("learning_rate", lr[0], lr[1], log=True),
        "batch_size": trial.suggest_categorical("batch_size", bs)
    }

def hyper_parameter_search(df, study_name, name):
    db_url = "sqlite:///optuna_study.db"
    study = optuna.create_study(study_name=study_name, storage=db_url, direction='minimize', load_if_exists=True)

    def compute_objective(trial: Trial, df):
        if trial.number >= 15:
            study.stop()

        hp_space = optuna_hp_space(trial)  # define the hyperparam space

        # preparing weighted loss functions
        weight_me = torch.tensor([len(df[df['ME']==0]) / len(df[df['ME']==1])], device=device)
        weight_dr = torch.tensor([len(df[df['DR']==0]) / len(df[df['DR']==1])], device=device)
        criterion_me = nn.BCEWithLogitsLoss(pos_weight=weight_me)
        criterion_dr = nn.BCEWithLogitsLoss(pos_weight=weight_dr)
        weights_multiclass = torch.tensor([len(df['glaucoma'])/len(df[df['glaucoma']==0]), len(df['glaucoma'])/len(df[df['glaucoma']==1]), len(df['glaucoma'])/len(df[df['glaucoma']==2])], device=device)
        criterion_multiclass = nn.CrossEntropyLoss(weight=weights_multiclass)

        # dataset splitting
        batch_size = hp_space['batch_size']
        dataset_train, dataset_valid, dataset_test = split_datasets(df, pretrained=True, root_dir="Images/Retina-SLO_dataset/")
        print(f"[INFO]: Number of training images: {len(dataset_train)}")
        print(f"[INFO]: Number of validation images: {len(dataset_valid)}")

        train_loader, valid_loader, test_loader = get_data_loaders(dataset_train, dataset_valid, dataset_test, batch_size)

        model = build_model(   # build the model
            pretrained=True, 
            fine_tune=True
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=hp_space['learning_rate'])   # instantiate optimizer

        # early stopping parameters
        min_loss = float('inf')
        epoch_no_improvement = 0
        patience = 3

        for epoch in range(20):
            print(f"[INFO]: Epoch {epoch+1} of {20}")
            train_epoch_loss, train_epoch_acc_me, train_epoch_acc_dr, train_epoch_acc_glaucoma, train_epoch_f1_me, train_epoch_f1_dr, train_epoch_f1_glaucoma = train(model, train_loader, 
                                                    optimizer, criterion_me, criterion_dr, criterion_multiclass)
            valid_epoch_loss, valid_epoch_acc_me, valid_epoch_acc_dr, valid_epoch_acc_glaucoma, valid_epoch_f1_me, valid_epoch_f1_dr, valid_epoch_f1_glaucoma  = validate(model, valid_loader,  
                                                        criterion_me, criterion_dr, criterion_multiclass)
            # early stopping
            if valid_epoch_loss < min_loss:
                min_loss = valid_epoch_loss
                epoch_no_improvement = 0
            else:
                epoch_no_improvement += 1
            if epoch_no_improvement >= patience:
                break

        return min_loss # return the validation loss of the best model

    study.optimize(lambda trial: compute_objective(trial, df), n_trials=15) # actual optimisation for lambda see here: https://optuna.readthedocs.io/en/latest/faq.html#how-to-define-objective-functions-that-have-own-arguments

    print(f"Best trial:{study.best_params}, val_loss:{study.best_value}, trial_number:{study.best_trial.number}")

    # after this we take the best lr and bs retrain the model and save it 

    best_lr = study.best_params['learning_rate']
    best_bs = study.best_params['batch_size']

    # preparing weighted loss functions
    weight_me = torch.tensor([len(df[df['ME']==0]) / len(df[df['ME']==1])], device=device)
    weight_dr = torch.tensor([len(df[df['DR']==0]) / len(df[df['DR']==1])], device=device)
    criterion_me = nn.BCEWithLogitsLoss(pos_weight=weight_me)
    criterion_dr = nn.BCEWithLogitsLoss(pos_weight=weight_dr)
    weights_multiclass = torch.tensor([len(df['glaucoma'])/len(df[df['glaucoma']==0]), len(df['glaucoma'])/len(df[df['glaucoma']==1]), len(df['glaucoma'])/len(df[df['glaucoma']==2])], device=device)
    criterion_multiclass = nn.CrossEntropyLoss(weight=weights_multiclass)

    # dataset splitting
    dataset_train, dataset_valid, dataset_test = split_datasets(df, pretrained=True, root_dir="Images/Retina-SLO_dataset/")
    print(f"[INFO]: Number of training images: {len(dataset_train)}")
    print(f"[INFO]: Number of validation images: {len(dataset_valid)}")

    train_loader, valid_loader, test_loader = get_data_loaders(dataset_train, dataset_valid, dataset_test, best_bs)

    model = build_model(   # build the model
        pretrained=True, 
        fine_tune=True
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=best_lr)

    # early stopping parameters
    min_loss = float('inf')
    epoch_no_improvement = 0
    patience = 3
    best_model_path = f"best_model_pt_True.pth"


    train_loss, valid_loss = [], []
    train_acc_me, valid_acc_me = [], []
    train_acc_dr, valid_acc_dr = [], []
    train_acc_glaucoma, valid_acc_glaucoma = [], []

    for epoch in range(20):
        print(f"[INFO]: Epoch {epoch+1} of {20}")
        train_epoch_loss, train_epoch_acc_me, train_epoch_acc_dr, train_epoch_acc_glaucoma, train_epoch_f1_me, train_epoch_f1_dr, train_epoch_f1_glaucoma = train(model, train_loader, 
                                                optimizer, criterion_me, criterion_dr, criterion_multiclass)
        valid_epoch_loss, valid_epoch_acc_me, valid_epoch_acc_dr, valid_epoch_acc_glaucoma, valid_epoch_f1_me, valid_epoch_f1_dr, valid_epoch_f1_glaucoma  = validate(model, valid_loader,  
                                                    criterion_me, criterion_dr, criterion_multiclass)

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
    
    model.load_state_dict(torch.load(best_model_path))
    save_model_onnx(model, (3,IMAGE_SIZE,IMAGE_SIZE) ,True, name)
    save_feature_model_onnx(model, (3,IMAGE_SIZE,IMAGE_SIZE) ,True, name)
    save_plots(train_acc_me, valid_acc_me, train_acc_dr, valid_acc_dr, train_acc_glaucoma, valid_acc_glaucoma, train_loss, valid_loss, True, name)

if __name__ == '__main__':


    study_name = "Efficient_net"
    # dataset preparation
    df = pd.read_csv("Images/Retina-SLO_dataset/Retina-SLO_labels/Retina-SLO.txt",
                    sep=":|,", engine='python')
    df.rename(columns={'img_path ': 'img_path', ' ME_GT': 'ME', ' DR_GT': 'DR', ' glaucoma_GT': 'glaucoma'}, inplace=True)
    #df = df[:20]
    df['glaucoma'] = df['glaucoma'].str.strip("; ")
    df['img_path'] = df['img_path'].str.strip()
    df = df[df['img_path'].str.contains('study1')]
    df = df[df['DR'] != -100]
    df = df[df['ME'] != -100]
    df['glaucoma'] = pd.to_numeric(df['glaucoma'], errors='coerce')   # turn to numeric
    df = df[df['glaucoma'] != -100]
    df.reset_index(drop=True, inplace=True)

    hyper_parameter_search(df, study_name, "efficientnetb0_optuna")