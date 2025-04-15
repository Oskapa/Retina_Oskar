# https://debuggercafe.com/transfer-learning-using-efficientnet-pytorch/

import torch
import matplotlib
import matplotlib.pyplot as plt
import torch.onnx

torch.manual_seed(1)

matplotlib.style.use('ggplot')

def save_feature_model_onnx(model, input_size, pretrained, name):
    """
    Function to save the trained model to ONNX format. 
    Will save the model with last classification layer removed to do the feature extraction
    """
    dummy_input = torch.randn(1, *input_size)  # Adjust input size as needed
    onnx_path = f"./outputs/feature_extractor_pretrained_{pretrained}_{name}.onnx"

    features_extractor = torch.nn.Sequential(*list(model.children())[:-1])  # Remove the final classification layer
    
    torch.onnx.export(
        features_extractor, 
        dummy_input, 
        onnx_path, 
        export_params=True,
        opset_version=11,   
        do_constant_folding=True,
        input_names=["input"], 
        output_names=["output"]
    )

def save_model_onnx(model, input_size, pretrained, name):
    """
    Function to save the trained model to ONNX format. 
    Will save the model with last classification layer removed to do the feature extraction
    """
    dummy_input = torch.randn(1, *input_size)  # Adjust input size as needed
    onnx_path = f"./outputs/model_pretrained_{pretrained}_{name}.onnx"

    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        export_params=True,
        opset_version=11,   
        do_constant_folding=True,
        input_names=["input"], 
        output_names=["output"]
    )


def save_plots(train_acc_me, valid_acc_me, train_acc_dr, valid_acc_dr, train_acc_glaucoma, valid_acc_glaucoma, train_loss, valid_loss, pretrained, name):
    """
    Function to save the loss and accuracy plots to disk.
    """                   
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc_me, color='green', linestyle='-',
        label='train accuracy me'
    )
    plt.plot(
        valid_acc_me, color='blue', linestyle='-',
        label='validation accuracy me'
    )
    plt.plot(
        train_acc_dr, color='orange', linestyle='-',
        label='train accuracy dr'
    )
    plt.plot(
        valid_acc_dr, color='red', linestyle='-',
        label='validation accuracy dr'
    )
    plt.plot(
        train_acc_glaucoma, color='purple', linestyle='-',
        label='train accuracy glaucoma'
    )
    plt.plot(
        valid_acc_glaucoma, color='pink', linestyle='-',
        label='validation accuracy glaucoma'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f"./outputs/accuracy_pretrained_{pretrained}_{name}.png")
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-',
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-',
        label='validation loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"./outputs/loss_pretrained_{pretrained}_{name}.png")
