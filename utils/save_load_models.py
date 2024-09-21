import torch

def save_model(checkpoint = None, filename = None):
    if checkpoint is None:
        raise ValueError("Checkpoint can't be none")
    elif filename is None:
        raise ValueError("Filename is not provided")
    else:
        print("Saving Models Checkpoints....")
        torch.save(checkpoint , filename)
        print("Model Checkpoints Saved !")
        

def load_model(model = None , optimizer = None , checkpoint_path = None):
    if checkpoint_path is None:
        raise ValueError("Path can't be none")
    else:
        ckpts = torch.load(checkpoint_path)
        model.load_state_dict(ckpts['model_state_dict'])
        optimizer.load_state_dict(ckpts['optimizer_state_dict'])