import os
import torch

def restore_models():

    src = "./to_be_pseudo"
    names = os.listdir(src)

    for name in names:
        if hasattr(torch.load(os.path.join(src, name), map_location='cpu'), 'module'):
            model = torch.load(os.path.join(src, name), map_location='cpu').module
        else:
            model = torch.load(os.path.join(src, name), map_location='cpu')
        if torch.__version__ == "1.6.0":
            torch.save(model, "./model_backup/%s" % name, _use_new_zipfile_serialization=False)
        else:
            torch.save(model, "./model_backup/%s" % name)
        print(name + " has been restored")


if __name__ == "__main__":
    restore_models()