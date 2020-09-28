import os
import torch

def restore_models():

    src = "./exp"
    names = os.listdir(src)

    for name in names:
        model = torch.load(os.path.join(src, name), map_location='cpu').module
        if torch.__version__ == "1.6.0":
            torch.save(model, "./ensemble/%s" % name, _use_new_zipfile_serialization=False)
        else:
            torch.save(model, "./ensemble/%s" % name)
        print(name + " has been restored")


if __name__ == "__main__":
    restore_models()