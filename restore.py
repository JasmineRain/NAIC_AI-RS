import os
import torch


@torch.no_grad()
def restore_models():

    src = "./new_transform_models"
    names = os.listdir(src)

    for name in names:
        if hasattr(torch.load(os.path.join(src, name), map_location='cpu'), 'module'):
            print("yes")
            model = torch.load(os.path.join(src, name), map_location='cpu').module
        else:
            print("no")
            model = torch.load(os.path.join(src, name), map_location='cpu')

        # print(model)
        # img = torch.rand(1, 3, 256, 256)
        # model(img)
        if torch.__version__ == "1.6.0":
            torch.save(model, "./model_backup/%s" % name, _use_new_zipfile_serialization=False)
        else:
            torch.save(model, "./model_backup/%s" % name)
        print(name + " has been restored")


if __name__ == "__main__":
    restore_models()