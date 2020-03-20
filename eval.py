import numpy as np
from models.model import TL_ResNet50
from utils.utils import get_device, read_parameters, separate_train_val
from models.controller import Controller
from PIL import Image

def load_image(image_name, transformer=None):
    image = Image.open(image_name)
    if transformer == None:
        image = np.array(image).transpose(2, 0, 1)/255
        image = image.astype(np.float32)
    else:
        image = transformer(image)
    return image

def evaluate(model, data):
    out = model(data)
    return out

if __name__ == "__main__":
    configs = read_parameters()
    device = get_device
    ResNet50 = TL_ResNet50(configs["network_parameters"], pretrained = True).to(device)
    Control = Controller(ResNet50)
    Control.load()
    ResNet50.eval()
