import torch
import torchvision.transforms as transforms

from utils import seed_everything

DATASET = 'data'
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
seed_everything()
NUM_WORKERS = 2
BATCH_SIZE = 64
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0
NUM_EPOCHS = 1000
PIN_MEMORY = True
LOAD_MODEL = True
SAVE_MODEL = False
TEST_MODE = False
LOAD_MODEL_FILE = "checkpoint.pth.tar"
IMG_DIR = DATASET + "/images/"
LABEL_DIR = DATASET + "/labels/"


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes


transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(), ])
