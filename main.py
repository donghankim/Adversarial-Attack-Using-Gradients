import torch
from torchvision import models
from config import get_args
from dataset import Cifar10, Imagenet, Single_image
from runner import Runner

import attack

def main():
    config = get_args()
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Dataset = Single_image(config)
    model = models.resnet18(pretrained = False, progress = True).to(config.device)
    runner = Runner(config, Dataset, model)

    if config.train:
        runner.train()
    elif config.test:
        runner.test()
    elif config.generate:
        runner.generate()



if __name__ == "__main__":
    main()


