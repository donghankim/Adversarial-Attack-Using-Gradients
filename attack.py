import numpy as np
import torch
import torch.nn.functional as F
#from torchvision import transforms
#import matplotlib.pyplot as plt
import pdb
from tqdm import tqdm

class Adversarial():
    def __init__(self, config):
        self.config = config

    # basic fast gradient sign method
    def fgsm(self, data, epsilon, grad):
        sign = grad.sign()
        noise = epsilon*sign
        perturbed_img = data + noise
        return perturbed_img

    def flgm(self, data, epsilon, grad):
        sign = grad.sign()
        norm_sign = F.normalize(sign)
        slope = self.config.slope
        noise = torch.maximum(slope*norm_sign, torch.tensor(epsilon))
        perturbed_img = data + noise
        return perturbed_img

    def fsgm(self, data, epsilon, grad):
        sign = grad.sign()
        norm_squared = (F.normalize(sign))**2
        slope = self.config.slope
        noise = torch.maximum(slope*norm_squared, torch.tensor(epsilon))
        perturbed_img = data + noise
        return perturbed_img

    def flogm(self, data, epsilon, grad):
        sign = grad.sign()
        norm_log = torch.log(torch.abs((F.normalize(sign)) + 1))
        slope = self.config.slope
        noise = torch.maximum(slope*norm_log, torch.tensor(epsilon))
        perturbed_img = data + noise
        return perturbed_img


    """
    For generating plots.

    images = [data.squeeze(0).detach(), sign.squeeze(0), noise.squeeze(0), perturbed_img.squeeze(0).detach()]
    self.show_img(images)

    def show_img(self, img, denormalize=True):
        if denormalize:
            for i in range(len(img)):
                img[i] = self.inv_norm(img[i])

        fig, ax = plt.subplots(1,4, figsize = (16,8))
        ax[0].imshow(np.transpose(img[0].numpy(), (1,2,0)))
        ax[0].axis('off')
        ax[0].set_title("Original")
        ax[1].imshow(np.transpose(img[1].numpy(), (1,2,0)))
        ax[1].axis('off')
        ax[1].set_title("Gradients")
        ax[2].imshow(np.transpose(img[2].numpy(), (1,2,0)))
        ax[2].axis('off')
        ax[2].set_title("Noise")
        ax[3].imshow(np.transpose(img[3].numpy(), (1,2,0)))
        ax[3].axis('off')
        ax[3].set_title("FSGM Example")
        plt.show()

        pdb.set_trace()
    """












