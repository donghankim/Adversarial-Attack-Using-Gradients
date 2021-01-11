import numpy as np
import torch
import torch.nn.functional as F

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






