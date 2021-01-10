import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from attack import Adversarial
from tqdm import tqdm
import os, pdb, time

class Runner():
    def __init__(self, config, dataset, model):
        self.config = config
        self.dataset = dataset
        self.model = model
        self.model_path = os.path.join(config.model_dir, 'resnet' + f'_{self.config.dataset}.pth')
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.config.lr)
        self.Adv = Adversarial(self.config)

    def load_model_weights(self):
        if os.path.exists(self.model_path):
            if self.config.device == torch.device('cpu'):
                map_location = 'cpu'
            else:
                map_location = lambda storage, loc: storage.cuda()

            self.model.load_state_dict(torch.load(self.model_path, map_location = map_location))
            return True
        else:
            print("No trained model available.")
            return False

    def train(self):
        losses = []
        start_time = time.time()
        for epoch in tqdm(range(self.config.epochs)):
            for images, labels in self.dataset.train_loader:
                y_hat = self.model(images.to(self.config.device))
                loss = self.criterion(y_hat, labels.to(self.config.device))
                losses.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print(f"Loss at epoch{epoch+1}: {losses[-1]}")
        end_time = time.time()
        print(f"Final loss: {losses[-1]} Training Time:{(end_time - start_time)/60}")

        torch.save(self.model.state_dict(), self.model_path)

    # test against image dataset (pure or adv)
    def test(self):
        if self.load_model_weights():
            total_correct = 0
            with torch.no_grad():
                for images, labels in tqdm(self.dataset.test_loader):
                    y_hat = self.model(images.to(self.config.device))
                    loss = self.criterion(y_hat, labels.to(self.config.device))

                    predictions = torch.max(y_hat.data, 1)[1]
                    total_correct += (predictions.to(self.config.device) == labels).sum().item()
                print(f"Test Accuracy: {(total_correct/10000)*100}%")
        else:
            print("Could not test.")

    # generate adversarial examples
    def generate(self):
        if self.load_model_weights():

            adv_path = os.path.join(self.config.adv_dir, (self.config.attack + "/"))
            total_correct = 0
            idx = 1

            for images, labels in tqdm(self.dataset.test_loader):
                images.requires_grad = True
                y_hat = self.model(images.to(self.config.device))
                loss = self.criterion(y_hat, labels.to(self.config.device))
                self.optimizer.zero_grad()
                loss.backward()

                if self.config.attack == 'fgsm':
                    perturbed_images = self.Adv.fgsm(images, 0.05, images.grad.data)
                elif self.config.attack == 'flgm':
                    perturbed_images = self.Adv.flgm(images, 0.05, images.grad.data)
                elif self.config.attack == 'fsgm':
                    perturbed_images = self.Adv.fsgm(images, 0.05, images.grad.data)
                elif self.config.attack == 'flogm':
                    perturbed_images = self.Adv.fsgm(images, 0.05, images.grad.data)

                if self.config.save_adv:
                    torchvision.utils.save_image(self.dataset.inv_norm(perturbed_images), (adv_path + f"{idx}.png"))

                if idx == 1:
                    self.dataset.show_img(images[0].detach())
                    self.dataset.show_img(perturbed_images[0].detach())

                y_hat_adv = self.model(perturbed_images.to(self.config.device))
                pred_adv = torch.max(y_hat_adv.data, 1)[1]
                total_correct += (pred_adv.to(self.config.device) == labels.to(self.config.device)).sum().item()

                idx += 1

            print(f"Test Accuracy: {(total_correct/10000)*100}%")
