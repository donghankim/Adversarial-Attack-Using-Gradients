# Adversarial-Attack-Using-Gradients
Variations of FGSM (fast gradient sign attack) explored for generating adversarial examples.

## Overview
This is a project I undertook in a deep learning class at Korea University. The idea behind the project was to further investigate different methods of tweaking the popular FGSM algorithm to produce better adversarial examples that would cause a trained network to misclassify images. All of the algorithms developed were algorithms that used the internal gradients of the network in attack (white-box methods). A detailed summary of all of the algorithms shown in this repository are available in the report.pdf file.

Our algorithms were tested on the MNIST, CIFAR10 and Imagenet datasets (although the report only showns examples from the MNIST and CIFAR10 dataset).

## Installation & Run
Run the following code to install all of the packages needed to run this repository:
```python
pip install -r requirements.txt
```
In order to generate your own adversarial examples, run the following:
```python
python main.py --generate --attack fgsm
```
Also please note that you need to provide your own data (images). You can download the MNIST and CIFAR10 datasets using the datasets object from torchvision. By default, my code is set to use the CIFAR10 dataset. You can change this from the config.py file, or add arguments before you run the code. Make sure you create a data folder or specifiy a location where you want the image files to be downloaded.

You can choose which attack algorithm you want to run. Furthermore, this will also test the algorihtm to see how well it performs against the network, and return a percentage indicating the number of correct classifications.

I have also uploaded the model parameters for ResNet18. There are two .pth files, one for CIFAR10 and one for Imagenet. These parameters are only trained on original images, and are not models trained for adversarial examples. For those, refer to MadryLab's adversarial examples challenge [here.](https://github.com/MadryLab/cifar10_challenge)

## Examples & Results
Please refer to the report.pdf file for explanation of each algorithm, as well as performance on MNIST and CIFAR10 datasets. To following images show the results of using each attack algorithm. A random image of a bull was selected from the ImageNet dataset to test on. The attack algorithms are responsible for using the gradients to generate the "noise". The adversarial example is simply the original image plus the noise generated. For the examples shown below, the epsilon value was set to 0.05, with a slope value of 0.3. The accuracies reported are from randomly selecting 10000 images from the ImageNet dataset.

For reference, the Resnet model achived <strong>90% accuracy</strong> on the ImageNet dataset, after 11 epochs of training. All attack algorithms should produce a much lower accuracy than 90%.

### FGSM
<img src="images/fgsm.png"
     alt="FGSM results"
     style="text-align:center" />

<strong>Accuracy: 25%</strong>
The FGSM algorithm sets a good benchmark for comparison. You can read the research for FGSM [here.](https://arxiv.org/abs/1412.6572)

### FLGM
<img src="images/flgm.png"
     alt="FLGM results"
     style="text-align:center" />
<strong>Accuracy: 19%</strong>
This is out best performing algorithm. A simple linear operation on the generated gradients produce better results than the FGSM algorithm. Although the improvements are not major, it does go to show that simple manipulations to the FGSM algorithm can produce better results.

### FSGM
<img src="images/fsgm.png"
     alt="FSGM results"
     style="text-align:center" />
<strong>Accuracy: 85%</strong>
We were optimistic to see the results from this algorihtm. Mainly because the adversarial example produced looks almost identical to the input (original) image. However, results were not desirable. With an accuracy of 85%, not only does it not out perfrom the FGSM algorithm, but its performance is almost identical to the accuracy produced from un-tampered images. We assume that because the normalized gradients are squared, we get noise values are too small to make any impact/change on the adversarial example.

### FLOGM
<img src="images/flogm.png"
     alt="FLOGM results"
     style="text-align:center" />
<strong>Accuracy: 87%</strong>
Similar to the FSGM algorithm, the FLOGM algorithm also did not perform very well. Its interesting to see that the FLOGM algorithm perfromed decently on the CIFAR10 dataset, but fails to produce desirable results on the ImageNet dataset. Further investigation could provide a good explanation for this behavior.

## Conclusion
Despite good results on the MNIST and CIFAR10 datasets, most of the algorithms explored did not do well on the ImageNet dataset. Perhapes the higher detail of ImageNet images makes it more difficult for simple gradient based algorithms to perform well on. Regardless, the objective of this project was to explore different methods of tweaking gradient based attacks for generating adversarial examples, and this project achieved just that. Hopefully this project can inspire further developments in creating gradient based adversarial examples.
