import argparse

def get_args():
    argp = argparse.ArgumentParser(description='adversarial examples',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # GENERAL
    argp.add_argument('--device', type = str, default = 'cpu')
    argp.add_argument('--train', action = 'store_true')
    argp.add_argument('--test', action = 'store_true')
    argp.add_argument('--generate', action = 'store_true')

    # DATA
    argp.add_argument('--dataset', type = str, choices = ['cifar10', 'imagenet'], default = 'imagenet')
    argp.add_argument('--save_adv', action = 'store_true')

    # PATH
    argp.add_argument('--root_dir', type = str, default = 'data/')
    argp.add_argument('--imagenet_dir', type = str, default = 'data/imagenet_images')
    argp.add_argument('--adv_dir', type = str, default = 'adv_data/')
    argp.add_argument('--model_dir', type = str, default = 'model_weights/')

    # PARAMETERS
    argp.add_argument('--epochs', type = int, default = 11)
    argp.add_argument('--lr', type = float, default = 0.001)
    argp.add_argument('--batch_size', type = int, default = 100)
    argp.add_argument('--test_size', type = int, default = 10000) # this is only for imagenet.

    # ADVERSARIAL ATTACK
    argp.add_argument('--attack', type = str, choices = ['fgsm', 'flgm', 'fsgm', 'flogm'], default = 'fgsm')
    argp.add_argument('--slope', type = float, default = 0.3)

    return argp.parse_args()



