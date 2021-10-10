import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=16, help='batch_size for train_dataloader and test_dataloader')
parser.add_argument('--label', type=str, default='blues', help='the True label of the image you want to predict')
parser.add_argument('--image_num', type=str, default='0', help='the image number' )
parser.add_argument('--arch', type=str, default='tv', help='the model from torchvision (tv) or handcraft (hc)', choices=['tv', 'hc'])

args = parser.parse_args()