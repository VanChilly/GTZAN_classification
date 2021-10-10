import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torchvision import transforms
from data.data_transforms import GroupNormalize
from args import args

if args.arch == 'tv':
    from torchvision.models import resnet34
    checkpoint_path = 'codes/checkpoints/2021_10_10 13:14:28/ckpt.best.pth.tar'
else:
    from models.resnet import resnet34
    checkpoint_path = 'codes/checkpoints/2021_10_10 10:05:33/ckpt.best.pth.tar'
label = args.label
num = args.image_num
if len(num) == 1:
    num = '0' + num
image_path = 'Data/images_original/' + label + '/' + label + '000' + num + '.png'


img = Image.open(image_path).convert('RGB')
classes = {0:'blues', 1:'classical', 2:'country', 3:'disco', 4:'hiphop', 5:'jazz', 6:'metal', 7:'pop', 8:'reggae', 9:'rock'}
normalize = GroupNormalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])
img = transform(img).unsqueeze(0)
model = resnet34(pretrained=True)
model = nn.DataParallel(model)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
model.eval()
output = model(img)
output = nn.functional.softmax(output, dim=1)
probability, index = torch.max(output, 1)
print('The prediction result of {} is'.format(image_path))
print('Pred:{}, Label:{}, Probability:{:.4f}%'.format(classes[index.item()], label, probability.item() * 100))