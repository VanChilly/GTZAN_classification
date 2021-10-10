from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from PIL import Image
from torchvision import transforms
from data.data_transforms import GroupNormalize

def get_dataset(images_path):
    normalize = GroupNormalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        # transforms.RandomRotation(10),
        # transforms.Resize(224),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(p=0.5),
        # transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        normalize,
        ])
    test_transform = transforms.Compose([
        # transforms.Resize(224),
        transforms.ToTensor(),
        normalize,
    ])
    train_dataloader = DataLoader(datasets.ImageFolder(images_path + 'train/', transform=train_transform,), batch_size=16,  shuffle=True, num_workers=8)
    test_dataloader = DataLoader(datasets.ImageFolder(images_path + 'test/', transform=test_transform, ), batch_size=16, shuffle=False, num_workers=8)
    return train_dataloader, test_dataloader

# class GTZAN_Dataset(Dataset):
#     def __init__(self, path) -> None:
#         self.len = self.read_file(path)

#     def __getitem__(self, index):

#         return

#     def __len__(self):
#         return self.len
#     def read_file(self, path):
#         if os.isdir(path):
#             print("Directory {} Exists".format(path))
#             file
