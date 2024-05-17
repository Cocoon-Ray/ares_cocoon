import argparse
import random
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.models import resnet50,ResNet50_Weights
from ares.dataset import ImageFolder
from ares.attack.extraction import extract_model, extract_from_offline_preds

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--data_dir", type=str, default='/home/xieliuwei/project/ares_cocoon/classification/imagenet/', help="Directory to storage data.")
    # save
    parser.add_argument("--save_dir", type=str, default='/home/xieliuwei/project/ares_cocoon/classification/ckpt', help="Directory to save stolen model.")
    
    # attack and model
    parser.add_argument('--criterion', type=str, default='soft', choices=['soft', 'soft_hard', 'hard'], help='The way to compute loss')
    parser.add_argument('--active_query', type=str, default='all', choices=['all', 'hard', 'fgsm', 'pgd', 'bim', 'tim'], help='The way to select or create the samples to query')
    args = parser.parse_args()

    data_dir = args.data_dir
    save_dir = args.save_dir
    criterion = args.criterion
    active_query = args.active_query

    data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    dataset = ImageFolder(data_dir, data_transforms)
    dataset_sizes = len(dataset)
    train_num = int(dataset_sizes * 0.8)
    idx_list = list(range(dataset_sizes))
    random.shuffle(idx_list)
    train_idx = idx_list[:train_num]
    val_idx = idx_list[train_num:]

    train_dataset = Subset(dataset, train_idx)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_dataset = Subset(dataset, val_idx)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)


    # model
    target_model=resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model=resnet50()

    # optim
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 90], gamma=0.1)


    extract_model(model, target_model, train_dataloader, val_dataloader, optimizer, exp_lr_scheduler, save_dir, criterion=criterion, active_query=active_query, epochs_per_eval=1, num_epochs=100)
    # extract_from_offline_preds(model, train_dataloader, val_dataloader, optimizer, exp_lr_scheduler, criterion=nn.CrossEntropyLoss(), save_dir=save_dir)
    # vis_pair_model(model, target_model, train_dataloader, gpu=0, vis_title='result', save_dir=None)