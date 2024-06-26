import torch
from tqdm import tqdm
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from ...utils.registry import registry
from ...utils.metrics import AverageMeter, accuracy
from ...attack.attack_configs import attack_configs
from ...dataset import ImageFolder


@torch.no_grad()
def evaluate_cifar10(model, data_dir, gpu=0):
    """
    Evaluate model on cifar10 dataset with original clean data

    Args:
    - model (nn.Module): The model to be tested. Taken input, it outputs the result with the shape of (bs, nc) .
    - data_dir (str): The dir of the cifar-10 dataset
    """
    test_transforms = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.CIFAR10(data_dir, train=False, download=False, transform=test_transforms)

    test_loader = DataLoader(
        test_dataset,
        batch_size=8, num_workers=2,
        shuffle=False, pin_memory=True, drop_last=False
    )
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    top1_m = AverageMeter()
    for i, (images, labels) in enumerate(tqdm(test_loader)):
        # load data
        batchsize = images.shape[0]
        images, labels = images.to(device), labels.to(device)

        # clean acc
        with torch.no_grad():
            logits = model(images)

        clean_acc = accuracy(logits, labels)[0]
        top1_m.update(clean_acc.item(), batchsize)

    print(f"Clean accuracy: {round(top1_m.avg, 2)} %")


def evaluate_cifar10_attack(model, data_dir, attack_name, gpu=0):
    """
    Evaluate model on cifar10 dataset with adversarial data 

    Args:
    - model (nn.Module): The model to be tested. Taken input, it outputs the result with the shape of (bs, nc) .
    - data_dir (str): The dir of the cifar-10 dataset
    - attack_name (str): The attack mode supported by ares. You can check the ares.attack.attack_configs and refer to the keys of the config
    """
    test_transforms = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.CIFAR10(data_dir, train=False, download=False, transform=test_transforms)

    test_loader = DataLoader(
        test_dataset,
        batch_size=8, num_workers=2,
        shuffle=False, pin_memory=True, drop_last=False
    )
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    # initialize attacker
    attacker_cls = registry.get_attack(attack_name)
    attack_config = attack_configs[attack_name]
    attacker = attacker_cls(model=model, device=device, **attack_config)
    adv_top1_m = AverageMeter()

    # attack process
    for i, (images, labels) in enumerate(tqdm(test_loader)):
        # load data
        batchsize = images.shape[0]
        images, labels = images.to(device), labels.to(device)

        # robust acc
        adv_images = attacker(images, labels=labels, target_labels=None)
        if attack_name == 'autoattack':
            if adv_images is None:
                adv_acc = 0.0
            else:
                adv_acc = adv_images.size(0) / batchsize * 100
        else:
            with torch.no_grad():
                adv_logits = model(adv_images)
            adv_acc = accuracy(adv_logits, labels)[0]
            adv_acc = adv_acc.item()
        adv_top1_m.update(adv_acc, batchsize)

    print(f"Robust accuracy: {round(adv_top1_m.avg, 2)} %")

imagenet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@torch.no_grad()
def evaluate_with_loader(model, test_loader, gpu=0):
    """
    Evaluate model by a given test dataloader.
    
    Args:
    - model (nn.Module): The model to be tested. Taken input, it outputs the result with the shape of (bs, nc) .
    - test_loader (DataLoader): The source of the data.
    """
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()
    top1_m = AverageMeter()
    for i, (images, labels) in enumerate(test_loader):
        # load data
        batchsize = images.shape[0]
        images, labels = images.to(device), labels.to(device)

        # clean acc
        with torch.no_grad():
            logits = model(images)

        clean_acc = accuracy(logits, labels)[0]
        top1_m.update(clean_acc.item(), batchsize)

    acc = round(top1_m.avg, 2)
    return acc


@torch.no_grad()
def evaluate_dataset(model, data_dir, transform=imagenet_transform, gpu=0):
    """
    Evaluate model on a custom dataset in a specific form with clean data.
    Form:
        data_dir/0/xxx.png
        data_dir/0/xxy.png
        data_dir/0/[...]/xxz.png
        ...
        data_dir/1/123.png
        data_dir/1/nsdf3.png
        data_dir/1/[...]/asd932_.png
    0 and 1 etc are labels of the images the corresponding dir contains.

    Args:
    - model (nn.Module): The model to be tested. Taken input, it outputs the result with the shape of (bs, nc) .
    - data_dir (str): The dir of the cifar-10 dataset
    - transform (transforms.Compose): The transform to perform per image in the dataset
    """
    test_dataset = ImageFolder(root=data_dir, transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=8, shuffle=False, pin_memory=True, num_workers=8)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()
    top1_m = AverageMeter()
    for i, (images, labels) in enumerate(tqdm(test_loader)):
        # load data
        batchsize = images.shape[0]
        images, labels = images.to(device), labels.to(device)

        # clean acc
        with torch.no_grad():
            logits = model(images)

        clean_acc = accuracy(logits, labels)[0]
        top1_m.update(clean_acc.item(), batchsize)

    acc = round(top1_m.avg, 2)
    print(f"Clean accuracy: {acc} %")


def evaluate_dataset_attack(model, data_dir, attack_name, transform=imagenet_transform, gpu=0):
    """
    Evaluate model on a custom dataset in a specific form with adversarial data.
    Form:
        data_dir/0/xxx.png
        data_dir/0/xxy.png
        data_dir/0/[...]/xxz.png
        ...
        data_dir/1/123.png
        data_dir/1/nsdf3.png
        data_dir/1/[...]/asd932_.png
    0 and 1 etc are labels of the images the corresponding dir contains.

    Args:
    - model (nn.Module): The model to be tested. Taken input, it outputs the result with the shape of (bs, nc) .
    - data_dir (str): The dir of the cifar-10 dataset
    - attack_name (str): The attack mode supported by ares. You can check the ares.attack.attack_configs and refer to the keys of the config
    - transform (transforms.Compose): The transform to perform per image in the dataset
    """
    test_dataset = ImageFolder(root=data_dir, transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=8, shuffle=False, pin_memory=True, num_workers=8)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    # initialize attacker
    attacker_cls = registry.get_attack(attack_name)
    attack_config = attack_configs[attack_name]
    attacker = attacker_cls(model=model, device=device, **attack_config)
    adv_top1_m = AverageMeter()

    # attack process
    for i, (images, labels) in enumerate(tqdm(test_loader)):
        # load data
        batchsize = images.shape[0]
        images, labels = images.to(device), labels.to(device)

        # robust acc
        adv_images = attacker(images, labels=labels, target_labels=None)
        if attack_name == 'autoattack':
            if adv_images is None:
                adv_acc = 0.0
            else:
                adv_acc = adv_images.size(0) / batchsize * 100
        else:
            with torch.no_grad():
                adv_logits = model(adv_images)
            adv_acc = accuracy(adv_logits, labels)[0]
            adv_acc = adv_acc.item()
        adv_top1_m.update(adv_acc, batchsize)

    print(f"Robust accuracy: {round(adv_top1_m.avg, 2)} %")
