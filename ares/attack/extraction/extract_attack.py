import os
import copy
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn

from ...benchmark import evaluate_with_loader
from .extract_utils import get_query_inputs, get_distill_loss


def extract_model(model,
                target_model,
                train_dataloader,
                val_dataloader,
                optimizer,
                scheduler,
                save_dir,
                active_query='all',
                criterion='soft',
                gpu=0,
                num_epochs=25,
                iters_per_log=50,
                epochs_per_eval=5,
                exp_name=None):
    """
    This function supports to extract a target model in a real time.

    Args:
    - model (nn.Module): The substitute model to be trained. Taken input, it outputs the result with the shape of (bs, nc).
    There is no need to keep the structure of the substitute model the same as that of the threat model. 
    - target_model (nn.Module): The threat model to be extracted. Taken input, it outputs the result with the shape of (bs, nc).
    - train_dataloader (DataLoader): The source of the data for extraction. It supports two ways, supervised extraction and 
     unsupervised extraction, differentiated by the form of the dataloader. On supervised mode, (image, labels) should be returned 
    in each iteration. On unsupervised mode, only images need to be returned in each iteration. If active_query in ['all', 'hard'], 
    both supervised and unsupervised extraction are supported. Otherwise, only supervised extraction can work.
    - val_dataloader (DataLoader): Test dataloader which contains original test data. (image, labels) should be returned 
    in each iteration.
    - optimizer (optim.Optimizer): optimizer in pytorch.
    - scheduler (optim.Schedular): schedular in pytorch.
    - save_dir (str): The dir to save resultant extraction checkpoint.
    - active_query (str): pick one from ['all', 'hard', 'fgsm', 'pgd', 'bim', 'tim']. 'all' means using all data for quering, while 'hard'
    means using data with high uncertainty for quering. The others mean using adversarial data created by substitute model for quering to
    get high consistency with the  threat model and better robustness.
    - criterion (str): pick one from ['soft', 'hard', 'soft_hard']. The corresponding loss would be loaded for optimization. The value
    'soft' or 'hard' means using soft or hard label of the prediction for extraction respectively. 'soft_hard' means combining the two 
    way for optimization.
    - gpu (str | int): device where computation is performed.
    - num_epochs (int): num of epochs to train.
    - iters_per_log (int): The frequency to refresh the training loss.
    - epochs_per_eval (int): The frequency to perform evaluation.
    - exp_name (str): The name of this experiment. This will affection the name of the final save dir. 
    """

    if exp_name is None:            
        now = datetime.now()
        exp_name = now.strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join(save_dir, exp_name)
    save_path = os.path.join(save_dir, 'best_model.pth')
    os.makedirs(save_dir, exist_ok=True)

    assert active_query in ['all', 'hard', 'fgsm', 'pgd', 'bim', 'tim'], "pick one from ['all', 'hard', 'fgsm', 'pgd', 'bim', 'tim']"
    assert criterion in ['soft', 'soft_hard', 'hard'], "pick one from ['soft', 'soft_hard', 'hard']"
    torch.backends.cudnn.benchmark = True
    if active_query == 'hard':
        torch.backends.cudnn.benchmark = False
    iter_data = next(iter(train_dataloader))
    if not isinstance(iter_data, list):
        assert active_query in ['all', 'hard'], "active_query should only be pick from ['all', 'hard'] if use image-only dataset"

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    target_model = target_model.to(device)
    target_model.eval()

    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        input_count = 0
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=True)
        for idx, data in enumerate(progress_bar, 1):
            try:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
            except:
                inputs = data
                inputs = inputs.to(device)
                labels = None

            inputs = get_query_inputs(active_query, model, inputs, labels)
            input_count += inputs.shape[0]
            with torch.no_grad():    
                target_outputs = target_model(inputs)
                target_pred_id = target_outputs.argmax(dim=-1)
                target_outputs_after_softmax = nn.Softmax(dim=-1)(target_outputs)

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = get_distill_loss(criterion, outputs, target_pred_id, target_outputs_after_softmax)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            iter_mean_loss = running_loss / idx
            if idx % iters_per_log == 0:
                progress_bar.set_postfix({'mean_loss': f'{iter_mean_loss:.4f}'})
        
        scheduler.step()
        epoch_loss = running_loss / input_count
        print(f'Training Loss: {epoch_loss:.4f}')

        if epoch % epochs_per_eval == 0:
            eval_acc = evaluate_with_loader(model, val_dataloader, gpu)
            if eval_acc > best_acc:
                best_acc = eval_acc
                model_for_save = copy.deepcopy(model)
                model_for_save = model_for_save.cpu()

                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.detach_()
                        param.grad = None

                torch.save(model_for_save, save_path)
                print(f'Save at {save_path}.')
            print(f'Evaluation Acc: {eval_acc}')


def extract_from_offline_preds(model,
                                train_dataloader,
                                val_dataloader,
                                optimizer,
                                scheduler,
                                criterion,
                                save_dir,
                                gpu=0,
                                num_epochs=25,
                                iters_per_log=50,
                                epochs_per_eval=1,
                                exp_name=None):
    """
    After storaging the predictive label of the threat model in a specific form. The predictive results are seen as
    true label for extraction attack. This function supports to extract model with predictive label loaded with a 
    dataloader.

    Args:
    - model (nn.Module): The substitute model to be trained. Taken input, it outputs the result with the shape of (bs, nc).
    There is no need to keep the structure of the substitute model the same as that of the threat model. 
    - train_dataloader (DataLoader): The dataloader in which inputs and predictive labels are loaded. (image, labels) 
    should be returned in each iteration.
    - val_dataloader (DataLoader): Test dataloader which contains original test data. (image, labels) should be returned 
    in each iteration.
    - optimizer (optim.Optimizer): optimizer in pytorch.
    - scheduler (optim.Schedular): schedular in pytorch.
    - criterion (Function | Class): a class or function which can be used in such a way: loss = criterion(outputs, labels)
    - save_dir (str): The dir to save resultant extraction checkpoint.
    - gpu (str | int): device where computation is performed.
    - num_epochs (int): num of epochs to train.
    - iters_per_log (int): The frequency to refresh the training loss.
    - epochs_per_eval (int): The frequency to perform evaluation.
    - exp_name (str): The name of this experiment. This will affection the name of the final save dir. 
    """
        
    if exp_name is None:            
        now = datetime.now()
        exp_name = now.strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join(save_dir, exp_name)
    save_path = os.path.join(save_dir, 'best_model.pth')
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        input_count = 0
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=True)
        for idx, (inputs, labels) in enumerate(progress_bar, 1):
            inputs = inputs.to(device)
            labels = labels.to(device)
            input_count += inputs.shape[0]

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            iter_mean_loss = running_loss / idx
            if idx % iters_per_log == 0:
                progress_bar.set_postfix({'mean_loss': f'{iter_mean_loss:.4f}'})
        
        scheduler.step()
        epoch_loss = running_loss / input_count
        print(f'Training Loss: {epoch_loss:.4f}')

        if epoch % epochs_per_eval == 0:
            eval_acc = evaluate_with_loader(model, val_dataloader, gpu)
            if eval_acc > best_acc:
                best_acc = eval_acc
                model_for_save = copy.deepcopy(model)
                model_for_save = model_for_save.cpu()

                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.detach_()
                        param.grad = None

                torch.save(model_for_save, save_path)
                print(f'Save at {save_path}.')
            print(f'Evaluation Acc: {eval_acc} %')