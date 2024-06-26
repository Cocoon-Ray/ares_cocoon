import os
from tqdm import tqdm

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...attack.attack_configs import attack_configs
from ...utils.registry import registry

def abs_outputs_diff_loss(output, target_outputs_after_softmax):
    output_after_softmax = nn.Softmax(dim=-1)(output)
    res = torch.abs(output_after_softmax-target_outputs_after_softmax).sum() / output.shape[0]
    return res

def soft_cross_entropy(output, target_outputs_after_softmax):
    log_likelihood = -F.log_softmax(output, dim=1)
    loss = torch.sum(log_likelihood * target_outputs_after_softmax) / output.shape[0]
    return loss

def get_distill_loss(criterion, outputs, target_pred_id, target_outputs_after_softmax, weights=[0.5, 0.5]):
    hard_criterion = nn.CrossEntropyLoss()
    soft_criterion = abs_outputs_diff_loss
    hard_loss = 0
    soft_loss = 0
    if 'soft' in criterion:
        soft_loss += soft_criterion(outputs, target_outputs_after_softmax)
    if 'hard' in criterion:
        hard_loss += hard_criterion(outputs, target_pred_id)
    loss = soft_loss + hard_loss
    if 'soft' in criterion and 'hard' in criterion:
        loss = soft_loss * weights[0] + hard_loss * weights[1]
    return loss

@torch.no_grad()
def get_hard_inputs(model, inputs, threshold=0.35):
    model.eval()
    outputs = model(inputs)
    logits = nn.Softmax(dim=-1)(outputs)
    hard_inputs = inputs[logits.max(dim=1)[0] < threshold]
    if hard_inputs.shape[0] == 0:
        hard_inputs = inputs[:int(inputs.shape[0]/2)]
    model.train()
    
    return hard_inputs

def get_attack_inputs(active_query, model, inputs, labels):
    model.eval()
    attacker_cls = registry.get_attack(active_query)
    attack_config = attack_configs[active_query]
    attacker = attacker_cls(model=model, device=inputs.device, **attack_config)
    hard_inputs = attacker(images = inputs, labels = labels)
    model.train()

    return hard_inputs

def get_query_inputs(active_query, model, inputs, labels=None):
    if active_query == 'all':
        hard_inputs = inputs
    elif active_query == 'hard':
        hard_inputs = get_hard_inputs(model, inputs)
    else:
        assert labels is not None, "perform attack need labels."
        hard_inputs = get_attack_inputs(active_query, model, inputs, labels)
    return hard_inputs

def vis_pair_model(student_model, teacher_model, dataloader, gpu=0, vis_title='result', save_dir=None):
    """
    This function visualises the difference between the substitute model and threat model.

    Parameters:
    - student_model (nn.Module): substitute model.
    - teacher_model (nn.Module): theat model.
    - dataloader (DataLoader): Visualization requires inference result for combining the distribution of the output.
    Dataloader provides the input data.
    - gpu (str | int): device where computation is performed.
    - vis_title (str): The name of the visualization result for save.
    - save_dir (str): dir for save.
    """
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    student_model = student_model.to(device)
    teacher_model = teacher_model.to(device)
    student_model.eval()
    teacher_model.eval()

    iter_data = next(iter(dataloader))
    with_label = True
    if not isinstance(iter_data, list):
        with_label = False

    student_outputs = []
    teacher_outputs = []
    with torch.no_grad():
        for data in tqdm(dataloader):
            if with_label:
                inputs, _ = data
            else:
                inputs = data
            inputs = inputs.to(device)
            student_outputs.append(student_model(inputs))
            teacher_outputs.append(teacher_model(inputs))

    student_outputs = torch.cat(student_outputs, dim=0).cpu()
    teacher_outputs = torch.cat(teacher_outputs, dim=0).cpu()

    plt.figure(figsize=(10, 5))
    plt.hist(student_outputs.numpy().flatten(), bins=50, alpha=0.8, label='Student')
    plt.hist(teacher_outputs.numpy().flatten(), bins=50, alpha=0.8, label='Teacher')
    plt.legend()
    plt.title('Output Distribution of Student and Teacher Models')
    save = vis_title+'.png'
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save = os.path.join(save_dir, save)

    plt.savefig(save)