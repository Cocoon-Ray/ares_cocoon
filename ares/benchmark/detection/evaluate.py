import os
import time
import argparse

import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

import torch
import torch.distributed as dist
from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmengine.registry import MODELS, DefaultScope
from mmengine.evaluator.evaluator import Evaluator
from ares.utils.logger import setup_logger
# from ...attack.detection.trainer import Trainer
# from ares.attack.detection.attacker import UniversalAttacker
from ares.attack.detection.utils import all_reduce, mkdirs_if_not_exists, HiddenPrints
from ...attack.detection.utils import modify_test_pipeline, modify_train_pipeline

from ...attack.detection.attacker import UniversalAttacker


# attack_cfg 内部构建
# def evaluate_global_attack_detection(attack_cfg, detector_cfg):
#     detector_cfg = Config.fromfile(detector_cfg)
#     detector_cfg.test_dataloader.batch_size = 2
#     modify_test_pipeline(detector_cfg)
#
#     if not detector_cfg.model.data_preprocessor.get('mean', False):
#         detector_cfg.model.data_preprocessor.mean = [0.0] * 3
#     if not detector_cfg.model.data_preprocessor.get('std', False):
#         detector_cfg.model.data_preprocessor.std = [1.0] * 3
#
#     detector_cfg.test_dataloader.dataset.filter_cfg = dict(filter_empty_gt=True)
#
#     if detector_cfg.test_dataloader.dataset.get('kept_classes'):
#         del detector_cfg.test_dataloader.dataset.kept_classes
#     if detector_cfg.train_dataloader.dataset.get('kept_classes'):
#         del detector_cfg.test_dataloader.dataset.kept_classes
#     if detector_cfg.test_evaluator.get('specified_classes'):
#         del detector_cfg.test_evaluator.specified_classes
#
#     if torch.cuda.is_available():
#         device = torch.device(0)
#
#     DefaultScope.get_instance('attack', scope_name='mmdet')
#     detector = MODELS.build(detector_cfg.model)
#     detector.eval()
#     test_dataloader = Runner.build_dataloader(detector_cfg.test_dataloader, seed=0, diff_rank_seed=False)
#
#     evaluator = Evaluator(detector_cfg.test_evaluator)
#     evaluator.dataset_meta = test_dataloader.dataset.metainfo
#
#     attacker = UniversalAttacker(attack_cfg, detector, None, device)
#     trainer = Trainer(attack_cfg, attacker, None, test_dataloader, evaluator, None)
#
#     trainer.eval(eval_on_clean=True)
#
#     pass

def evaluate_test_detection(detector_cfg, detector_ckpt):
    detector_cfg = Config.fromfile(detector_cfg)
    detector_cfg.test_dataloader.batch_size = 2
    modify_test_pipeline(detector_cfg)

    if not detector_cfg.model.data_preprocessor.get('mean', False):
        detector_cfg.model.data_preprocessor.mean = [0.0] * 3
    if not detector_cfg.model.data_preprocessor.get('std', False):
        detector_cfg.model.data_preprocessor.std = [1.0] * 3

    detector_cfg.test_dataloader.dataset.filter_cfg = dict(filter_empty_gt=True)

    if detector_cfg.test_dataloader.dataset.get('kept_classes'):
        del detector_cfg.test_dataloader.dataset.kept_classes
    if detector_cfg.train_dataloader.dataset.get('kept_classes'):
        del detector_cfg.test_dataloader.dataset.kept_classes
    if detector_cfg.test_evaluator.get('specified_classes'):
        del detector_cfg.test_evaluator.specified_classes

    DefaultScope.get_instance('attack', scope_name='mmdet')
    detector = MODELS.build(detector_cfg.model)
    state_dict = torch.load(detector_ckpt, map_location='cpu')['state_dict']
    detector.load_state_dict(state_dict, strict=False)

    detector.eval()
    test_dataloader = Runner.build_dataloader(detector_cfg.test_dataloader, seed=0, diff_rank_seed=False)
    test_dataloader.sampler.shuffle = False

    evaluator = Evaluator(detector_cfg.test_evaluator)
    evaluator.dataset_meta = test_dataloader.dataset.metainfo

    if torch.cuda.is_available():
        device = torch.device(0)
    detector = detector.to(device)
    for _, parameter in detector.named_parameters():
        parameter.requires_grad = False

    for i, batch_data in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        with torch.cuda.amp.autocast(enabled=True):
            batch_data = detector.data_preprocessor(batch_data)
            preds = detector.predict(batch_data['inputs'], batch_data['data_samples'])
        evaluator.process(data_samples=preds)
    res = evaluator.evaluate(len(test_dataloader.dataset))
    print(res)
