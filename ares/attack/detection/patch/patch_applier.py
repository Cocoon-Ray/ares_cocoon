import torch.nn as nn
from .patch_transform import *
from ares.utils.registry import Registry

@Registry.register_patch_applier('LabelBasedPatchApplier')
class LabelBasedPatchApplier(nn.Module):
    """This class transforms adversarial patches and applies them to bboxes.

    Args:
        # TODO
        cfg (mmengine.config.ConfigDict): Configs of adversarial patches.
    """
    def __init__(self, size=200, per_label_per_patch=False, attacked_labels=None, train_transforms=None, test_transforms=None):
        super().__init__()
        self.size = size
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.per_label_per_patch = per_label_per_patch
        self.attacked_labels = attacked_labels
        self.train_transforms = self.build_transforms(training=True)
        self.test_transforms = self.build_transforms(training=False)

    def forward(self, img_batch: torch.Tensor, adv_patch: torch.Tensor, data_samples, training: bool):
        """ This function transforms and applies corresponding adversarial patches for each provided bounding box.

        Args:
            img_batch (torch.Tensor): Batch image tensor. Shape: [N, C=3, H, W].
            adv_patch: Adversarial patch tensor. Shape: [num_clasess, C=3, H, W].
            data_samples: used to filter bboxes and labels based on attacked_labels
            training: Modify transforms based on different phases.
        Returns:
            torch.Tensor: Image tensor with patches applied to. Shape: [N,C,H,W].
        """


        bboxes_list, labels_list = self.get_target_bbox_and_label(data_samples)
        self.unique_labels_related = torch.cat((labels_list)).unique()
        max_num_bboxes_per_image = max([bboxes.shape[0] for bboxes in bboxes_list])

        if max_num_bboxes_per_image == 0:
            # no gt bboxes to apply patches
            return img_batch
        adv_patch_batch, padded_bboxes = self.pad_patches_boxes(adv_patch, bboxes_list, labels_list,
                                                                max_num_bboxes_per_image)
        target_size = img_batch.shape[-2:]  # (H, W)
        max_, min_ = padded_bboxes.max(), padded_bboxes.min()
        if max_ > 1.0001 and min_ >= 0.0:
            bbox_coordinate_mode = 'pixel'
        elif max_ <= 1.0 and min_ >= 0.0:
            bbox_coordinate_mode = 'normed'
        else:
            raise ValueError(f'Not supported bbox coordinate mode. Expected bbox coorninate range [0, 1] or [0, image_size], but got max value {max_}, min value {min_}')
        if bbox_coordinate_mode != 'pixel':
            padded_bboxes[:, :, 0::2] *= target_size[1]
            padded_bboxes[:, :, 1::2] *= target_size[0]

        if training:
            adv_patch_batch = self.train_transforms(adv_patch_batch, padded_bboxes, target_size)
        else:
            adv_patch_batch = self.test_transforms(adv_patch_batch, padded_bboxes, target_size)
        adv_img_batch = self.apply_patch(img_batch, adv_patch_batch)
        return adv_img_batch

    def pad_patches_boxes(self, adv_patch, bboxes_list, labels_list, max_num_bboxes_per_image):
        padded_adv_patches = []
        padded_bboxes = []
        assert adv_patch.ndim == 4
        for i in range(len(bboxes_list)):
            if self.per_label_per_patch:
                patches = adv_patch[labels_list[i]]
            else:
                adv_patch = adv_patch[0].unsqueeze(0)
                patches = adv_patch.repeat(len(labels_list[i]), *[1] * (len(adv_patch.shape) - 1))
            patches = torch.cat((patches, torch.zeros((max_num_bboxes_per_image - patches.shape[0], *patches.shape[1:]),
                                                      device=patches.device)), dim=0)
            bboxes = bboxes_list[i]
            bboxes = torch.cat(
                (bboxes, torch.zeros((max_num_bboxes_per_image - bboxes.shape[0], 4), device=bboxes.device)), dim=0)
            padded_adv_patches.append(patches)
            padded_bboxes.append(bboxes)
        adv_patch_batch = torch.stack(padded_adv_patches)
        padded_bboxes = torch.stack(padded_bboxes)
        return adv_patch_batch, padded_bboxes

    def apply_patch(self, images, adv_patches):
        advs = torch.unbind(adv_patches, 1)
        for adv in advs:
            images = torch.where((adv == 0), images, adv)
        return images

    def build_transforms(self, training=True):
        transforms = []
        transform_pipeline = self.train_transforms if training else self.test_transforms
        transform_pipeline = transform_pipeline if transform_pipeline is not None else []
        for transform in transform_pipeline:
            name = transform['type']
            kwargs = transform['kwargs']
            if name == 'ScalePatchesToBoxes':
                kwargs.update({'size': self.size})
            transforms.append(Registry.get_transform(name)(**kwargs))
        return Compose(transforms)

    def get_target_bbox_and_label(self, data_samples):
        bboxes_list, labels_list = [], []
        for i, data in enumerate(data_samples):
            bboxes = data.gt_instances.bboxes.clone()
            labels = data.gt_instances.labels

            if self.attacked_labels is None:
                bboxes_list.append(bboxes)
                labels_list.append(labels)
            else:
                mask = (labels[:, None] == self.attacked_labels).any(dim=1)
                bboxes_list.append(bboxes[mask])
                labels_list.append(labels[mask])
        return bboxes_list, labels_list

    def get_applied_patch_for_tvloss(self, adv_patch):
        if not self.per_label_per_patch:
            return adv_patch[0]
        else:
            return adv_patch[self.unique_labels_related]
