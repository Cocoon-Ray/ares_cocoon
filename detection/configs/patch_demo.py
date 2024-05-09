_base_ = './patch/base.py'
batch_size = 2

# detector = dict(
#     cfg_file='src/fcos_r50-dcn-caffe_fpn_gn-head-center-normbbox-centeronreg-giou_1x_coco/fcos_r50-dcn-caffe_fpn_gn-head-center-normbbox-centeronreg-giou_1x_coco.py',
#     weight_file='src/fcos_r50-dcn-caffe_fpn_gn-head-center-normbbox-centeronreg-giou_1x_coco/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_1x_coco-ae4d8b3d.pth'
    # )

# detector = dict(
#     cfg_file='src/yolov3_mobilenetv2_8xb24-ms-416-300e_coco/yolov3_mobilenetv2_8xb24-ms-416-300e_coco.py',
#     weight_file='src/yolov3_mobilenetv2_8xb24-ms-416-300e_coco/yolov3_mobilenetv2_mstrain-416_300e_coco_20210718_010823-f68a07b3.pth'
# )

detector = dict(
    cfg_file='src/yolox_l_8xb8-300e_coco/yolox_l_8xb8-300e_coco.py',
    weight_file='src/yolox_l_8xb8-300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth'
    )
attacked_classes = ['person'] # ['person'], None
patch_applier = dict(
            type = 'LabelBasedPatchApplier',
            attacked_classes = attacked_classes, # special feature
            per_label_per_patch=True, # special feature
            train_transforms=[dict(type='RandomHorizontalFlip', kwargs=dict(p=0.5)),
                              dict(type='MedianPool2d', kwargs=dict(kernel_size=7, same=True)),
                              dict(type='RandomJitter', kwargs=dict(min_contrast=0.8, max_contrast=1.2, min_brightness=-0.1, max_brightness=0.1, noise_factor=0.10)),
                              dict(type='ScalePatchesToBoxes', kwargs=dict(scale_rate=0.2, rand_rotate=True))
                             ],
            test_transforms=[dict(type='MedianPool2d', kwargs=dict(kernel_size=7, same=True)),
                             dict(type='ScalePatchesToBoxes', kwargs=dict(scale_rate=0.2, rand_rotate=False))
                            ]
            )

adv_image = dict(save=True, save_folder='adv_images', with_bboxes=True)
clean_image = dict(save=True, save_folder='clean_images', with_bboxes=True)

eval_period = 1