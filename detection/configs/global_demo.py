_base_ = './global/base.py'
batch_size = 2
# detector = dict(
#     cfg_file='src/fcos_r50-dcn-caffe_fpn_gn-head-center-normbbox-centeronreg-giou_1x_coco/fcos_r50-dcn-caffe_fpn_gn-head-center-normbbox-centeronreg-giou_1x_coco.py',
#     weight_file='src/fcos_r50-dcn-caffe_fpn_gn-head-center-normbbox-centeronreg-giou_1x_coco/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_1x_coco-ae4d8b3d.pth'
#     )

detector = dict(
    cfg_file='src/yolox_l_8xb8-300e_coco/yolox_l_8xb8-300e_coco.py',
    weight_file='src/yolox_l_8xb8-300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth'
    )
attack_method = dict(type='pgd',
                     kwargs=dict(
                     eps=3/255,
                     norm='inf')
                     )

adv_image = dict(save=False, save_folder='adv_images', with_bboxes=True)
