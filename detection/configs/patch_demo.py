_base_ = './patch/base.py'
batch_size = 2
# detector = dict(
#     cfg_file='src/fcos_r50-dcn-caffe_fpn_gn-head-center-normbbox-centeronreg-giou_1x_coco/fcos_r50-dcn-caffe_fpn_gn-head-center-normbbox-centeronreg-giou_1x_coco.py',
#     weight_file='src/fcos_r50-dcn-caffe_fpn_gn-head-center-normbbox-centeronreg-giou_1x_coco/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_1x_coco-ae4d8b3d.pth'
#     )
detector = dict(
    cfg_file='src/yolox_l_8xb8-300e_coco/yolox_l_8xb8-300e_coco.py',
    weight_file='src/yolox_l_8xb8-300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth'
    )
lr_scheduler = dict(type='MultiStepLR',
                    kwargs=dict(milestones=[40, 70])
                    )

# patch = dict(size=200,
#         upatch=True,
#         resume_path='src/fcos_r50-dcn-caffe_fpn_gn-head-center-normbbox-centeronreg-giou_1x_coco/best-patches.pth')
patch = dict(size=200,
        upatch=True,
        resume_path='src/fcos_r50-dcn-caffe_fpn_gn-head-center-normbbox-centeronreg-giou_1x_coco/upatch@epoch-1.pth',
        save_period=1)
adv_image = dict(save=True, save_folder='adv_images', with_bboxes=True)
clean_image = dict(save=True, save_folder='clean_images', with_bboxes=True)
#attacked_classes = ['person']