# attack_mode: str. 'global' or 'patch'
attack_mode = 'patch'

# object_vanish_only: bool. If True, the attack goal is to make objects vanish only.
# Otherwise, the attack goal is to make objects vanish, misclassify objects and generate false objects in the background.
object_vanish_only = True

# use_detector_train_pipeline: bool. Whether to use the detector training pipeline
# provided in the detector config file as the training pipeline for attacking.
use_detector_train_pipeline = True

# epochs: int. Training epochs to train the adversarial patches.
epochs = 100

# batch_size: int. Batch size for each GPU.
batch_size = 4

# amp: bool. Whether to use 'Automatic Mixed Precision'
amp = False

# optimizer: a dict with following attributes:
## type: str. Optimizer name. See torch.optim for details.
## kwargs: dict. Corresponding kwargs.
optimizer = dict(type='Adam', kwargs=dict(lr=0.02))

# lr_scheduler: a dict with following attributes:
## type: str. Learning rate scheduler name. See ares.attack.detection.custom.lr_schedule for details.
## kwargs: dict. Corresponding kwargs.
lr_scheduler = dict(type='MultiStepLR',
                    kwargs=dict(verbose=True,milestones=[40, 70])
                    )

# auto_lr_scaler:  a dict with following attributes:
# base_batch_size: int. Corresponding batch size for the learning rate in optimizer.
# The real learning rate will be scaled linearly based on the real batch_size and base_batch_size.
auto_lr_scaler = dict(base_batch_size=8)

# detector: a dict with following attributes:
# cfg_file: str. Path to your mmdet-style detector config file.
# weight_file: str. Path to your detector weight file.
detector = dict(cfg_file='path_to_your_detector_config_file',
                weight_file='path_to_your_detector_weight_file')

# patch:  a dict with following attributes:
# size: int. Size of the square adversarial patches.
# init_mode: str. Supported values: 'gray', 'white', 'black' and 'random'.
# save_period: int. Period for saving adversarial patches.
# save_folder: str. Folder to save adversarial patches.
# resume_path: str. Path to resumed patches.
# resume_all: bool. Ignore patch and applier setting and inherit them from checkpoint by force,
# only work when resume_path is not empty
patch = dict(size=200,
             init_mode='gray',
             save_period=1, # 30
             save_folder='saved_patch',
             resume_path='',
             resume_all=True
             )

# loss_fn: a dict with following attributes:
# tv_loss: dict. Whether to enable and its parameters.
# excluded_losses: list. It specifies detector losses not be used to attack.
loss_fn = dict(tv_loss=dict(enable=True, tv_scale=2.5, tv_thresh=0.1),
               excluded_losses=['loss_bbox', 'loss_iou', 'loss_xy', 'loss_wh']
               )

# attacked_classes: list or None. If None, all classes will be attacked.
# if list, only classes in it will be attacked.
attacked_classes = None

# adv_image: a dict with following attributes:
# save: bool. Whether to save adversarial images. Suggested value is False to avoid frequently saving amount of images at each evaluation
# save_folder: str.
# with_bboxes: bool. Whether to save adversarial images with bboxes drawn.
adv_image = dict(save=False, save_folder='adv_images', with_bboxes=True)

# clean_image: a dict with followinng attributes:
# save: bool. Whether to save the original clean images.
# save_folder: str.
# with_bboxes: bool. Whether to save adversarial images with bboxes drawn.
clean_image = dict(save=False, save_folder='clean_images', with_bboxes=True)

# eval_period: int. Period to evaluate detection performance on attacked images.
eval_period = 30 # 30

# log_period: int. Period to log information.
log_period = 20 # 20
