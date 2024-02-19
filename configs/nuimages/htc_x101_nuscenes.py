_base_ = './htc_r50_fpn_1x_nuim.py'

num_gpus = 4
batch_size = 16
num_iters_per_epoch = int(28130 // (num_gpus * batch_size) * 4.554)
num_epochs = 20
checkpoint_epoch_interval = 4

train_sequences_split_num = 2
test_sequences_split_num = 1

filter_empty_gt = False

# Intermediate Checkpointing to save GPU memory.
with_cp = False

###############################################################################
# High-level Model & Training Details

base_bev_channels = 80

# Long-Term Fusion Parameters
do_history = True
history_cat_num = 16
history_cat_conv_out_channels = 160

# Short-Term Fusion Parameters
do_history_stereo_fusion = True
stereo_out_feats = 64
history_stereo_prev_step = 1
stereo_sampling_num = 7

# BEV Head Parameters
bev_encoder_in_channels = (
    base_bev_channels if not do_history else history_cat_conv_out_channels)

# Loss Weights
depth_loss_weight = 3.0
velocity_code_weight = 1.0

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]
data_config={
    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
    'Ncams': 6,
    'input_size': (256, 704),
    'src_size': (900, 1600),

    # Augmentation
    'resize': (0, 0),
    'rot': (0, 0),
    'flip': False,
    'crop_h': (0.0, 0.0),
    'resize_test':0,
}
grid_config={
    'xbound': [-51.2, 51.2, 0.8],
    'ybound': [-51.2, 51.2, 0.8],
    'zbound': [-10.0, 10.0, 20.0],
    'dbound': [2.0, 58.0, 0.5],}

voxel_size = [0.1, 0.1, 0.2] # For CenterHead

model = dict(
    pretrained='open-mmlab://resnext101_64x4d',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)))

data = dict(samples_per_gpu=1, workers_per_gpu=1)
# learning policy
lr_config = dict(step=[16, 19])
runner = dict(max_epochs=20)

load_from = 'http://download.openmmlab.com/mmdetection/v2.0/htc/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco_20200312-946fd751.pth'  # noqa


dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles_BEVDet', is_train=True, 
         data_config=data_config),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
        update_img2lidar=True),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5,
        update_img2lidar=True),
    dict(type='PointToMultiViewDepth', grid_config=grid_config),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles_BEVDet', data_config=data_config),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img_inputs'])
        ])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(type='LoadMultiViewImageFromFiles_BEVDet', data_config=data_config),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['img_inputs'])
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=batch_size,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        modality=input_modality,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR',
        speed_mode=None,
        max_interval=None,
        min_interval=None,
        prev_only=None,
        fix_direction=None,
        img_info_prototype='bevdet',
        use_sequence_group_flag=True,
        sequences_split_num=train_sequences_split_num,
        filter_empty_gt=filter_empty_gt),
    val=dict(pipeline=test_pipeline, 
             classes=class_names,
             ann_file=data_root + 'nuscenes_infos_val.pkl',
             modality=input_modality, 
             img_info_prototype='bevdet',
             use_sequence_group_flag=True,
             sequences_split_num=test_sequences_split_num),
    test=dict(pipeline=test_pipeline, 
              classes=class_names,
              ann_file=data_root + 'nuscenes_infos_val.pkl',
              modality=input_modality,
              img_info_prototype='bevdet',
              use_sequence_group_flag=True,
              sequences_split_num=test_sequences_split_num))

