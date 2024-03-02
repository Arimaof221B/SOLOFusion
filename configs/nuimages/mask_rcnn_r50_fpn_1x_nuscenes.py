_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/schedules/mmdet_schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=10), mask_head=dict(num_classes=10)))


dataset_type = 'NuScenesDataset2D'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
data_config={
    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
    'Ncams': 6,
    'input_size': (256, 704),
    # 'input_size': (900, 1600),
    'src_size': (900, 1600),

    # Augmentation
    'resize': (0, 0),
    'rot': (0, 0),
    'flip': False,
    'crop_h': (0.0, 0.0),
    'resize_test':0.0,
}
grid_config={
    'xbound': [-51.2, 51.2, 0.8],
    'ybound': [-51.2, 51.2, 0.8],
    'zbound': [-10.0, 10.0, 20.0],
    'dbound': [2.0, 58.0, 0.5],}

voxel_size = [0.1, 0.1, 0.2] # For CenterHead

train_sequences_split_num = 2
test_sequences_split_num = 1
filter_empty_gt = False

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
    dict(type='Collect3D', keys=['img'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles_BEVDet', data_config=data_config),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='ExtractImages'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1600, 900),
        flip=False,
        transforms=[
            # dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            # dict(type='Normalize', **img_norm_cfg),
            # dict(type='Pad', size_divisor=32),
            dict(type='Collect3D', keys=['img'], meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                                                            'depth2img', 'cam2img', 'pad_shape',
                                                            'scale_factor', 'flip', 'pcd_horizontal_flip',
                                                            'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                                                            'img_norm_cfg', 'pcd_trans', 'sample_idx',
                                                            'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                                                            'transformation_3d_flow', 'cam_sweep_ids',
                                                            'sequence_group_idx', 'curr_to_prev_lidar_rt',
                                                            'start_of_sequence', 'index', 'global_to_curr_lidar_rt',
                                                            'prev_lidar_to_global_rt', 'sample_index', 'ori_filename', 
                                                            'sequence_group_idx', 'flip_direction')),
        ])
    # dict(
    #     type='MultiScaleFlipAug3D',
    #     img_scale=(1333, 800),
    #     pts_scale_ratio=1,
    #     flip=False,
    #     transforms=[
    #         dict(
    #             type='DefaultFormatBundle3D',
    #             class_names=class_names,
    #             with_label=False),
    #         dict(type='Collect3D', keys=['img'])
    #     ])
]


# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(type='LoadMultiViewImageFromFiles_BEVDet', data_config=data_config),
    dict(type='ExtractImages'),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['img'])
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
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
    val=dict(type=dataset_type,
             pipeline=test_pipeline, 
             classes=class_names,
             test_mode=True,
             ann_file=data_root + 'nuscenes_infos_val.pkl',
             modality=input_modality, 
             img_info_prototype='bevdet',
             use_sequence_group_flag=True,
             sequences_split_num=test_sequences_split_num),
    test=dict(type=dataset_type,
              pipeline=test_pipeline, 
              classes=class_names,
              test_mode=True,
              ann_file=data_root + 'nuscenes_infos_val.pkl',
              modality=input_modality,
              img_info_prototype='bevdet',
              use_sequence_group_flag=True,
              sequences_split_num=test_sequences_split_num))



