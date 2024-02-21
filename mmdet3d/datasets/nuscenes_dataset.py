# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch
import math
import numpy as np
import pyquaternion
import tempfile
from nuscenes.utils.data_classes import Box as NuScenesBox
from os import path as osp
import copy
from collections import defaultdict

from mmdet.datasets import DATASETS
from ..core import show_result
from ..core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes
from .custom_3d import Custom3DDataset
from .pipelines import Compose
from ..utils import nuscenes_get_rt_matrix

import os
import time
import json
from pyquaternion import Quaternion
from typing import List, Dict, Tuple, Callable
from nuscenes.eval.detection.evaluate import DetectionEval
from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import load_prediction, load_gt, add_center_dist, filter_eval_boxes
from nuscenes.eval.detection.algo import calc_ap, calc_tp
from nuscenes.eval.detection.constants import TP_METRICS
from nuscenes.eval.detection.data_classes import DetectionConfig, DetectionMetrics, DetectionBox, \
    DetectionMetricDataList, DetectionMetricData
from nuscenes.eval.common.utils import center_distance, scale_iou, yaw_diff, velocity_l2, attr_acc, cummean
from nuscenes.eval.detection.render import summary_plot, class_pr_curve, class_tp_curve, dist_pr_curve, visualize_sample

def add_points_per_box(points_per_box_file_path, gt_boxes):
    import pickle
    with open(points_per_box_file_path, 'rb') as f:
        points_per_box = pickle.load(f)
    assert len(points_per_box) == len(gt_boxes)
    for sample_token in gt_boxes.sample_tokens:
        # print(len(points_per_box[sample_token]['points_per_box_lidar']), len(gt_boxes[sample_token]))
        assert len(points_per_box[sample_token]['points_per_box_lidar']) == len(gt_boxes[sample_token])
        for i, box in enumerate(gt_boxes[sample_token]):
            box.points = {'points_per_box_lidar': points_per_box[sample_token]['points_per_box_lidar'][i], \
                'points_per_box_global': points_per_box[sample_token]['points_per_box_global'][i]}
    return gt_boxes

def load_prediction(result_path: str, max_boxes_per_sample: int, box_cls, verbose: bool = False) \
        -> Tuple[EvalBoxes, Dict]:
    """
    Loads object predictions from file.
    :param result_path: Path to the .json result file provided by the user.
    :param max_boxes_per_sample: Maximim number of boxes allowed per sample.
    :param box_cls: Type of box to load, e.g. DetectionBox or TrackingBox.
    :param verbose: Whether to print messages to stdout.
    :return: The deserialized results and meta data.
    """

    # Load from file and check that the format is correct.
    with open(result_path) as f:
        data = json.load(f)
    assert 'results' in data, 'Error: No field `results` in result file. Please note that the result format changed.' \
                              'See https://www.nuscenes.org/object-detection for more information.'

    # Deserialize results and get meta data.
    all_results = EvalBoxes.deserialize(data['results'], box_cls)
    # meta = data['meta']
    meta = None
    if verbose:
        print("Loaded results from {}. Found detections for {} samples."
              .format(result_path, len(all_results.sample_tokens)))

    # Check that each sample has no more than x predicted boxes.
    for sample_token in all_results.sample_tokens:
        assert len(all_results.boxes[sample_token]) <= max_boxes_per_sample, \
            "Error: Only <= %d boxes per sample allowed!" % max_boxes_per_sample

    return all_results, meta

# def load_gt(nusc: NuScenes, eval_split: str, box_cls, verbose: bool = False, tokens = None) -> EvalBoxes:
#     """
#     Loads ground truth boxes from DB.
#     :param nusc: A NuScenes instance.
#     :param eval_split: The evaluation split for which we load GT boxes.
#     :param box_cls: Type of box to load, e.g. DetectionBox or TrackingBox.
#     :param verbose: Whether to print messages to stdout.
#     :return: The GT boxes.
#     """
    
    
#     # Init.
#     if box_cls == DetectionBox:
#         attribute_map = {a['token']: a['name'] for a in nusc.attribute}

#     if verbose:
#         print('Loading annotations for {} split from nuScenes version: {}'.format(eval_split, nusc.version))
#     # Read out all sample_tokens in DB.
#     sample_tokens_all = [s['token'] for s in nusc.sample]
#     assert len(sample_tokens_all) > 0, "Error: Database has no samples!"

#     # Only keep samples from this split.
#     splits = create_splits_scenes()

#     # Check compatibility of split with nusc_version.
#     version = nusc.version
#     if eval_split in {'train', 'val', 'train_detect', 'train_track'}:
#         assert version.endswith('trainval'), \
#             'Error: Requested split {} which is not compatible with NuScenes version {}'.format(eval_split, version)
#     elif eval_split in {'mini_train', 'mini_val'}:
#         assert version.endswith('mini'), \
#             'Error: Requested split {} which is not compatible with NuScenes version {}'.format(eval_split, version)
#     elif eval_split == 'test':
#         assert version.endswith('test'), \
#             'Error: Requested split {} which is not compatible with NuScenes version {}'.format(eval_split, version)
#     else:
#         raise ValueError('Error: Requested split {} which this function cannot map to the correct NuScenes version.'
#                          .format(eval_split))

#     if eval_split == 'test':
#         # Check that you aren't trying to cheat :).
#         assert len(nusc.sample_annotation) > 0, \
#             'Error: You are trying to evaluate on the test set but you do not have the annotations!'

#     sample_tokens = []
#     if tokens is not None:
#         sample_tokens_all = tokens
#     for sample_token in sample_tokens_all:
#         scene_token = nusc.get('sample', sample_token)['scene_token']
#         scene_record = nusc.get('scene', scene_token)
#         if scene_record['name'] in splits[eval_split]:
#             sample_tokens.append(sample_token)

#     all_annotations = EvalBoxes()

#     # Load annotations and filter predictions and annotations.
#     tracking_id_set = set()
#     for sample_token in tqdm.tqdm(sample_tokens, leave=verbose):

#         sample = nusc.get('sample', sample_token)
#         sample_annotation_tokens = sample['anns']

#         sample_boxes = []
#         for sample_annotation_token in sample_annotation_tokens:

#             sample_annotation = nusc.get('sample_annotation', sample_annotation_token)
#             if box_cls == DetectionBox:
#                 # Get label name in detection task and filter unused labels.
#                 detection_name = category_to_detection_name(sample_annotation['category_name'])
#                 if detection_name is None:
#                     continue

#                 # Get attribute_name.
#                 attr_tokens = sample_annotation['attribute_tokens']
#                 attr_count = len(attr_tokens)
#                 if attr_count == 0:
#                     attribute_name = ''
#                 elif attr_count == 1:
#                     attribute_name = attribute_map[attr_tokens[0]]
#                 else:
#                     raise Exception('Error: GT annotations must not have more than one attribute!')

#                 sample_boxes.append(
#                     box_cls(
#                         sample_token=sample_token,
#                         translation=sample_annotation['translation'],
#                         size=sample_annotation['size'],
#                         rotation=sample_annotation['rotation'],
#                         velocity=nusc.box_velocity(sample_annotation['token'])[:2],
#                         num_pts=sample_annotation['num_lidar_pts'] + sample_annotation['num_radar_pts'],
#                         detection_name=detection_name,
#                         detection_score=-1.0,  # GT samples do not have a score.
#                         attribute_name=attribute_name
#                     )
#                 )
#             elif box_cls == TrackingBox:
#                 # Use nuScenes token as tracking id.
#                 tracking_id = sample_annotation['instance_token']
#                 tracking_id_set.add(tracking_id)

#                 # Get label name in detection task and filter unused labels.
#                 # Import locally to avoid errors when motmetrics package is not installed.
#                 from nuscenes.eval.tracking.utils import category_to_tracking_name
#                 tracking_name = category_to_tracking_name(sample_annotation['category_name'])
#                 if tracking_name is None:
#                     continue

#                 sample_boxes.append(
#                     box_cls(
#                         sample_token=sample_token,
#                         translation=sample_annotation['translation'],
#                         size=sample_annotation['size'],
#                         rotation=sample_annotation['rotation'],
#                         velocity=nusc.box_velocity(sample_annotation['token'])[:2],
#                         num_pts=sample_annotation['num_lidar_pts'] + sample_annotation['num_radar_pts'],
#                         tracking_id=tracking_id,
#                         tracking_name=tracking_name,
#                         tracking_score=-1.0  # GT samples do not have a score.
#                     )
#                 )
#             else:
#                 raise NotImplementedError('Error: Invalid box_cls %s!' % box_cls)

#         all_annotations.add_boxes(sample_token, sample_boxes)

#     if verbose:
#         print("Loaded ground truth annotations for {} samples.".format(len(all_annotations.sample_tokens)))

#     return all_annotations

def filter_eval_boxes_by_tokens(
                      eval_boxes: EvalBoxes,
                      tokens,
                      verbose: bool = False) -> EvalBoxes:
    # Retrieve box type for detectipn/tracking boxes.
    all_annotations = EvalBoxes()

    for sample_token, sample_boxes in eval_boxes.boxes.items():
        if sample_token in tokens:
            all_annotations.add_boxes(sample_token, sample_boxes)

    return all_annotations


class NuScenesEval(DetectionEval):
    def __init__(self,
                 nusc: NuScenes,
                 config: DetectionConfig,
                 result_path: str,
                 eval_set: str,
                 output_dir: str = None,
                 verbose: bool = True):
        """
        Initialize a DetectionEval object.
        :param nusc: A NuScenes object.
        :param config: A DetectionConfig object.
        :param result_path: Path of the nuScenes JSON result file.
        :param eval_set: The dataset split to evaluate on, e.g. train, val or test.
        :param output_dir: Folder to save plots and results to.
        :param verbose: Whether to print to stdout.
        """
        self.nusc = nusc
        self.result_path = result_path
        self.eval_set = eval_set
        self.output_dir = output_dir
        self.verbose = verbose
        self.cfg = config

        # Check result file exists.
        assert os.path.exists(result_path), 'Error: The result file does not exist!'

        # Make dirs.
        self.plot_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        # Load data.
        if verbose:
            print('Initializing nuScenes detection evaluation')
        
        tokens = ['e261f474caa34797b0759aa2ca8a576d', '3947c3c1848e46c3849a1e178e2e728b', 'c0e18beded2d45d29662c2129503d4f7', 'c3286d413f524369ac040fbcd0b0e9b6', 'e7683144f74449309873faf2cb7e76c8', 'a4fa9d49433944d893b977da960fc458', 'd55a20a3225b4e258cbba4dd9e1b597f', '1a0f8e5b0cc248cd832716486455a257', 'f9b825f4020e4da8b6a182160e0b0705', '77d239a5f93b49a38ec0a6185c87147d', 'fdb723ef23d64b47aead300ed1d573f7', '6e49bfd9f21447858938b77180bcb6b8', '4c5a8356ea1741548238807ba41235d3', 'e69a14f90e684e0c8dbcee8336e837ad', 'a77fe48ab6e44959860a1a2b44f7be5e', '160faaad600249d7ab8f95ee44648e6f']
        
        self.pred_boxes, self.meta = load_prediction(self.result_path, self.cfg.max_boxes_per_sample, DetectionBox,
                                                     verbose=verbose)
        # self.gt_boxes = load_gt(self.nusc, self.eval_set, DetectionBox, verbose=verbose, tokens=self.pred_boxes.sample_tokens)
        self.gt_boxes = load_gt(self.nusc, self.eval_set, DetectionBox, verbose=verbose)

        # Add center distances.
        self.pred_boxes = add_center_dist(nusc, self.pred_boxes)
        self.gt_boxes = add_center_dist(nusc, self.gt_boxes)
        self.gt_boxes = add_points_per_box('/media/msc-auto/HDD/dataset/nuscenes/SOLOFusion/points_per_box.pkl', self.gt_boxes)
        self.gt_boxes = filter_eval_boxes_by_tokens(self.gt_boxes, tokens=tokens)
        self.pred_boxes = filter_eval_boxes_by_tokens(self.pred_boxes, tokens=tokens)
        
        print(self.pred_boxes.sample_tokens)

        assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), \
            "Samples in split doesn't match samples in predictions."

        # Filter boxes (distance, points per box, etc.).
        if verbose:
            print('Filtering predictions')
        self.pred_boxes = filter_eval_boxes(nusc, self.pred_boxes, self.cfg.class_range, verbose=verbose)
        if verbose:
            print('Filtering ground truth annotations')
        self.gt_boxes = filter_eval_boxes(nusc, self.gt_boxes, self.cfg.class_range, verbose=verbose)

        self.sample_tokens = self.gt_boxes.sample_tokens

    def evaluate(self) -> Tuple[DetectionMetrics, DetectionMetricDataList]:
        """
        Performs the actual evaluation.
        :return: A tuple of high-level and the raw metric data.
        """
        start_time = time.time()

        # -----------------------------------
        # Step 1: Accumulate metric data for all classes and distance thresholds.
        # -----------------------------------
        if self.verbose:
            print('Accumulating metric data...')
        metric_data_list = DetectionMetricDataList()

        num_cars = 0
        num_1m_10d = 0


        for class_name in self.cfg.class_names:
            for dist_th in self.cfg.dist_ths:
                md, extra_data = accumulate(self.gt_boxes, self.pred_boxes, class_name, self.cfg.dist_fcn_callable, dist_th)

                print("X" * 30)
                print(class_name, dist_th, len(extra_data['add_dist']), extra_data['1m_1d'])
                print("Y" * 30)
                # if dist_th == 4.0 and class_name == 'car':
                #     trans_err = md.trans_err
                #     orient_err = md.orient_err
                #     num_cars += len(trans_err)
                #     num_1m_10d += ((trans_err < 1) & (orient_err < 10)).sum()
                #     print(len(add_dist))
                    # add_dist = md.add_dist
                    # print("K" * 30)
                    # print(add_dist)
                metric_data_list.set(class_name, dist_th, md)

        # -----------------------------------
        # Step 2: Calculate metrics from the data.
        # -----------------------------------
        if self.verbose:
            print('Calculating metrics...')
        metrics = DetectionMetrics(self.cfg)
        for class_name in self.cfg.class_names:
            # Compute APs.
            for dist_th in self.cfg.dist_ths:
                metric_data = metric_data_list[(class_name, dist_th)]
                ap = calc_ap(metric_data, self.cfg.min_recall, self.cfg.min_precision)
                metrics.add_label_ap(class_name, dist_th, ap)

            # Compute TP metrics.
            for metric_name in TP_METRICS:
                metric_data = metric_data_list[(class_name, self.cfg.dist_th_tp)]
                if class_name in ['traffic_cone'] and metric_name in ['attr_err', 'vel_err', 'orient_err']:
                    tp = np.nan
                elif class_name in ['barrier'] and metric_name in ['attr_err', 'vel_err']:
                    tp = np.nan
                else:
                    tp = calc_tp(metric_data, self.cfg.min_recall, metric_name)
                metrics.add_label_tp(class_name, metric_name, tp)

        # Compute evaluation time.
        metrics.add_runtime(time.time() - start_time)

        return metrics, metric_data_list

def cal_add_dist(gt_box, pred_box):
    RT_gt = np.eye(4)
    RT_gt[:3, 3] = gt_box.translation
    RT_gt[:3, :3] = Quaternion(gt_box.rotation).rotation_matrix
    RT_pred = np.eye(4)
    RT_pred[:3, 3] = pred_box.translation
    RT_pred[:3, :3] = Quaternion(pred_box.rotation).rotation_matrix
    gt_points = gt_box.points['points_per_box_global']
    gt_points_4d = np.concatenate((gt_points, np.ones_like(gt_points[:, :1])), axis=-1)
    pred_points = gt_points_4d @ (RT_pred @ np.linalg.inv(RT_gt)).T[:, :3]
    # print(gt_points.shape)
    assert gt_points.shape[0] > 0
    return np.linalg.norm(gt_points - pred_points, axis=1).mean()


def accumulate(gt_boxes: EvalBoxes,
               pred_boxes: EvalBoxes,
               class_name: str,
               dist_fcn: Callable,
               dist_th: float,
               verbose: bool = False) -> DetectionMetricData:
    """
    Average Precision over predefined different recall thresholds for a single distance threshold.
    The recall/conf thresholds and other raw metrics will be used in secondary metrics.
    :param gt_boxes: Maps every sample_token to a list of its sample_annotations.
    :param pred_boxes: Maps every sample_token to a list of its sample_results.
    :param class_name: Class to compute AP on.
    :param dist_fcn: Distance function used to match detections and ground truths.
    :param dist_th: Distance threshold for a match.
    :param verbose: If true, print debug messages.
    :return: (average_prec, metrics). The average precision value and raw data for a number of metrics.
    """
    # ---------------------------------------------
    # Organize input and initialize accumulators.
    # ---------------------------------------------

    # Count the positives.
    npos = len([1 for gt_box in gt_boxes.all if gt_box.detection_name == class_name])
    if verbose:
        print("Found {} GT of class {} out of {} total across {} samples.".
              format(npos, class_name, len(gt_boxes.all), len(gt_boxes.sample_tokens)))

    # For missing classes in the GT, return a data structure corresponding to no predictions.
    if npos == 0:
        return DetectionMetricData.no_predictions()

    # Organize the predictions in a single list.
    pred_boxes_list = [box for box in pred_boxes.all if box.detection_name == class_name]
    pred_confs = [box.detection_score for box in pred_boxes_list]

    if verbose:
        print("Found {} PRED of class {} out of {} total across {} samples.".
              format(len(pred_confs), class_name, len(pred_boxes.all), len(pred_boxes.sample_tokens)))

    # Sort by confidence.
    sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(pred_confs))][::-1]

    # Do the actual matching.
    tp = []  # Accumulator of true positives.
    fp = []  # Accumulator of false positives.
    conf = []  # Accumulator of confidences.

    # match_data holds the extra metrics we calculate for each match.
    match_data = {'trans_err': [],
                'vel_err': [],
                'scale_err': [],
                'orient_err': [],
                'attr_err': [],
                'conf': []}

    extra_data = {
        'add_dist': [], 
        '1m_1d': 0, 
    }
    
    # ---------------------------------------------
    # Match and accumulate match data.
    # ---------------------------------------------

    taken = set()  # Initially no gt bounding box is matched.
    for ind in sortind:
        pred_box = pred_boxes_list[ind]
        min_dist = np.inf
        match_gt_idx = None

        for gt_idx, gt_box in enumerate(gt_boxes[pred_box.sample_token]):

            # Find closest match among ground truth boxes
            if gt_box.detection_name == class_name and not (pred_box.sample_token, gt_idx) in taken:
                this_distance = dist_fcn(gt_box, pred_box)
                if this_distance < min_dist:
                    min_dist = this_distance
                    match_gt_idx = gt_idx

        # If the closest match is close enough according to threshold we have a match!
        is_match = min_dist < dist_th

        if is_match:
            taken.add((pred_box.sample_token, match_gt_idx))

            #  Update tp, fp and confs.
            tp.append(1)
            fp.append(0)
            conf.append(pred_box.detection_score)

            # Since it is a match, update match data also.
            gt_box_match = gt_boxes[pred_box.sample_token][match_gt_idx]

            match_data['trans_err'].append(center_distance(gt_box_match, pred_box))
            match_data['vel_err'].append(velocity_l2(gt_box_match, pred_box))
            match_data['scale_err'].append(1 - scale_iou(gt_box_match, pred_box))

            # Barrier orientation is only determined up to 180 degree. (For cones orientation is discarded later)
            period = np.pi if class_name == 'barrier' else 2 * np.pi
            match_data['orient_err'].append(yaw_diff(gt_box_match, pred_box, period=period))

            match_data['attr_err'].append(1 - attr_acc(gt_box_match, pred_box))
            match_data['conf'].append(pred_box.detection_score)
            if dist_th == 4.0 and class_name == 'car': 
                extra_data['add_dist'].append(cal_add_dist(gt_box_match, pred_box))

        else:
            # No match. Mark this as a false positive.
            tp.append(0)
            fp.append(1)
            conf.append(pred_box.detection_score)

    # Check if we have any matches. If not, just return a "no predictions" array.
    if len(match_data['trans_err']) == 0:
        return DetectionMetricData.no_predictions(), extra_data

    extra_data['1m_1d'] = ((np.array(match_data['trans_err']) <= 1) & (np.array(match_data['orient_err']) <= 5 / 180 * np.pi)).sum() / len(taken)
    # ---------------------------------------------
    # Calculate and interpolate precision and recall
    # ---------------------------------------------

    # Accumulate.
    tp = np.cumsum(tp).astype(float)
    fp = np.cumsum(fp).astype(float)
    conf = np.array(conf)

    # Calculate precision and recall.
    prec = tp / (fp + tp)
    rec = tp / float(npos)

    rec_interp = np.linspace(0, 1, DetectionMetricData.nelem)  # 101 steps, from 0% to 100% recall.
    prec = np.interp(rec_interp, rec, prec, right=0)
    conf = np.interp(rec_interp, rec, conf, right=0)
    rec = rec_interp

    # ---------------------------------------------
    # Re-sample the match-data to match, prec, recall and conf.
    # ---------------------------------------------

    for key in match_data.keys():
        if key == "conf":
            continue  # Confidence is used as reference to align with fp and tp. So skip in this step.
        
        else:
            # For each match_data, we first calculate the accumulated mean.
            tmp = cummean(np.array(match_data[key]))

            # Then interpolate based on the confidences. (Note reversing since np.interp needs increasing arrays)
            match_data[key] = np.interp(conf[::-1], match_data['conf'][::-1], tmp[::-1])[::-1]

    # ---------------------------------------------
    # Done. Instantiate MetricData and return
    # ---------------------------------------------
    detection_metric = DetectionMetricData(recall=rec,
                               precision=prec,
                               confidence=conf,
                               trans_err=match_data['trans_err'],
                               vel_err=match_data['vel_err'],
                               scale_err=match_data['scale_err'],
                               orient_err=match_data['orient_err'],
                               attr_err=match_data['attr_err'])
    # detection_metric.add_dist = match_data.pop('add_dist')

    return detection_metric, extra_data


@DATASETS.register_module()
class NuScenesDataset(Custom3DDataset):
    r"""NuScenes Dataset.

    This class serves as the API for experiments on the NuScenes Dataset.

    Please refer to `NuScenes Dataset <https://www.nuscenes.org/download>`_
    for data downloading.

    Args:
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        data_root (str): Path of dataset root.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        load_interval (int, optional): Interval of loading the dataset. It is
            used to uniformly sample the dataset. Defaults to 1.
        with_velocity (bool, optional): Whether include velocity prediction
            into the experiments. Defaults to True.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes.
            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        eval_version (bool, optional): Configuration version of evaluation.
            Defaults to  'detection_cvpr_2019'.
        use_valid_flag (bool): Whether to use `use_valid_flag` key in the info
            file as mask to filter gt_boxes and gt_names. Defaults to False.
    """
    NameMapping = {
        'movable_object.barrier': 'barrier',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.car': 'car',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.motorcycle': 'motorcycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'movable_object.trafficcone': 'traffic_cone',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck'
    }
    DefaultAttribute = {
        'car': 'vehicle.parked',
        'pedestrian': 'pedestrian.moving',
        'trailer': 'vehicle.parked',
        'truck': 'vehicle.parked',
        'bus': 'vehicle.moving',
        'motorcycle': 'cycle.without_rider',
        'construction_vehicle': 'vehicle.parked',
        'bicycle': 'cycle.without_rider',
        'barrier': '',
        'traffic_cone': '',
    }
    AttrMapping = {
        'cycle.with_rider': 0,
        'cycle.without_rider': 1,
        'pedestrian.moving': 2,
        'pedestrian.standing': 3,
        'pedestrian.sitting_lying_down': 4,
        'vehicle.moving': 5,
        'vehicle.parked': 6,
        'vehicle.stopped': 7,
    }
    AttrMapping_rev = [
        'cycle.with_rider',
        'cycle.without_rider',
        'pedestrian.moving',
        'pedestrian.standing',
        'pedestrian.sitting_lying_down',
        'vehicle.moving',
        'vehicle.parked',
        'vehicle.stopped',
    ]
    # https://github.com/nutonomy/nuscenes-devkit/blob/57889ff20678577025326cfc24e57424a829be0a/python-sdk/nuscenes/eval/detection/evaluate.py#L222 # noqa
    ErrNameMapping = {
        'trans_err': 'mATE',
        'scale_err': 'mASE',
        'orient_err': 'mAOE',
        'vel_err': 'mAVE',
        'attr_err': 'mAAE'
    }
    CLASSES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
               'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
               'barrier')

    def __init__(self,
                 ann_file,
                 pipeline=None,
                 data_root=None,
                 classes=None,
                 load_interval=1,
                 with_velocity=True,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 eval_version='detection_cvpr_2019',
                 use_valid_flag=False,
                 img_info_prototype='mmcv',
                 speed_mode='abs_dis',
                 max_interval=3,
                 min_interval=0,
                 prev_only=False,
                 next_only=False,
                 test_adj = 'prev',
                 fix_direction=False,
                 test_adj_ids=None,
                 use_sequence_group_flag=False,
                 sequences_split_num=1):
        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode)

        self.with_velocity = with_velocity
        self.eval_version = eval_version
        from nuscenes.eval.detection.config import config_factory
        self.eval_detection_configs = config_factory(self.eval_version)
        if self.modality is None:
            self.modality = dict(
                use_camera=False,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )

        self.img_info_prototype = img_info_prototype

        self.speed_mode = speed_mode
        self.max_interval = max_interval
        self.min_interval = min_interval
        self.prev_only = prev_only
        self.next_only = next_only
        self.test_adj = test_adj
        self.fix_direction = fix_direction
        self.test_adj_ids = test_adj_ids
        
        self.use_sequence_group_flag = use_sequence_group_flag
        self.sequences_split_num = sequences_split_num
        # sequences_split_num splits eacgh sequence into sequences_split_num parts.
        if self.test_mode:
            assert self.sequences_split_num == 1
        if self.use_sequence_group_flag:
            self._set_sequence_group_flag() # Must be called after load_annotations b/c load_annotations does sorting.

    def _set_sequence_group_flag(self):
        """
        Set each sequence to be a different group
        """
        res = []

        curr_sequence = 0
        for idx in range(len(self.data_infos)):
            if idx != 0 and len(self.data_infos[idx]['sweeps']) == 0:
                # Not first frame and # of sweeps is 0 -> new sequence
                curr_sequence += 1
            res.append(curr_sequence)

        self.flag = np.array(res, dtype=np.int64)

        if self.sequences_split_num != 1:
            if self.sequences_split_num == 'all':
                self.flag = np.array(range(len(self.data_infos)), dtype=np.int64)
            else:
                bin_counts = np.bincount(self.flag)
                new_flags = []
                curr_new_flag = 0
                for curr_flag in range(len(bin_counts)):
                    curr_sequence_length = np.array(
                        list(range(0, 
                                bin_counts[curr_flag], 
                                math.ceil(bin_counts[curr_flag] / self.sequences_split_num)))
                        + [bin_counts[curr_flag]])

                    for sub_seq_idx in (curr_sequence_length[1:] - curr_sequence_length[:-1]):
                        for _ in range(sub_seq_idx):
                            new_flags.append(curr_new_flag)
                        curr_new_flag += 1

                assert len(new_flags) == len(self.flag)
                assert len(np.bincount(new_flags)) == len(np.bincount(self.flag)) * self.sequences_split_num
                self.flag = np.array(new_flags, dtype=np.int64)

    def get_cat_ids(self, idx):
        """Get category distribution of single scene.

        Args:
            idx (int): Index of the data_info.

        Returns:
            dict[list]: for each category, if the current scene
                contains such boxes, store a list containing idx,
                otherwise, store empty list.
        """
        info = self.data_infos[idx]
        if self.use_valid_flag:
            mask = info['valid_flag']
            gt_names = set(info['gt_names'][mask])
        else:
            gt_names = set(info['gt_names'])

        cat_ids = []
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file)
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        data_infos = data_infos[::self.load_interval]
        self.metadata = data['metadata']
        self.version = self.metadata['version']
        return data_infos

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            timestamp=info['timestamp'] / 1e6,
        )

        if self.modality['use_camera']:
            if self.img_info_prototype == 'mmcv':
                image_paths = []
                lidar2img_rts = []
                for cam_type, cam_info in info['cams'].items():
                    image_paths.append(cam_info['data_path'])
                    # obtain lidar to image transformation matrix
                    lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                    lidar2cam_t = cam_info[
                        'sensor2lidar_translation'] @ lidar2cam_r.T
                    lidar2cam_rt = np.eye(4)
                    lidar2cam_rt[:3, :3] = lidar2cam_r.T
                    lidar2cam_rt[3, :3] = -lidar2cam_t
                    intrinsic = cam_info['cam_intrinsic']
                    viewpad = np.eye(4)
                    viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                    lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                    lidar2img_rts.append(lidar2img_rt)

                input_dict.update(
                    dict(
                        img_filename=image_paths,
                        lidar2img=lidar2img_rts,
                    ))
            elif self.img_info_prototype == 'bevdet':
                input_dict.update(dict(img_info=info['cams']))
            elif self.img_info_prototype == 'bevdet_sequential':
                if info ['prev'] is None or info['next'] is None:
                    adjacent= 'prev' if info['next'] is None else 'next'
                else:
                    if self.prev_only or self.next_only:
                        adjacent = 'prev' if self.prev_only else 'next'
                    elif self.test_mode:
                        adjacent = self.test_adj
                    else:
                        adjacent = np.random.choice(['prev', 'next'])
                if type(info[adjacent]) is list:
                    if self.test_mode:
                        if self.test_adj_ids is not None:
                            info_adj=[]
                            select_id = self.test_adj_ids
                            for id_tmp in select_id:
                                id_tmp = min(id_tmp, len(info[adjacent])-1)
                                info_adj.append(info[adjacent][id_tmp])
                        else:
                            select_id = min((self.max_interval+self.min_interval)//2,
                                            len(info[adjacent])-1)
                            info_adj = info[adjacent][select_id]
                    else:
                        if len(info[adjacent])<= self.min_interval:
                            select_id = len(info[adjacent])-1
                        else:
                            select_id = np.random.choice([adj_id for adj_id in range(
                                min(self.min_interval,len(info[adjacent])),
                                min(self.max_interval,len(info[adjacent])))])
                        info_adj = info[adjacent][select_id]
                else:
                    info_adj = info[adjacent]
                input_dict.update(dict(img_info=info['cams'],
                                       curr=info,
                                       adjacent=info_adj,
                                       adjacent_type=adjacent))

            if self.use_sequence_group_flag:
                input_dict['sample_index'] = index
                input_dict['sequence_group_idx'] = self.flag[index]
                input_dict['start_of_sequence'] = index == 0 or self.flag[index - 1] != self.flag[index]
                # Get a transformation matrix from current keyframe lidar to previous keyframe lidar
                # if they belong to same sequence.
                if not input_dict['start_of_sequence']:
                    input_dict['curr_to_prev_lidar_rt'] = torch.FloatTensor(nuscenes_get_rt_matrix(
                        self.data_infos[index], self.data_infos[index - 1],
                        "lidar", "lidar"))
                    input_dict['prev_lidar_to_global_rt'] = torch.FloatTensor(nuscenes_get_rt_matrix(
                        self.data_infos[index - 1], self.data_infos[index],
                        "lidar", "global")) # TODO: Note that global is same for all.
                else:
                    input_dict['curr_to_prev_lidar_rt'] = torch.eye(4).float()
                    input_dict['prev_lidar_to_global_rt'] = torch.FloatTensor(nuscenes_get_rt_matrix(
                        self.data_infos[index], self.data_infos[index],
                        "lidar", "global"))

                input_dict['global_to_curr_lidar_rt'] = torch.FloatTensor(nuscenes_get_rt_matrix(
                    self.data_infos[index], self.data_infos[index],
                    "global", "lidar"))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
            if self.img_info_prototype == 'bevdet_sequential':
                bbox = input_dict['ann_info']['gt_bboxes_3d'].tensor
                if 'abs' in self.speed_mode:
                    bbox[:, 7:9] = bbox[:, 7:9] + torch.from_numpy(info['velo']).view(1,2).to(bbox)
                if input_dict['adjacent_type'] == 'next' and not self.fix_direction:
                    bbox[:, 7:9] = -bbox[:, 7:9]
                if 'dis' in self.speed_mode:
                    time = abs(input_dict['timestamp'] - 1e-6 * input_dict['adjacent']['timestamp'])
                    bbox[:, 7:9] = bbox[:, 7:9] * time
                input_dict['ann_info']['gt_bboxes_3d'] = LiDARInstance3DBoxes(bbox,
                                                                              box_dim=bbox.shape[-1],
                                                                              origin=(0.5, 0.5, 0.0))
        return input_dict

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos[index]
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info['valid_flag']
        else:
            mask = info['num_lidar_pts'] > 0
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info['gt_velocity'][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d)
        return anns_results

    def _format_bbox(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print('Start to convert detection format...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            boxes = output_to_nusc_box(det, self.data_infos[sample_id],
                                       self.speed_mode, self.img_info_prototype,
                                       self.max_interval, self.test_adj,
                                       self.fix_direction,
                                       self.test_adj_ids)
            sample_token = self.data_infos[sample_id]['token']
            boxes = lidar_nusc_box_to_global(self.data_infos[sample_id], boxes,
                                             mapped_class_names,
                                             self.eval_detection_configs,
                                             self.eval_version)
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                if np.sqrt(box.velocity[0]**2 + box.velocity[1]**2) > 0.2:
                    if name in [
                            'car',
                            'construction_vehicle',
                            'bus',
                            'truck',
                            'trailer',
                    ]:
                        attr = 'vehicle.moving'
                    elif name in ['bicycle', 'motorcycle']:
                        attr = 'cycle.with_rider'
                    else:
                        attr = NuScenesDataset.DefaultAttribute[name]
                else:
                    if name in ['pedestrian']:
                        attr = 'pedestrian.standing'
                    elif name in ['bus']:
                        attr = 'vehicle.stopped'
                    else:
                        attr = NuScenesDataset.DefaultAttribute[name]

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    detection_name=name,
                    detection_score=box.score,
                    attribute_name=attr)
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path

    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        from nuscenes import NuScenes
        # from nuscenes.eval.detection.evaluate import NuScenesEval

        output_dir = osp.join(*osp.split(result_path)[:-1])
        nusc = NuScenes(
            version=self.version, dataroot=self.data_root, verbose=False)
        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        nusc_eval = NuScenesEval(
            nusc,
            config=self.eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=False)
        nusc_eval.main(render_curves=False)

        # record metrics
        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        metric_prefix = f'{result_name}_NuScenes'
        for name in self.CLASSES:
            for k, v in metrics['label_aps'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['label_tp_errors'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['tp_errors'].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}'.format(metric_prefix,
                                      self.ErrNameMapping[k])] = val

        detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']
        detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']
        return detail

    def format_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a \
                dict containing the json filepaths, `tmp_dir` is the temporal \
                directory created for saving json files when \
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        # currently the output prediction results could be in two formats
        # 1. list of dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...)
        # 2. list of dict('pts_bbox' or 'img_bbox':
        #     dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...))
        # this is a workaround to enable evaluation of both formats on nuScenes
        # refer to https://github.com/open-mmlab/mmdetection3d/issues/449
        if not ('pts_bbox' in results[0] or 'img_bbox' in results[0]):
            result_files = self._format_bbox(results, jsonfile_prefix)
        else:
            # should take the inner dict out of 'pts_bbox' or 'img_bbox' dict
            result_files = dict()
            for name in results[0]:
                print(f'\nFormating bboxes of {name}')
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                result_files.update(
                    {name: self._format_bbox(results_, tmp_file_)})
        return result_files, tmp_dir

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['pts_bbox'],
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in nuScenes protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        # result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
        result_files = {'pts_bbox': '/media/msc-auto/HDD/dataset/nuscenes/SOLOFusion/pred_results/pts_bbox/results_nusc.json'}
        # result_files = {'pts_bbox': '/media/msc-auto/HDD/dataset/nuscenes/SOLOFusion/pred_results/pts_bbox/nus_results.json'}
        tmp_dir = None

        if isinstance(result_files, dict):
            results_dict = dict()
            for name in result_names:
                print('Evaluating bboxes of {}'.format(name))
                ret_dict = self._evaluate_single(result_files[name])
            results_dict.update(ret_dict)
        elif isinstance(result_files, str):
            results_dict = self._evaluate_single(result_files)

        if tmp_dir is not None:
            tmp_dir.cleanup()

        if show:
            self.show(results, out_dir, pipeline=pipeline)
        return results_dict

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        pipeline = [
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadPointsFromMultiSweeps',
                sweeps_num=10,
                file_client_args=dict(backend='disk')),
            dict(
                type='DefaultFormatBundle3D',
                class_names=self.CLASSES,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ]
        return Compose(pipeline)

    def show(self, results, out_dir, show=True, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Visualize the results online.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)
        for i, result in enumerate(results):
            if 'pts_bbox' in result.keys():
                result = result['pts_bbox']
            data_info = self.data_infos[i]
            pts_path = data_info['lidar_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            points = self._extract_data(i, pipeline, 'points').numpy()
            # for now we convert points into depth mode
            points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
                                               Coord3DMode.DEPTH)
            inds = result['scores_3d'] > 0.1
            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d'].tensor.numpy()
            show_gt_bboxes = Box3DMode.convert(gt_bboxes, Box3DMode.LIDAR,
                                               Box3DMode.DEPTH)
            pred_bboxes = result['boxes_3d'][inds].tensor.numpy()
            show_pred_bboxes = Box3DMode.convert(pred_bboxes, Box3DMode.LIDAR,
                                                 Box3DMode.DEPTH)
            show_result(points, show_gt_bboxes, show_pred_bboxes, out_dir,
                        file_name, show)


def output_to_nusc_box(detection, info, speed_mode,
                       img_info_prototype, max_interval, test_adj, fix_direction,
                       test_adj_ids):
    """Convert the output to the box class in the nuScenes.

    Args:
        detection (dict): Detection results.

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.

    Returns:
        list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
    """
    box3d = detection['boxes_3d']
    scores = detection['scores_3d'].numpy()
    labels = detection['labels_3d'].numpy()

    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()
    # TODO: check whether this is necessary
    # with dir_offset & dir_limit in the head
    box_yaw = -box_yaw - np.pi / 2

    velocity_all = box3d.tensor[:, 7:9]
    if img_info_prototype =='bevdet_sequential':
        if info['prev'] is None or info['next'] is None:
            adjacent = 'prev' if info['next'] is None else 'next'
        else:
            adjacent = test_adj
        if adjacent == 'next' and not fix_direction:
            velocity_all = -velocity_all
        if type(info[adjacent]) is list:
            select_id = min(max_interval // 2, len(info[adjacent]) - 1)
            # select_id = min(2, len(info[adjacent]) - 1)
            info_adj = info[adjacent][select_id]
        else:
            info_adj = info[adjacent]
        if 'dis' in speed_mode and test_adj_ids is None:
            time = abs(1e-6 * info['timestamp'] - 1e-6 * info_adj['timestamp'])
            velocity_all = velocity_all / time

    box_list = []
    for i in range(len(box3d)):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        velocity = (*velocity_all[i,:], 0.0)
        # velo_val = np.linalg.norm(box3d[i, 7:9])
        # velo_ori = box3d[i, 6]
        # velocity = (
        # velo_val * np.cos(velo_ori), velo_val * np.sin(velo_ori), 0.0)
        box = NuScenesBox(
            box_gravity_center[i],
            box_dims[i],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity)
        box_list.append(box)
    return box_list


def lidar_nusc_box_to_global(info,
                             boxes,
                             classes,
                             eval_configs,
                             eval_version='detection_cvpr_2019'):
    """Convert the box from ego to global coordinate.

    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        classes (list[str]): Mapped classes in the evaluation.
        eval_configs (object): Evaluation configuration object.
        eval_version (str): Evaluation version.
            Default: 'detection_cvpr_2019'

    Returns:
        list: List of standard NuScenesBoxes in the global
            coordinate.
    """
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.rotate(pyquaternion.Quaternion(info['lidar2ego_rotation']))
        box.translate(np.array(info['lidar2ego_translation']))
        # filter det in ego.
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        # Move box to global coord system
        box.rotate(pyquaternion.Quaternion(info['ego2global_rotation']))
        box.translate(np.array(info['ego2global_translation']))
        box_list.append(box)
    return box_list
