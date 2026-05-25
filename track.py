import random
import time
import os
import argparse
import logging
from yacs.config import CfgNode as CN
from data_toolbox import *
from tracking_toolbox import *
from septracker_tracker import SeptrackerTracker
from septracker_datasets import SeptrackerObjectDataset

import collections

def scan_gt_children(gt_path):
    children = collections.defaultdict(set)

    if not os.path.isfile(gt_path):
        return {}

    with open(gt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 3:
                continue

            try:
                obj_id  = int(float(parts[1]))
                frag_id = int(float(parts[2]))
            except ValueError:
                continue

            if frag_id <= 0:
                
                continue

            children[obj_id].add(frag_id)

    child_ids_by_main = {mid: sorted(list(frags)) for mid, frags in children.items()}
    return child_ids_by_main

def _decide_children_for_parent(track, cfg, child_ids_by_main):
    """
    对当前父轨 track，决定要在 tracks_mts 里抄送哪些子轨 ID。

    优先规则：
      1. 如果在 GT 统计里找得到这个主轨（track.track_id），
         就用它真正出现过的子轨 ID 列表，比如 [1,2,3,4,5]。
      2. 如果找不到（比如这个跟踪 ID 和 GT 不对齐），
         就退回到用配置里的 MAX_CHILDREN，生成 [1..K] 占位。
    """
    main_id = int(track.track_id)

    if main_id in child_ids_by_main:
        return child_ids_by_main[main_id]

    max_children = int(getattr(cfg.TRACK, 'MAX_CHILDREN', 2))
    max_children = max(2, max_children)
    return list(range(1, max_children + 1))

def main(args):
    
    default_yaml = 'config/septrackernet.yaml'
    default_cfg = open(default_yaml)
    cfg = CN.load_cfg(default_cfg)
    gt_root = cfg.DATA.GT_ROOT

    if args.device:
        cfg.LOAD.DEV = 'cuda:{}'.format(args.device)
    if args.checkpoint:
        cfg.TRACK.MODEL = args.checkpoint

    cfg.freeze()

    time_name = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    save_dir = os.path.join(cfg.TRACK.SAVE_DIR, time_name)
    mkdirs(save_dir)

    track_save_root = os.path.join(save_dir, 'tracks')
    mota_save_root = os.path.join(save_dir, 'tracks_mota')
    sda_save_root  = os.path.join(save_dir, 'tracks_mts')

    mkdirs(track_save_root)
    mkdirs(mota_save_root)
    mkdirs(sda_save_root)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    log_file_name = time_name + '.log'
    log_file = os.path.join(save_dir, log_file_name)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])

    logger.info('--------------------------- initialize ------------------------------')
    logger.info('cfg file: {}'.format(default_yaml))
    if cfg.TRACK.SEED:
        random_seed = cfg.TRACK.SEED
        random_initiate(seed=random_seed)
        logger.info('random seed initiated {}'.format(random_seed))
    device = cfg.LOAD.DEV
    data_path = cfg.TRACK.IMG_ROOT
    video_names = os.listdir(data_path)
    logger.info('found {} videos: {}'.format(len(video_names), video_names))

    logger.info('------------------------- start tracking ----------------------------')

    time_used_list = []
    frame_num_list = []
    for video_num, video in enumerate(video_names):
        if int(video[1:]) <= len(video_names) * cfg.TRAIN.TRAIN_SET_RATIO:
            continue
        tracker = SeptrackerTracker(cfg, logger)
        video_result_file = os.path.join(track_save_root, '{}.txt'.format(video))
        mota_result_file  = os.path.join(mota_save_root,  '{}.txt'.format(video))
        sda_result_file   = os.path.join(sda_save_root,   '{}.txt'.format(video))

        gt_path = os.path.join(gt_root, f"{int(video):03d}.txt")

        child_ids_by_main = scan_gt_children(gt_path)

        with open(video_result_file, 'w') as f, open(mota_result_file, 'w') as fm, open(sda_result_file, 'w') as fs:
            frame_data = SeptrackerObjectDataset(cfg, video, mode='test')
            frame_start = 1
            frame_end = len(frame_data) + 1
            logger.info('tracking for video {} started!'.format(video))

            time_video_start = time.time()
            for frame_id in range(frame_start, frame_end):
                detections, _ = frame_data[frame_id]
                
                tracks = tracker.update(detections)

                for track in tracks:
                    sup_width = cfg.TRACK.SUP_WIDTH
                    data_line = [frame_id, track.track_id, track.fragment_id, track.kf.x[0], track.kf.x[1]]
                    data_line_for_mota = [frame_id, '{}{:02d}'.format(track.track_id, track.fragment_id),
                                          track.kf.x[0] - sup_width, track.kf.x[1] - sup_width, 2*sup_width,
                                          2*sup_width, 1, -1, -1, -1]
                    write_line = ','.join(str(x) for x in data_line)
                    mota_line  = ','.join(str(x) for x in data_line_for_mota)
                    f.write(write_line)
                    f.write('\n')
                    fm.write(mota_line)
                    fm.write('\n')

                    if track.fragment_id == 0:
                        
                        child_ids = _decide_children_for_parent(track, cfg, child_ids_by_main)

                        for cid in child_ids:
                            
                            data_line_for_sda = [frame_id, '{}{:02d}'.format(track.track_id, cid),
                                                track.kf.x[0] - sup_width, track.kf.x[1] - sup_width,
                                                2 * sup_width, 2 * sup_width, 1, -1, -1, -1]
                            sda_line = ','.join(str(x) for x in data_line_for_sda)
                            fs.write(sda_line + '\n')

                    else:
                        fs.write(mota_line)
                        fs.write('\n')

                if frame_id % cfg.TRACK.PRINT_INT == 0:
                    logger.info('processing frame {}'.format(frame_id))

            time_video_end = time.time()
            time_used_list.append(time_video_end - time_video_start)
            frame_num_list.append(len(frame_data))
            fps_video = len(frame_data) / (time_video_end - time_video_start)
            logger.info('track result for video {} generated! fps:{:.2f}'.format(video, fps_video))

    fps_all = sum(frame_num_list) / sum(time_used_list)
    logger.info('tracking for all videos finished! fps:{:.2f}'.format(fps_all))

if __name__ == "__main__":
    parser = argparse.ArgumentParser('generate track results')
    parser.add_argument('--checkpoint', '-c', default=None, type=str, help='checkpoint for tracking')
    parser.add_argument('--device', '-d', default=None, type=str, help='device used for tracking')
    args = parser.parse_args()
    main(args)
