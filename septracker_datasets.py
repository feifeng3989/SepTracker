import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from tracking_toolbox import *

class SeptrackerObjectDataset(Dataset):
    def __init__(self, cfg, data_root, mode='train'):
        super(SeptrackerObjectDataset, self).__init__()
        det_path = os.path.join(cfg.DATA.DET_ROOT, '{}.txt'.format(data_root))
        gt_path = os.path.join(cfg.DATA.GT_ROOT, '{}.txt'.format(data_root))
        self.dets = pd.read_csv(det_path, header=None).values.tolist()
        self.dets = self.transform_det_list(self.dets)
        self.mode = mode
        if self.mode == 'train':
            self.gts = pd.read_csv(gt_path, header=None).values.tolist()
            self.gts = self.transform_list(self.gts)
        else:
            self.gts = []

    def __len__(self):
        if self.mode == 'train':
            return min(len(self.dets), len(self.gts)) - 1
        else:
            return len(self.dets) - 1

    def transform_list(self, input_list):
        max_frame = int(max(item[0] for item in input_list) if input_list else 0)
        new_list = [[] for _ in range(max_frame + 1)]

        for item in input_list:
            frame, main_id, frag_id, x, y = item[:5]
            new_list[int(frame)].append([main_id, frag_id, x, y])

        return new_list

    def transform_det_list(self, input_list):
        input_list = sorted(input_list, key=lambda x: (x[0], x[1]))
        max_frame = int(max(item[0] for item in input_list) if input_list else 0)
        new_list = [[] for _ in range(max_frame + 1)]

        for item in input_list:
            frame, x, y = item
            new_list[int(frame)].append([x, y])

        return new_list

    def __getitem__(self, index):
        if self.mode == 'train':
            gt = self.gts[index]
            for i in range(len(gt)):
                gt[i][0] = int(gt[i][0])
        else:
            gt = []

        det = self.dets[index] if index < len(self.dets) else []

        return det, gt

class GroundTruthTracks(Dataset):
    
    def __init__(self, video_names, cfg):
        super(GroundTruthTracks, self).__init__()
        self.videos = video_names
        self.cfg = cfg
        self.gt_tracks = self.generate_tracks()

    def __getitem__(self, item):
        return self.gt_tracks[item]

    def __len__(self):
        return len(self.videos)

    def generate_tracks(self):
        gt_tracks = {}
        for video in self.videos:
            frame_data = SeptrackerObjectDataset(self.cfg, video, mode='train')
            gt_track = [[] for _ in range(self.cfg.PARAM.MAX_TRACK)]

            has_frag3 = set()
            has_frag4 = set()
            has_frag5 = set()
            for frame_id in range(1, len(frame_data) + 1):
                _, gts_scan = frame_data[frame_id]
                for gt in gts_scan:
                    main_id, frag_id, _, _ = gt
                    if int(frag_id) == 3:
                        has_frag3.add(int(main_id))
                    if int(frag_id) == 4:
                        has_frag4.add(int(main_id))
                    if int(frag_id) == 5:
                        has_frag5.add(int(main_id))

            for frame_id in range(1, len(frame_data) + 1):
                _, gts = frame_data[frame_id]
                for gt in gts:
                    main_id, frag_id, x, y = gt
                    track_id = int(f'{int(main_id)}{int(frag_id):02d}')

                    frag_1 = int(f'{int(main_id)}{1:02d}')
                    frag_2 = int(f'{int(main_id)}{2:02d}')
                    frag_3 = int(f'{int(main_id)}{3:02d}')
                    frag_4 = int(f'{int(main_id)}{4:02d}')
                    frag_5 = int(f'{int(main_id)}{5:02d}')

                    if int(frag_id) == 0:
                        gt_track[track_id].append([frame_id, x, y])
                        gt_track[frag_1].append([frame_id, x, y])
                        gt_track[frag_2].append([frame_id, x, y])
                        if int(main_id) in has_frag3:
                            gt_track[frag_3].append([frame_id, x, y])
                        if int(main_id) in has_frag4:
                            gt_track[frag_4].append([frame_id, x, y])
                        if int(main_id) in has_frag5:
                            gt_track[frag_5].append([frame_id, x, y])

                    else:
                        gt_track[track_id].append([frame_id, x, y])

            gt_tracks.update({video: gt_track})
        return gt_tracks

class SeptrackerTrackDataset(Dataset):
    def __init__(self, pred_track_list, det_track_list, track_positions, detections, det_group_result, cfg):
        super(SeptrackerTrackDataset, self).__init__()
        self.pred_track_list = pred_track_list
        self.det_track_list = det_track_list
        self.cfg = cfg
        device = self.cfg.LOAD.DEV
        if device == 'cpu':
            self.track_positions = track_positions
            self.detections = detections
            self.det_group_result = det_group_result
        else:
            self.track_positions = track_positions.to(device)
            self.detections = detections.to(device)
            self.det_group_result = det_group_result.to(device)

    def __len__(self):
        return len(self.pred_track_list)

    def __getitem__(self, item):
        pred_track_list_end = self.pred_track_list[item].index(-10)
        det_track_list_end = self.det_track_list[item].index(-10)

        end_positions = [pred_track_list_end, det_track_list_end]

        return self.pred_track_list[item], self.det_track_list[item], \
               self.track_positions[item], self.detections[item], self.det_group_result[item], end_positions
