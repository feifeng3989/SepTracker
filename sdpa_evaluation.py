import os
import pandas as pd
import numpy as np
from tracking_toolbox import *
from collections import Counter
import time

class SDAEvaluationDataset():
    def __init__(self, track_path, gt_path):
        super(SDAEvaluationDataset, self).__init__()
        self.track_data = pd.read_csv(track_path, header=None).values.tolist()
        self.gt_data = pd.read_csv(gt_path, header=None).values.tolist()
        self.len_for_track = int(max(item[0] for item in self.track_data))
        self.len_for_gt = int(max(item[0] for item in self.gt_data))
        self.len_for_data = max(self.len_for_track, self.len_for_gt)
        self.tracks_by_frame = self.transform_list(self.track_data)
        self.gt_by_frame = self.transform_list(self.gt_data)

    def __getitem__(self, item):
        return self.tracks_by_frame[item], self.gt_by_frame[item]

    def __len__(self):
        return self.len_for_data - 1

    def transform_list(self, input_list):
        
        new_list = [[] for _ in range(self.len_for_data + 1)]

        for item in input_list:
            frame, main_id, frag_id, x, y = item[:5]
            new_list[int(frame)].append([main_id, frag_id, x, y])

        return new_list

    def get_gt_inform(self):
        max_track_for_gt = int(max(item[1] for item in self.gt_data))
        gt_breakpoint_list = [0 for _ in range(max_track_for_gt + 1)]

        for line in self.gt_data:
            frame, main_id, frag_id, x, y = line[:5]
            if frag_id != 0:
                if gt_breakpoint_list[int(main_id)] == 0:
                    gt_breakpoint_list[int(main_id)] = frame

        return max_track_for_gt, gt_breakpoint_list

def euclid_dist(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def find_most_common_element(lst):

    if len(lst) == 0:
        return 0, 0
    counter = Counter(lst)
    most_common_element, count = counter.most_common(1)[0]

    return most_common_element, count

def sdpa_evaluation(cfg, track_result_path, gt_data_path):
    all_track_results = sorted(os.listdir(track_result_path))
    all_gts = sorted(os.listdir(gt_data_path))
    file_list = []
    total_tracks = 0
    total_success = 0
    for track_file in all_track_results:
        for gt_file in all_gts:
            if track_file[-7:-4] == gt_file[-7:-4]:
                file_list.append([track_file, gt_file])

    for file_pair in file_list:
        track_file, gt_file = file_pair
        file_name = track_file[-7:-4]
        track_full_path = os.path.join(track_result_path, track_file)
        gt_full_path = os.path.join(gt_data_path, gt_file)
        eval_data = SDAEvaluationDataset(track_full_path, gt_full_path)

        max_track, gt_breakpoint_list = eval_data.get_gt_inform()
        max_child_id = 0
        for line in eval_data.gt_data:
            
            if len(line) >= 3:
                frag_id = int(line[2])
                if frag_id > max_child_id:
                    max_child_id = frag_id
        
        max_children = max(1, max_child_id)

        track_result_list_main = [[] for _ in range(max_track + 1)]
        
        track_result_list_children = [
            [[] for _ in range(max_children + 1)] for _ in range(max_track + 1)
        ]

        for track_id in range(1, len(track_result_list_main)):
            frame_low = max(1, int(gt_breakpoint_list[track_id]) - cfg.SDA.JUDGE_LEN)
            frame_high = min(int(gt_breakpoint_list[track_id]) + cfg.SDA.JUDGE_LEN, len(eval_data))

            for frame_id in range(frame_low, frame_high):
                gt_main = []
                gt_children = [[] for _ in range(max_children + 1)]
                tracks, gts = eval_data[frame_id]

                for line in gts:
                    main_id, sub_id, x_gt, y_gt = line
                    main_id = int(main_id)
                    sub_id = int(sub_id)
                    if main_id == track_id:
                        if sub_id == 0:
                            gt_main = [x_gt, y_gt]
                        elif 1 <= sub_id <= max_children:
                            gt_children[sub_id] = [x_gt, y_gt]

                if len(gt_main) > 0:
                    min_dist = cfg.SDA.MAX_DIST
                    match_track_id = 0
                    for line in tracks:
                        main_id, sub_id, x_track, y_track = line
                        if int(sub_id) == 0:
                            obj_dist = euclid_dist(gt_main[0], gt_main[1], x_track, y_track)
                            if obj_dist <= min_dist:
                                match_track_id = int(main_id)
                                min_dist = obj_dist

                    track_result_list_main[track_id].append(int(match_track_id))

                for child_id in range(1, max_children + 1):
                    child_pos = gt_children[child_id]
                    if len(child_pos) == 0:
                        continue
                    min_dist = cfg.SDA.MAX_DIST
                    match_track_id = 0
                    for line in tracks:
                        main_id, sub_id, x_track, y_track = line
                        
                        if int(sub_id) != 0:
                            obj_dist = euclid_dist(child_pos[0], child_pos[1], x_track, y_track)
                            if obj_dist <= min_dist:
                                match_track_id = int(main_id)
                                min_dist = obj_dist

                    track_result_list_children[track_id][child_id].append(int(match_track_id))

        success_num = 0

        for track_id in range(1, max_track + 1):
            main_list = track_result_list_main[track_id]
            main_id, main_cnt = find_most_common_element(main_list)

            child_modes = []
            child_cnts = []
            for child_id in range(1, max_children + 1):
                child_list = track_result_list_children[track_id][child_id]
                cid, ccnt = find_most_common_element(child_list)
                if ccnt > 0:
                    child_modes.append(cid)
                    child_cnts.append(ccnt)

            if len(child_modes) == 0:
                continue
           
            if main_id != 0 and len(child_modes) > 0:
                match_count = sum(1 for cid in child_modes if cid == main_id and cid != 0)
                
                if match_count > 0:
                    
                    min_child_cnt = min(child_cnts)
                    if (main_cnt + min_child_cnt) >= cfg.SDA.SUCCESS_FRAME:
                        success_num += match_count / len(child_modes)

        total_tracks += max_track
        total_success += success_num
        print('evaluation of video {} finished! sda: {:.3f}'.format(
            file_name, success_num / max_track if max_track > 0 else 0.0))

    print('evaluation finished! total sda: {:.3f}'.format(
        total_success / total_tracks if total_tracks > 0 else 0.0))
