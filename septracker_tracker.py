import numpy as np
import torch
from queue import Queue
from filterpy.kalman import KalmanFilter
from septracker_model import SeptrackerNet
from tracking_toolbox import *
from yacs.config import CfgNode as CN
import time
from collections import deque
from itertools import combinations

class TrackBuffer:
    def __init__(self, buffer_size, position_now):
        self.buffer_size = buffer_size
        self.queue = Queue(maxsize=buffer_size)

        for _ in range(buffer_size):
            padding_tensor = position_now
            self.queue.put(padding_tensor)

    def __len__(self):
        return self.buffer_size

    @property
    def as_list(self):
        result_list = []
        for i in range(self.buffer_size):
            element = self.queue.queue[i]
            result_list.append(element)

        return result_list

    def update(self, position):
        tensor = torch.tensor(position)

        if self.queue.full():
            self.queue.get()

        self.queue.put(tensor)

class Track:
    def __init__(self, detection, track_id, cfg, fragment_id=0):

        self.cfg = cfg
        self.track_id = track_id
        self.fragment_id = fragment_id
        self.hits = 0
        self.age = 0
        self.time_since_update = 0
        self.septracker_hits = 0
        self.septracker_gap = 0
        self.septracker_position1 = []
        self.septracker_position2 = []
        self.septracker_position3 = []
        self.septracker_position4 = []
        self.septracker_position5 = []
        self.dt = 1
        self.is_alive = True
        self.buffer_size = cfg.PARAM.TRACK_REWIND - 1
        self.track_buffer = TrackBuffer(self.buffer_size, torch.Tensor(detection))

        self.kf = KalmanFilter(dim_x=4, dim_z=2)

        self.kf.F = np.array([[1., 0., 1., 0.],
                              [0., 1., 0., 1.],
                              [0., 0., 1., 0.],
                              [0., 0., 0., 1.]])

        self.kf.Q = np.eye(4) * cfg.KF.Q_GAIN

        self.kf.H = np.array([[1., 0., 0., 0.],
                              [0., 1., 0., 0.]])

        self.kf.R = np.eye(2) * 1

        self.kf.x = np.array([float(t) for t in detection] + [0., 0.])
        self.kf.P = np.eye(4) * 1. * cfg.KF.P_GAIN

def septracker_distance(tracks, detections, model, cfg):
    if len(tracks) == 0 or len(detections) == 0:
        return [], []
    else:
        tracks_hist = np.clip([[np.array(t) for t in track.track_buffer.as_list[::-1]] for track in tracks], 1, cfg.PARAM.MAX_LEN)
        tracks_now = np.clip([track.kf.x[:2] for track in tracks], 1, cfg.PARAM.MAX_LEN)
        detections = np.clip(torch.tensor(np.array(detections)).unsqueeze(0), 1, cfg.PARAM.MAX_LEN)
        tracks = torch.tensor(np.array([np.concatenate((np.reshape(a, (1, len(a))), b), axis=0) for a, b in
                                        zip(tracks_now, tracks_hist)])).to(cfg.LOAD.DEV).unsqueeze(0)
        end_marks = torch.tensor([[len(tracks[0])], [len(detections[0])]]).to(cfg.LOAD.DEV)

        tracks_suppled = tensor_supple(tracks, cfg.PARAM.MAX_TGT, pad_value=cfg.PARAM.MAX_LEN+1)
        detections_suppled = tensor_supple_2d(detections, cfg.PARAM.MAX_DET, cfg.PARAM.MAX_LEN+1)
        tracks_suppled_tensor = torch.stack(tracks_suppled).to(cfg.LOAD.DEV)
        detections_suppled_tensor = torch.stack(detections_suppled).to(cfg.LOAD.DEV)

        match_score, group_score, _ = model(tracks_suppled_tensor, detections_suppled_tensor, end_marks)
        match_score_np = match_score[0].detach().cpu().numpy()
        group_score_np = group_score[0].detach().cpu().numpy()
        return match_score_np, group_score_np

def group(det_group_rows,det_group_cols):
    
    pairs = list(zip(det_group_rows, det_group_cols))
    from collections import defaultdict, deque
    adj = defaultdict(list)
    for a,b in pairs:
        adj[a].append(b); adj[b].append(a)

    groups = []
    visited = set()
    for v in adj.keys():
        if v in visited: continue
        comp = []
        q = deque([v]); visited.add(v)
        while q:
            u = q.popleft()
            comp.append(u)
            for w in adj[u]:
                if w not in visited:
                    visited.add(w); q.append(w)
        if len(comp) >= 2:
            groups.append(sorted(comp))

    result = [[int(x) for x in sublist] for sublist in groups]
    return result

def group_maxK_cover_first(P, K=5, thresh=1.0):
    
    P = np.asarray(P, dtype=float)
    n = P.shape[0]

    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if P[i, j] < thresh:
                adj[i].add(j)
                adj[j].add(i)

    vis = [False] * n
    comps = []
    for v in range(n):
        if not vis[v] and adj[v]:
            q = deque([v])
            vis[v] = True
            comp = [v]
            while q:
                u = q.popleft()
                for w in adj[u]:
                    if not vis[w]:
                        vis[w] = True
                        q.append(w)
                        comp.append(w)
            comps.append(sorted(comp))

    def score(S):
        
        return sum(P[i, j] for i, j in combinations(S, 2))

    groups = []
    for comp in comps:
        if len(comp) <= K:
            
            groups.append(comp)
        else:
            
            best_S = None
            best_score = float('inf')
            for S in combinations(comp, K):
                s = score(S)

                if s < best_score or (s == best_score and list(S) < best_S):
                    best_S = list(S)
                    best_score = s
            groups.append(best_S)

    return groups

def get_max_match_indices(match_scores, cols, rows, zero_fill=9999):
    
    if not cols or not rows:
        return []

    sub = match_scores[np.ix_(rows, cols)]
    local_row_idx = np.argmax(sub, axis=0)
    col_max = sub[local_row_idx, np.arange(sub.shape[1])]

    result = [rows[i] if col_max[j] == 0 else rows[i] for j, i in enumerate(local_row_idx)]
    
    result = [zero_fill if col_max[j] == 0 else rows[i] for j, i in enumerate(local_row_idx)]
    return result

class SeptrackerTracker:
    def __init__(self, cfg, logger, distance_method='septracker'):
        self.tracks = []
        self.cfg = cfg
        self.logger = logger
        self.device = cfg.TRACK.DEV
        self.confirm_age = cfg.TRACK.CONFIRM_AGE
        self.septracker_age = cfg.TRACK.SCATTER_AGE
        self.max_age = cfg.TRACK.MAX_AGE
        self.max_dist = cfg.TRACK.MAX_DIST
        self.print_remain = cfg.TRACK.PRINT_REMAIN
        self.del_age = cfg.TRACK.DEL_AGE
        self.septracker_max_gap = cfg.TRACK.MAX_GAP
        self.track_counter = 1
        self.unconfirmed_track_counter = -1
        self.distance_method = distance_method
        self.need_hit = cfg.TRACK.NEED_HIT
        self.model_path = cfg.TRACK.MODEL
        self.track_model = SeptrackerNet(cfg, logger=logger, is_eval=True).to(self.device)
        self.aa=1

        checkpoint = torch.load(self.model_path)
        self.track_model.load_state_dict(checkpoint['state_dict'], strict=False)
        self.track_model.eval()

    def get_tracks(self):
        return self.tracks
    
    def update(self, detections, img=None):

        if len(self.tracks) == 0:

            for i in range(len(detections)):
                track = Track(detections[i], self.unconfirmed_track_counter, cfg=self.cfg)
                self.unconfirmed_track_counter -= 1
                self.tracks.append(track)
        else:

            for track in self.tracks:
                track.kf.predict()
                track.time_since_update += 1
                track.age += 1

            if self.distance_method == 'septracker':
                match_scores, group_scores = septracker_distance(self.tracks, detections, self.track_model, self.cfg)

                col_inds_cmask = np.arange(match_scores.shape[1])
                row_inds_cmask = np.argmax(match_scores, axis=0)
                
                col_max = match_scores.max(axis=0)
                
                keep_mask = col_max != 0
                
                col_inds_cmask = col_inds_cmask[keep_mask]
                row_inds_cmask = row_inds_cmask[keep_mask]

                row_pos=[]
                unique_arr = np.unique(row_inds_cmask)
                for r in unique_arr:  
                    if self.tracks[r].track_id >= 0:
                        row_pos.append(r)
            else:
                raise ValueError

            if len(match_scores) > 0:
                
                tri_mask = np.ones_like(group_scores)
                group_scores[np.triu_indices(tri_mask.shape[0], k=0)] = 0
                
                det_group_rows, det_group_cols = lap_assignment_multi_threshold(group_scores, cost_limit=self.cfg.TRACK.GROUP_MIN_COST)
                
                all_det_list = group(det_group_rows,det_group_cols)
                
                supple_dets = []
                used_dets = []
                
                for row_ind, col_ind in zip(row_inds_cmask, col_inds_cmask):
                    
                    if row_ind >= 0:
                        if all(int(col_ind) not in sublis for sublis in all_det_list):
                            
                            self.tracks[row_ind].kf.update(detections[int(col_ind)])
                            self.tracks[row_ind].track_buffer.update(self.tracks[row_ind].kf.x[:2])
                            self.tracks[row_ind].time_since_update = 0
                            self.tracks[row_ind].septracker_gap += 1
                            self.tracks[row_ind].hits += 1
                            if self.tracks[row_ind].septracker_gap >= self.septracker_max_gap:
                                self.tracks[row_ind].septracker_hits = 0
                            used_dets.append(col_ind)
                            continue

                        else:
                            num_1=0
                            if int(col_ind) not in used_dets:
                                
                                for i, sublist in enumerate(all_det_list):
                                    
                                    if int(col_ind) in sublist:
                                        row_in=get_max_match_indices(match_scores,sublist,row_pos)
                                        all_equal = len(set(row_in)) == 1
                                        if all_equal and self.tracks[row_ind].fragment_id == 0:
                                            zuobiaos=[]
                                            for k, bosition in enumerate(sublist):
                                                used_dets.append(bosition)
                                                detectionss = detections[bosition]
                                                zuobiaos.append(detectionss)
                                                if k==0:
                                                    self.tracks[row_ind].septracker_position1 = detectionss
                                                if k==1:
                                                    self.tracks[row_ind].septracker_position2 = detectionss
                                                if k==2:
                                                    self.tracks[row_ind].septracker_position3 = detectionss
                                                if k==3:
                                                    self.tracks[row_ind].septracker_position4 = detectionss
                                                if k==4:
                                                    self.tracks[row_ind].septracker_position5 = detectionss
                                            
                                            col0 = [x[0] for x in zuobiaos]
                                            col1 = [x[1] for x in zuobiaos]
                                            avg0 = round(sum(col0) / len(col0), 1)
                                            avg1 = round(sum(col1) / len(col1), 1)
                                            zuobiao = [avg0, avg1]

                                            self.tracks[row_ind].kf.update(zuobiao)
                                            self.tracks[row_ind].track_buffer.update(self.tracks[row_ind].kf.x[:2])
                                            self.tracks[row_ind].time_since_update = 0
                                            self.tracks[row_ind].hits += 1
                                            self.tracks[row_ind].septracker_hits += 1
                                            self.tracks[row_ind].septracker_gap = 0
                                            num_1=1
                                            break
                            if num_1==0:
                                supple_dets.append(col_ind)

                supple_track_mask_list = []
                for i, track in enumerate(self.tracks):
                    if track.time_since_update == 0:
                        supple_track_mask_list.append(i)

                asso_std = np.ones_like(match_scores)
                supple_track_mask = np.zeros_like(match_scores)
                supple_track_mask[supple_track_mask_list, :] = 1
                supple_det_mask = np.ones_like(match_scores)
                supple_det_mask[:, supple_dets] = 0
                supple_cost = asso_std - match_scores + supple_track_mask + supple_det_mask

                supple_rows, supple_cols = lap_assignment(supple_cost, extend_cost=True, cost_limit=self.cfg.TRACK.ASSO_MAX_COST)

                for supple_row, supple_col in zip(supple_rows, supple_cols):
                    if supple_row >= 0:
                        self.tracks[supple_row].kf.update(detections[supple_col])
                        self.tracks[supple_row].track_buffer.update(self.tracks[supple_row].kf.x[:2])
                        self.tracks[supple_row].time_since_update = 0
                        self.tracks[supple_row].hits += 1
                        self.tracks[supple_row].septracker_gap += 1
                        if self.tracks[supple_row].septracker_gap >= self.septracker_max_gap:
                            self.tracks[supple_row].septracker_hits = 0
                        used_dets.append(supple_col)

            for track in self.tracks:
                if track.time_since_update != 0:
                    track.kf.update(track.kf.x[:2])
                    track.track_buffer.update(track.kf.x[:2]) 
                    track.hits = 0
                    track.septracker_gap += 1
                    if track.septracker_gap >= self.septracker_max_gap:
                        track.septracker_hits = 0
                    if track.time_since_update > self.max_age:
                        track.is_alive = False
                
                elif track.track_id < 0 and track.hits > self.confirm_age:
                    track.track_id = self.track_counter
                    self.track_counter += 1

                elif track.septracker_hits >= self.septracker_age:
                    
                    track.is_alive = False
                    septracker_track1 = Track(track.septracker_position1, track.track_id, self.cfg, fragment_id=1)
                    
                    self.tracks.append(septracker_track1)
                    
                    if getattr(track, 'septracker_position2', None) is not None and len(track.septracker_position2) == 2:
                        septracker_track2 = Track(track.septracker_position2, track.track_id, self.cfg, fragment_id=2)
                        self.tracks.append(septracker_track2)
 
                    if getattr(track, 'septracker_position3', None) is not None and len(track.septracker_position3) == 2:
                        septracker_track3 = Track(track.septracker_position3, track.track_id, self.cfg, fragment_id=3)
                        self.tracks.append(septracker_track3)
                    
                    if getattr(track, 'septracker_position4', None) is not None and len(track.septracker_position4) == 2:
                        septracker_track4 = Track(track.septracker_position4, track.track_id, self.cfg, fragment_id=4)
                        self.tracks.append(septracker_track4)
                    
                    if getattr(track, 'septracker_position5', None) is not None and len(track.septracker_position5) == 2:
                        septracker_track5 = Track(track.septracker_position5, track.track_id, self.cfg, fragment_id=5)
                        self.tracks.append(septracker_track5)

            for i, det in enumerate(detections):
                if i not in used_dets:
                    track = Track(detections[i], self.unconfirmed_track_counter, cfg=self.cfg)
                    self.unconfirmed_track_counter -= 1
                    self.tracks.append(track)

            self.tracks = [x for x in self.tracks if x.is_alive]

        if self.need_hit:
            tracks_print = [x for x in self.tracks if x.time_since_update <= self.print_remain and x.hits >= self.del_age and x.track_id > 0]
        else:
            tracks_print = [x for x in self.tracks if x.time_since_update <= self.print_remain and x.track_id > 0]
        return tracks_print

