import os
import pandas as pd
from collections import defaultdict
from data_toolbox import *

sup_width = 3

if __name__ == '__main__':
    print('old data deleted!')
    data_path = 'data/mota_gts'
    mts_path = 'data/mts_gts'
    del_path(data_path)
    del_path(mts_path)
    mkdirs(data_path)
    mkdirs(mts_path)

    gt_root = 'data/gts'
    gts = sorted(os.listdir(gt_root))

    for gt_idx, gt in enumerate(gts, start=1):
        gt_path = os.path.join(gt_root, gt)
        data_dataframe = pd.read_csv(gt_path, sep=',')
        data_list = data_dataframe.values.tolist()

        children_map = defaultdict(set)
        for row in data_list:
            
            frame_id, obj_id, frag_id, x, y, *rest = row
            obj_id_int = int(obj_id)
            frag_id_int = int(frag_id)

            if frag_id_int > 0:
                children_map[obj_id_int].add(frag_id_int)

        default_child_ids = [1, 2]

        det_file = os.path.join(data_path, '{:03d}.txt'.format(gt_idx))
        mts_file = os.path.join(mts_path, '{:03d}.txt'.format(gt_idx))

        with open(det_file, 'w') as f, open(mts_file, 'w') as fs:
            for data_line in data_list:
                frame_id, obj_id, frag_id, x, y, *rest = data_line

                frame_id = int(frame_id)
                obj_id_int = int(obj_id)
                frag_id_int = int(frag_id)

                mota_track_id = int(f'{obj_id_int}{frag_id_int:02d}')

                new_data_line = [
                    frame_id,
                    mota_track_id,
                    x - sup_width,
                    y - sup_width,
                    2 * sup_width,
                    2 * sup_width,
                    1, 1, 1
                ]
                write_line = ','.join(str(v) for v in new_data_line)
                f.write(write_line + '\n')

                if frag_id_int == 0:
                    
                    child_ids = sorted(children_map.get(obj_id_int, set()))

                    if not child_ids:
                        
                        child_ids = default_child_ids

                    for child_frag in child_ids:
                        mts_track_id = int(f'{obj_id_int}{child_frag:02d}')
                        data_line_for_mts = [
                            frame_id,
                            mts_track_id,
                            x - sup_width,
                            y - sup_width,
                            2 * sup_width,
                            2 * sup_width,
                            1, 1, 1
                        ]
                        mts_line = ','.join(str(v) for v in data_line_for_mts)
                        fs.write(mts_line + '\n')
                else:
                    
                    fs.write(write_line + '\n')

        print('mota and mts format gt for video {} generated!'.format(gt_idx))
