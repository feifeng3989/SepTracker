import copy
import os
import numpy as np
import math
import random
import shutil
from scipy.signal import convolve2d

def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def del_path(path):
    if os.path.exists(path):
        shutil.rmtree(path)

def cal_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def choose_start_with_min_dist(start_area, screen_size, existing_starts, min_init_dist=35, max_retry=100):
    screen_x, screen_y = screen_size
    for _ in range(max_retry):
        sx, sy, ang = initiate_track(start_area, screen_size)
        if sx == -1:
            continue
        ok = True
        for (px, py) in existing_starts:
            if cal_distance((sx, sy), (px, py)) < min_init_dist:
                ok = False
                break
        if ok:
            return sx, sy, ang
    
    return sx, sy, ang

def generate_tgt(x_len, y_len, snr, dist_ratio=1, kernel_len=3, noise_std=3.16):
    
    target = np.zeros((x_len, y_len), dtype=np.uint32)
    target_mid = np.zeros((x_len, y_len), dtype=np.uint32)
    max_num = noise_std * 2 * snr
    template = np.random.randint(max_num * 0.5, max_num * 1.5, size=(x_len, y_len))
    kernel = np.ones((kernel_len, kernel_len))
    target_filtered = convolve2d(template, kernel, mode='same')
    center_x = (x_len-1) // 2
    center_y = (y_len-1) // 2
    max_dist = cal_distance((x_len, y_len), (center_x, center_y))
    fix_ratio = 1 / kernel_len**2

    for i in range(x_len):
        for j in range(y_len):
            dist = cal_distance((i, j), (center_x, center_y))
            
            target_mid[i, j] = fix_ratio * target_filtered[i, j] * ((max_dist - dist) / max_dist)**dist_ratio

    target_second_filtered = convolve2d(target_mid, kernel, mode='same')
    for i in range(x_len):
        for j in range(y_len):
            dist = cal_distance((i, j), (center_x, center_y))
            target[i, j] = target_second_filtered[i, j] * ((max_dist - dist) / max_dist)**dist_ratio

    return target

def generate_curve(start_frame, track_id, start_pos, start_angle, speed, track_all, screen_size, tgt_class):
    
    screen_x, screen_y = screen_size
    frame_id = start_frame
    pos_now = start_pos
    angle_now = start_angle
    supple_len = int(min(max(50, 50 / speed), 400))
    break_len = random.randint(70, 90)
    break_angle1 = random.randint(35, 40) * math.pi / 180
    break_angle2 = random.randint(35, 40) * math.pi / 180
    break_angle3 = random.randint(15, 20) * math.pi / 180
    break_angle4 = random.randint(15, 20) * math.pi / 180
    break_angle5 = random.randint(0, 2) * math.pi / 180
    fix_angle = random.randint(10, 20)
    break_period = random.randint(5, 10)
    obj_class = 1 if break_len % 2 == 0 else 2
    fragment1_size = [3, 3]
    fragment2_size = [3, 3] if tgt_class == 1 else [4, 5]
    fragment3_size = [3, 3]
    fragment4_size = [3, 3]
    fragment5_size = [3, 3]
    main_size = [5, 5] if tgt_class == 1 else [5, 6]
    expand_size = [6, 6] if tgt_class == 1 else [6, 7]
    bright = 18.3
    upper_bright = 22.1
    for i in [1, 2, 3, 4, 5]:
        rand_num = random.choice([6,8,9,12,13.7])
        var_name = f"bright_obj{i}"
        globals()[var_name] = rand_num
        
    if obj_class == 2:
        bright = 35.5
        upper_bright = 41.4
        for i in [1, 2, 3, 4, 5]:
            rand_num = random.choice([5.3,13,18,22,29.4])
            var_name = f"bright_obj{i}"
            globals()[var_name] = rand_num
            
    angle_slice = fix_angle * math.pi / 180 / break_len if break_len % 2 == 0 else fix_angle * math.pi / -180 / break_len

    for i in range(break_len - break_period):     
        angle_now += angle_slice
        pos_now[0] += speed * math.cos(angle_now)
        pos_now[1] += speed * math.sin(angle_now)
        if 0 < pos_now[0] < screen_x and 0 < pos_now[1] < screen_y:
            
            curve = [frame_id, track_id, 0, pos_now[0], pos_now[1], main_size[0], main_size[1], bright]
            track_all.append(curve)
        frame_id += 1

    for i in range(break_period):
        angle_now += angle_slice
        pos_now[0] += speed * math.cos(angle_now)
        pos_now[1] += speed * math.sin(angle_now)
        if 0 < pos_now[0] < screen_x and 0 < pos_now[1] < screen_y:
            curve = [frame_id, track_id, 0, pos_now[0], pos_now[1], expand_size[0], expand_size[1], upper_bright]
            track_all.append(curve)
        frame_id += 1

    angle_obj1 = angle_now + break_angle1
    angle_obj2 = angle_now - break_angle2
    angle_obj3 = angle_now + break_angle3
    angle_obj4 = angle_now - break_angle4
    angle_obj5 = angle_now - break_angle5
    angle1_now = angle_obj1
    angle2_now = angle_obj2
    angle3_now = angle_obj3
    angle4_now = angle_obj4
    angle5_now = angle_obj5
    pos1_now = copy.copy(pos_now)
    pos2_now = copy.copy(pos_now)
    pos3_now = copy.copy(pos_now)
    pos4_now = copy.copy(pos_now)
    pos5_now = copy.copy(pos_now)

    ran_num = random.choice([2, 3, 4, 5])
    
    if ran_num == 2:                           
        for i in range(supple_len):
            angle1_now += angle_slice
            angle2_now += angle_slice
            pos1_now[0] += speed * math.cos(angle1_now)
            pos1_now[1] += speed * math.sin(angle1_now)
            pos2_now[0] += speed * math.cos(angle2_now)
            pos2_now[1] += speed * math.sin(angle2_now)
            if 0 < pos1_now[0] < screen_x and 0 < pos1_now[1] < screen_y:
                curve1 = [frame_id, track_id, 1, pos1_now[0], pos1_now[1], fragment1_size[0], fragment1_size[1], bright_obj1]
                
                track_all.append(curve1)
            if 0 < pos2_now[0] < screen_x and 0 < pos2_now[1] < screen_y:
                curve2 = [frame_id, track_id, 2, pos2_now[0], pos2_now[1], fragment2_size[0], fragment2_size[1], bright_obj2]
                
                track_all.append(curve2)
            frame_id += 1
    if ran_num == 3:
        for i in range(supple_len):
            angle1_now += angle_slice
            angle2_now += angle_slice
            angle5_now += angle_slice
            pos1_now[0] += speed * math.cos(angle1_now)
            pos1_now[1] += speed * math.sin(angle1_now)
            pos2_now[0] += speed * math.cos(angle2_now)
            pos2_now[1] += speed * math.sin(angle2_now)
            pos3_now[0] += speed * math.cos(angle5_now)
            pos3_now[1] += speed * math.sin(angle5_now)
            if 0 < pos1_now[0] < screen_x and 0 < pos1_now[1] < screen_y:
                curve1 = [frame_id, track_id, 1, pos1_now[0], pos1_now[1], fragment1_size[0], fragment1_size[1], bright_obj1]
                
                track_all.append(curve1)
            if 0 < pos2_now[0] < screen_x and 0 < pos2_now[1] < screen_y:
                curve2 = [frame_id, track_id, 2, pos2_now[0], pos2_now[1], fragment2_size[0], fragment2_size[1], bright_obj2]
                
                track_all.append(curve2)
            if 0 < pos3_now[0] < screen_x and 0 < pos3_now[1] < screen_y:
                curve3 = [frame_id, track_id, 3, pos3_now[0], pos3_now[1], fragment3_size[0], fragment3_size[1], bright_obj3]
                
                track_all.append(curve3)
            frame_id += 1

    if ran_num == 4:
        for i in range(supple_len):
            angle1_now += angle_slice
            angle2_now += angle_slice
            angle3_now += angle_slice
            angle4_now += angle_slice
            pos1_now[0] += speed * math.cos(angle1_now)
            pos1_now[1] += speed * math.sin(angle1_now)
            pos2_now[0] += speed * math.cos(angle2_now)
            pos2_now[1] += speed * math.sin(angle2_now)
            pos3_now[0] += speed * math.cos(angle3_now)
            pos3_now[1] += speed * math.sin(angle3_now)
            pos4_now[0] += speed * math.cos(angle4_now)
            pos4_now[1] += speed * math.sin(angle4_now)
            if 0 < pos1_now[0] < screen_x and 0 < pos1_now[1] < screen_y:
                curve1 = [frame_id, track_id, 1, pos1_now[0], pos1_now[1], fragment1_size[0], fragment1_size[1], bright_obj1]
                
                track_all.append(curve1)
            if 0 < pos2_now[0] < screen_x and 0 < pos2_now[1] < screen_y:
                curve2 = [frame_id, track_id, 2, pos2_now[0], pos2_now[1], fragment2_size[0], fragment2_size[1], bright_obj2]
                
                track_all.append(curve2)
            if 0 < pos3_now[0] < screen_x and 0 < pos3_now[1] < screen_y:
                curve3 = [frame_id, track_id, 3, pos3_now[0], pos3_now[1], fragment3_size[0], fragment3_size[1], bright_obj3]
                
                track_all.append(curve3)
            if 0 < pos4_now[0] < screen_x and 0 < pos4_now[1] < screen_y:
                curve4 = [frame_id, track_id, 4, pos4_now[0], pos4_now[1], fragment4_size[0], fragment4_size[1], bright_obj4]
                
                track_all.append(curve4)
            frame_id += 1

    if ran_num == 5:
        for i in range(supple_len):
            angle1_now += angle_slice
            angle2_now += angle_slice
            angle3_now += angle_slice
            angle4_now += angle_slice
            angle5_now += angle_slice
            pos1_now[0] += speed * math.cos(angle1_now)
            pos1_now[1] += speed * math.sin(angle1_now)
            pos2_now[0] += speed * math.cos(angle2_now)
            pos2_now[1] += speed * math.sin(angle2_now)
            pos3_now[0] += speed * math.cos(angle3_now)
            pos3_now[1] += speed * math.sin(angle3_now)
            pos4_now[0] += speed * math.cos(angle4_now)
            pos4_now[1] += speed * math.sin(angle4_now)
            pos5_now[0] += speed * math.cos(angle5_now)
            pos5_now[1] += speed * math.sin(angle5_now)
            if 0 < pos1_now[0] < screen_x and 0 < pos1_now[1] < screen_y:
                curve1 = [frame_id, track_id, 1, pos1_now[0], pos1_now[1], fragment1_size[0], fragment1_size[1], bright_obj1]
                
                track_all.append(curve1)
            if 0 < pos2_now[0] < screen_x and 0 < pos2_now[1] < screen_y:
                curve2 = [frame_id, track_id, 2, pos2_now[0], pos2_now[1], fragment2_size[0], fragment2_size[1], bright_obj2]
                
                track_all.append(curve2)
            if 0 < pos3_now[0] < screen_x and 0 < pos3_now[1] < screen_y:
                curve3 = [frame_id, track_id, 3, pos3_now[0], pos3_now[1], fragment3_size[0], fragment3_size[1], bright_obj3]
                
                track_all.append(curve3)
            if 0 < pos4_now[0] < screen_x and 0 < pos4_now[1] < screen_y:
                curve4 = [frame_id, track_id, 4, pos4_now[0], pos4_now[1], fragment4_size[0], fragment4_size[1], bright_obj4]
                
                track_all.append(curve4)
            if 0 < pos5_now[0] < screen_x and 0 < pos5_now[1] < screen_y:
                curve5 = [frame_id, track_id, 5, pos5_now[0], pos5_now[1], fragment5_size[0], fragment5_size[1], bright_obj5]
                
                track_all.append(curve5)
            frame_id += 1

def initiate_track(start_area, screen_size):
    
    screen_x, screen_y = screen_size
    position_comp_x = random.uniform(0, screen_x / 4)
    position_comp_y = random.uniform(0, screen_y / 4)
    angle_comp = random.uniform(math.pi / -6, math.pi / 6)
    if start_area == 1:
        start_x = position_comp_x + 5
        start_y = 5
        start_angle = math.pi / 4 + angle_comp
    elif start_area == 2:
        start_x = screen_x - position_comp_x - 5
        start_y = 5
        start_angle = math.pi * 3 / 4 + angle_comp
    elif start_area == 3:
        start_x = screen_x - 5
        start_y = position_comp_y + 5
        start_angle = math.pi * 3 / 4 + angle_comp
    elif start_area == 4:
        start_x = screen_x - 5
        start_y = screen_y - position_comp_y - 5
        start_angle = math.pi * 3 / -4 + angle_comp
    elif start_area == 5:
        start_x = screen_x - position_comp_x - 5
        start_y = screen_y - 5
        start_angle = math.pi * 3 / -4 + angle_comp
    elif start_area == 6:
        start_x = position_comp_x + 5
        start_y = screen_y - 5
        start_angle = math.pi / -4 + angle_comp
    elif start_area == 7:
        start_x = 5
        start_y = screen_y - position_comp_y - 5
        start_angle = math.pi / -4 + angle_comp
    elif start_area == 8:
        start_x = 5
        start_y = position_comp_y + 5
        start_angle = math.pi / 4 + angle_comp
    else:
        return -1, -1, -1
    return start_x, start_y, start_angle

def generate_track(video_seq, screen_size, speed_fix_rate=1):
    
    track_all = []
    area_seq = [1, 3, 5, 7]

    existing_starts = []

    speed_list1 = [0.79, 0.59, 0.4, 0.2,
                   0.4, 0.3, 0.2, 0.1,
                   0.26, 0.2, 0.13, 0.07,
                   0.2, 0.15, 0.1, 0.05,
                   0.16, 0.12, 0.08, 0.04,
                   0.13, 0.1, 0.07, 0.03]
    random.shuffle(speed_list1)

    speed_list2 = [1.33, 1., 0.67, 0.33,
                   0.67, 0.5, 0.33, 0.17,
                   0.44, 0.33, 0.22, 0.11,
                   0.33, 0.25, 0.17, 0.08,
                   0.27, 0.20, 0.13, 0.07,
                   0.22, 0.17, 0.11, 0.06]
    random.shuffle(speed_list2)

    speed_list = speed_list1 if video_seq % 2 == 0 else speed_list2
    tgt_class = 1 if video_seq % 2 == 1 else 2
    area_seq_move = random.randint(0, 8)
    areas_end_point = [0, 0, 0, 0]
    area_choose = [[8, 1],
                   [2, 3],
                   [4, 5],
                   [6, 7]]

    for track_id in range(1, 5):
        track_area = area_seq[(track_id + area_seq_move) % 4]
        track_speed = speed_list[track_id-1] * speed_fix_rate
        areas_end_point[(track_id + area_seq_move) % 4] = 90 + int(min(max(50, 50 / track_speed), 400))
        
        start_x, start_y, start_angle = choose_start_with_min_dist(track_area, screen_size, existing_starts, min_init_dist=35)
        existing_starts.append((start_x, start_y))
        if len(existing_starts) > 4:
            existing_starts = existing_starts[1:]

        start_frame = 1
        start_pos = [start_x, start_y]
        generate_curve(start_frame, track_id, start_pos, start_angle, track_speed, track_all, screen_size, tgt_class)
        
    for track_id in range(5, 25):
        first_valid_area = areas_end_point.index(min(areas_end_point))
        track_speed = speed_list[track_id - 1]
        direction = track_id % 2
        track_area = area_choose[first_valid_area][0] if direction == 0 else area_choose[first_valid_area][1]
        start_x, start_y, start_angle = choose_start_with_min_dist(track_area, screen_size, existing_starts, min_init_dist=35)
        existing_starts.append((start_x, start_y))
        if len(existing_starts) > 4:
            existing_starts = existing_starts[1:]

        start_frame = areas_end_point[first_valid_area] + 5
        start_pos = [start_x, start_y]
        areas_end_point[first_valid_area] += (95 + int(min(max(50, 50 / track_speed), 400)))
        
        generate_curve(start_frame, track_id, start_pos, start_angle, track_speed, track_all, screen_size, tgt_class)

    sorted_tracks = sorted(track_all, key=lambda x: x[0])
    mkdirs('data/gts')
    txt_name = 'data/gts/{:03d}.txt'.format(video_seq)
    with open(txt_name, 'w') as f:
        for row in sorted_tracks:
            row_str = ','.join(str(value) for value in row)
            f.write(row_str + '\n')

def bg_list_enhance(bgs):
    
    duplicated_bgs = [x for x in bgs for _ in range(3)]
    reversed_bgs = duplicated_bgs[::-1]
    duplicated_bgs.extend(reversed_bgs)
    return duplicated_bgs

def bg_stretch(img_as_list, top_extend=200, bot_extend=50):
    
    img = np.array(img_as_list)
    avg = np.mean(img, axis=None)
    top_limit = avg + top_extend
    bot_limit = avg - bot_extend
    img = np.clip(img, bot_limit, top_limit)
    img = linear_mapping(img, bot_limit, top_limit, 0, 255)
    return img

def linear_mapping(image, low_gray, high_gray, low_mapped, high_mapped):
    
    input_range = high_gray - low_gray
    output_range = high_mapped - low_mapped
    mapped_image = (image - low_gray) * (output_range / input_range) + low_mapped
    mapped_image = np.clip(mapped_image, 0, 255).astype(np.uint8)
    return mapped_image

def generate_hard_curve(start_frame, track_id, start_pos, start_angle, speed, track_all, screen_size, tgt_class):
    
    screen_x, screen_y = screen_size
    frame_id = start_frame
    pos_now = start_pos
    angle_now = start_angle
    supple_len = int(min(max(150, 150 / speed), 240))
    break_len = random.randint(60, 100)
    break_angle1 = random.randint(35, 40) * math.pi / 180
    break_angle2 = random.randint(35, 40) * math.pi / 180
    break_angle3 = random.randint(15, 20) * math.pi / 180
    break_angle4 = random.randint(15, 20) * math.pi / 180
    break_angle5 = random.randint(0, 3) * math.pi / 180
    fix_angle = random.randint(10, 20)
    break_period = random.randint(10, 20)
    total_len = supple_len + break_len + break_period
    speed_step = random.uniform(0.2, 0.4) * speed / total_len if total_len % 2 == 0 else \
        -1 * random.uniform(0.2, 0.4) * speed / total_len
    obj_class = 1 if break_len % 2 == 0 else 2
    fragment1_size = [3, 3]
    fragment2_size = [3, 3] if tgt_class == 1 else [4, 5]
    fragment3_size = [3, 3]
    fragment4_size = [3, 3]
    fragment5_size = [3, 3]
    main_size = [5, 5] if tgt_class == 1 else [5, 6]
    expand_size = [6, 6] if tgt_class == 1 else [6, 7]
    bright = 25.3
    upper_bright = 32.1
    for i in [1, 2, 3, 4, 5]:
        rand_num = random.choice([16,18,19,21,23])
        var_name = f"bright_obj{i}"
        globals()[var_name] = rand_num
        
    if obj_class == 2:
        bright = 35.5
        upper_bright = 41.4
        for i in [1, 2, 3, 4, 5]:
            rand_num = random.choice([21,25,26,28,29])
            var_name = f"bright_obj{i}"
            globals()[var_name] = rand_num
            
    angle_slice = fix_angle * math.pi / 180 / break_len if break_len % 2 == 0 else fix_angle * math.pi / -180 / break_len

    for i in range(break_len - break_period):
        speed += speed_step
        angle_now += angle_slice
        pos_now[0] += speed * math.cos(angle_now)
        pos_now[1] += speed * math.sin(angle_now)
        if 0 < pos_now[0] < screen_x and 0 < pos_now[1] < screen_y:
            curve = [frame_id, track_id, 0, pos_now[0], pos_now[1], main_size[0], main_size[1], bright]
            track_all.append(curve)
        frame_id += 1

    for i in range(break_period):
        speed += speed_step
        angle_now += angle_slice
        pos_now[0] += speed * math.cos(angle_now)
        pos_now[1] += speed * math.sin(angle_now)
        if 0 < pos_now[0] < screen_x and 0 < pos_now[1] < screen_y:
            curve = [frame_id, track_id, 0, pos_now[0], pos_now[1], expand_size[0], expand_size[1], upper_bright]
            track_all.append(curve)
        frame_id += 1

    angle_obj1 = angle_now + break_angle1
    angle_obj2 = angle_now - break_angle2
    angle_obj3 = angle_now + break_angle3
    angle_obj4 = angle_now - break_angle4 
    angle_obj5 = angle_now - break_angle5 
    angle1_now = angle_obj1
    angle2_now = angle_obj2
    angle3_now = angle_obj3
    angle4_now = angle_obj4
    angle5_now = angle_obj5  
    pos1_now = copy.copy(pos_now)
    pos2_now = copy.copy(pos_now)
    pos3_now = copy.copy(pos_now)
    pos4_now = copy.copy(pos_now)
    pos5_now = copy.copy(pos_now)
    
    ran_num = random.choice([2, 3, 4, 5])
    
    if ran_num == 2:                           
        for i in range(supple_len):
            angle1_now += angle_slice
            angle2_now += angle_slice
            pos1_now[0] += speed * math.cos(angle1_now)
            pos1_now[1] += speed * math.sin(angle1_now)
            pos2_now[0] += speed * math.cos(angle2_now)
            pos2_now[1] += speed * math.sin(angle2_now)
            if 0 < pos1_now[0] < screen_x and 0 < pos1_now[1] < screen_y:
                curve1 = [frame_id, track_id, 1, pos1_now[0], pos1_now[1], fragment1_size[0], fragment1_size[1], bright_obj1]
                
                track_all.append(curve1)
            if 0 < pos2_now[0] < screen_x and 0 < pos2_now[1] < screen_y:
                curve2 = [frame_id, track_id, 2, pos2_now[0], pos2_now[1], fragment2_size[0], fragment2_size[1], bright_obj2]
                
                track_all.append(curve2)
            frame_id += 1
    if ran_num == 3:
        for i in range(supple_len):
            angle1_now += angle_slice
            angle2_now += angle_slice
            angle5_now += angle_slice
            pos1_now[0] += speed * math.cos(angle1_now)
            pos1_now[1] += speed * math.sin(angle1_now)
            pos2_now[0] += speed * math.cos(angle2_now)
            pos2_now[1] += speed * math.sin(angle2_now)
            pos3_now[0] += speed * math.cos(angle5_now)
            pos3_now[1] += speed * math.sin(angle5_now)
            if 0 < pos1_now[0] < screen_x and 0 < pos1_now[1] < screen_y:
                curve1 = [frame_id, track_id, 1, pos1_now[0], pos1_now[1], fragment1_size[0], fragment1_size[1], bright_obj1]
                
                track_all.append(curve1)
            if 0 < pos2_now[0] < screen_x and 0 < pos2_now[1] < screen_y:
                curve2 = [frame_id, track_id, 2, pos2_now[0], pos2_now[1], fragment2_size[0], fragment2_size[1], bright_obj2]
                
                track_all.append(curve2)
            if 0 < pos3_now[0] < screen_x and 0 < pos3_now[1] < screen_y:
                curve3 = [frame_id, track_id, 3, pos3_now[0], pos3_now[1], fragment3_size[0], fragment3_size[1], bright_obj3]
                
                track_all.append(curve3)
            frame_id += 1

    if ran_num == 4:
        for i in range(supple_len):
            angle1_now += angle_slice
            angle2_now += angle_slice
            angle3_now += angle_slice
            angle4_now += angle_slice
            pos1_now[0] += speed * math.cos(angle1_now)
            pos1_now[1] += speed * math.sin(angle1_now)
            pos2_now[0] += speed * math.cos(angle2_now)
            pos2_now[1] += speed * math.sin(angle2_now)
            pos3_now[0] += speed * math.cos(angle3_now)
            pos3_now[1] += speed * math.sin(angle3_now)
            pos4_now[0] += speed * math.cos(angle4_now)
            pos4_now[1] += speed * math.sin(angle4_now)
            if 0 < pos1_now[0] < screen_x and 0 < pos1_now[1] < screen_y:
                curve1 = [frame_id, track_id, 1, pos1_now[0], pos1_now[1], fragment1_size[0], fragment1_size[1], bright_obj1]
                
                track_all.append(curve1)
            if 0 < pos2_now[0] < screen_x and 0 < pos2_now[1] < screen_y:
                curve2 = [frame_id, track_id, 2, pos2_now[0], pos2_now[1], fragment2_size[0], fragment2_size[1], bright_obj2]
                
                track_all.append(curve2)
            if 0 < pos3_now[0] < screen_x and 0 < pos3_now[1] < screen_y:
                curve3 = [frame_id, track_id, 3, pos3_now[0], pos3_now[1], fragment3_size[0], fragment3_size[1], bright_obj3]
                
                track_all.append(curve3)
            if 0 < pos4_now[0] < screen_x and 0 < pos4_now[1] < screen_y:
                curve4 = [frame_id, track_id, 4, pos4_now[0], pos4_now[1], fragment4_size[0], fragment4_size[1], bright_obj4]
                
                track_all.append(curve4)
            frame_id += 1

    if ran_num == 5:
        for i in range(supple_len):
            angle1_now += angle_slice
            angle2_now += angle_slice
            angle3_now += angle_slice
            angle4_now += angle_slice
            angle5_now += angle_slice
            pos1_now[0] += speed * math.cos(angle1_now)
            pos1_now[1] += speed * math.sin(angle1_now)
            pos2_now[0] += speed * math.cos(angle2_now)
            pos2_now[1] += speed * math.sin(angle2_now)
            pos3_now[0] += speed * math.cos(angle3_now)
            pos3_now[1] += speed * math.sin(angle3_now)
            pos4_now[0] += speed * math.cos(angle4_now)
            pos4_now[1] += speed * math.sin(angle4_now)
            pos5_now[0] += speed * math.cos(angle5_now)
            pos5_now[1] += speed * math.sin(angle5_now)
            if 0 < pos1_now[0] < screen_x and 0 < pos1_now[1] < screen_y:
                curve1 = [frame_id, track_id, 1, pos1_now[0], pos1_now[1], fragment1_size[0], fragment1_size[1], bright_obj1]
                
                track_all.append(curve1)
            if 0 < pos2_now[0] < screen_x and 0 < pos2_now[1] < screen_y:
                curve2 = [frame_id, track_id, 2, pos2_now[0], pos2_now[1], fragment2_size[0], fragment2_size[1], bright_obj2]
                
                track_all.append(curve2)
            if 0 < pos3_now[0] < screen_x and 0 < pos3_now[1] < screen_y:
                curve3 = [frame_id, track_id, 3, pos3_now[0], pos3_now[1], fragment3_size[0], fragment3_size[1], bright_obj3]
                
                track_all.append(curve3)
            if 0 < pos4_now[0] < screen_x and 0 < pos4_now[1] < screen_y:
                curve4 = [frame_id, track_id, 4, pos4_now[0], pos4_now[1], fragment4_size[0], fragment4_size[1], bright_obj4]
                
                track_all.append(curve4)
            if 0 < pos5_now[0] < screen_x and 0 < pos5_now[1] < screen_y:
                curve5 = [frame_id, track_id, 5, pos5_now[0], pos5_now[1], fragment5_size[0], fragment5_size[1], bright_obj5]
                
                track_all.append(curve5)
            frame_id += 1

def generate_hard_track(video_seq, screen_size, speed_fix_rate=3):
    
    track_all = []

    area_seq1 = [1, 3, 1, 3]
    area_seq2 = [5, 7, 5, 7]

    existing_starts = []

    area_seq = area_seq1 if video_seq % 2 == 0 else area_seq2

    speed_list1 = [0.79, 0.59, 0.4, 0.88]
    random.shuffle(speed_list1)
    speed_list2 = [0.5, 0.44, 0.22, 0.33]
    random.shuffle(speed_list2)

    speed_list = speed_list1 if video_seq % 2 == 0 else speed_list2
    tgt_class = 1 if video_seq % 2 == 1 else 2
    area_seq_move = random.randint(0, 8)
    areas_end_point = [0, 0, 0, 0]
    area_choose1 = [[8, 1], [2, 3], [8, 1], [2, 3]]
    area_choose2 = [[4, 5], [6, 7], [4, 5], [6, 7]]
    area_choose = area_choose1 if video_seq % 2 == 0 else area_choose2

    for track_id in range(1, 5):
        track_area = area_seq[(track_id + area_seq_move) % 4]
        track_speed = speed_list[(track_id - 1) % len(speed_list)] * speed_fix_rate
        areas_end_point[(track_id + area_seq_move) % 4] = 90 + int(min(max(50, int(50 / track_speed)), 400))

        start_x, start_y, start_angle = choose_start_with_min_dist(track_area, screen_size, existing_starts, min_init_dist=35)
        existing_starts.append((start_x, start_y))
        if len(existing_starts) > 4:
            existing_starts = existing_starts[1:]

        start_frame = 1
        start_pos = [start_x, start_y]
        generate_hard_curve(start_frame, track_id, start_pos, start_angle, track_speed, track_all, screen_size, tgt_class)

    for track_id in range(5, 25):
        first_valid_area = areas_end_point.index(min(areas_end_point))
        track_speed = speed_list[(track_id - 1) % len(speed_list)] * speed_fix_rate
        direction = track_id % 2
        track_area = area_choose[first_valid_area][0] if direction == 0 else area_choose[first_valid_area][1]
        
        start_x, start_y, start_angle = choose_start_with_min_dist(track_area, screen_size, existing_starts, min_init_dist=35)
        existing_starts.append((start_x, start_y))
        if len(existing_starts) > 4:
            existing_starts = existing_starts[1:]
            
        start_frame = areas_end_point[first_valid_area] - 50 + random.randint(-20, 100)
        start_pos = [start_x, start_y]
        areas_end_point[first_valid_area] += (95 + int(min(max(50, int(50 / track_speed)), 400)))
        generate_hard_curve(start_frame, track_id, start_pos, start_angle, track_speed, track_all, screen_size, tgt_class)

    sorted_tracks = sorted(track_all, key=lambda x: x[0])
    mkdirs('data/gts')
    txt_name = 'data/gts/{:03d}.txt'.format(video_seq)
    with open(txt_name, 'w') as f:
        for row in sorted_tracks:
            row_str = ','.join(str(value) for value in row)
            f.write(row_str + '\n')
