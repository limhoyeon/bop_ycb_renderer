from glob import glob
import json
import os
import numpy as np
import cv2

def get_instance_id_from_file_name(file_name, sequence_id):
    return sequence_id + "_" + str(int(os.path.basename(file_name).split(".")[0].split("_")[0]))

def read_json(file_path):
    with open(file_path, "r") as st_json:
        st_python = json.load(st_json)
    return st_python

def get_sequence_info_map(sequence_path, sequence_id):
    data_info_map = {}

    rgb_path_ls = glob(os.path.join(sequence_path, "rgb", "*"))
    mask_visib_ls = glob(os.path.join(sequence_path, "mask_visib", "*"))
    mask_ls = glob(os.path.join(sequence_path, "mask", "*"))
    depth_path_ls = glob(os.path.join(sequence_path, "depth", "*"))
    sequence_gt_map = read_json(os.path.join(sequence_path,"scene_gt.json"))
    sequence_gt_info_map = read_json(os.path.join(sequence_path,"scene_gt_info.json"))
    seqeunce_camera_map = read_json(os.path.join(sequence_path, "scene_camera.json"))
    
    for path in rgb_path_ls:
        instance_id = get_instance_id_from_file_name(path, sequence_id)
        data_info_map[instance_id] = {}
    
    for path in rgb_path_ls:
        instance_id = get_instance_id_from_file_name(path, sequence_id)
        data_info_map[instance_id]["rgb_path"] = path

    for path in mask_ls:
        instance_id = get_instance_id_from_file_name(path, sequence_id)
        if "mask_path_ls" not in data_info_map[instance_id]:
            data_info_map[instance_id]["mask_path_ls"] = []
        data_info_map[instance_id]["mask_path_ls"].append(path)

    for path in mask_visib_ls:
        instance_id = get_instance_id_from_file_name(path, sequence_id)
        if "mask_visib_path_ls" not in data_info_map[instance_id]:
            data_info_map[instance_id]["mask_visib_path_ls"] = []
        data_info_map[instance_id]["mask_visib_path_ls"].append(path)

    for path in depth_path_ls:
        instance_id = get_instance_id_from_file_name(path, sequence_id)
        data_info_map[instance_id]["depth_path"] = path

    for key in sequence_gt_map:
        instance_id = sequence_id + "_" + key
        data_info_map[instance_id]["scene_gt_pose_map"] = sequence_gt_map[key]

    for key in sequence_gt_info_map:
        instance_id = sequence_id + "_" + key
        data_info_map[instance_id]["sequence_gt_meta_info_map"] = sequence_gt_info_map[key]

    for key in seqeunce_camera_map:
        instance_id = sequence_id + "_" + key
        data_info_map[instance_id]["seqeunce_camera_info"] = seqeunce_camera_map[key]

    return data_info_map

def draw_bbox(img, bbox):
    """
        bbox_format: top_left_corner x, top_left_corner y, width, height 
    """
    img = img.copy()
    
    top_left = (bbox[0], bbox[1])
    bot_left = (bbox[0], bbox[1]+bbox[3])
    top_right = (bbox[0]+bbox[2], bbox[1])
    bot_right = (bbox[0]+bbox[2], bbox[1]+bbox[3])

    img = cv2.line(img, top_left, top_right, [255,255,255])
    img = cv2.line(img, top_left, bot_left, [255,255,255])
    img = cv2.line(img, top_right, bot_right, [255,255,255])
    img = cv2.line(img, bot_left, bot_right, [255,255,255])
    return img

def draw_circle(img, circle_center):
    img = img.copy()
    img = cv2.circle(img, tuple(np.round(circle_center).astype(int)), 1, [255,255,255], 3)
    return img 

def get_transform_matrix(rot, trans):
    if len(rot)==9:
        rot = np.array(rot).reshape(3,3)
        
    transform_mat = np.eye(4, dtype=np.float32)
    transform_mat[:3, :3], transform_mat[:3, 3] = rot, trans
    return transform_mat
