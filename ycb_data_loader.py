import torch
from glob import glob
import json
import os
import ycb_render.ycb_renderer
import numpy as np
import gen_utils as gu
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

def get_rendered_img_with_mask(renderer, camera_calibration_mat, object_rotation_mat, object_translation_vec, width, height):
    w = 640
    h = 480

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

class object_renderer:
    def __init__(self, width, height, sequence_path_ls):
        train_sequnece_path_ls = glob(os.path.join("/","home","hoyeon", "6dpose", "bop_data", "ycbv", "train_real","*"))
        train_sequnece_path_ls = sorted(train_sequnece_path_ls)
        sequence_info_map = {}
        for i in range(len(train_sequnece_path_ls)):
            sequence_info_map.update(get_sequence_info_map(train_sequnece_path_ls[i], "train_" + os.path.basename(train_sequnece_path_ls[i])))

        sequence_keys = []

#test
if __name__ == "__main__":
    train_sequnece_path_ls = glob(os.path.join("/","home","hoyeon", "6dpose", "bop_data", "ycbv", "train_real","*"))
    train_sequnece_path_ls = sorted(train_sequnece_path_ls)
    sequence_info_map = {}
    for i in range(len(train_sequnece_path_ls)):
        if i!=27:
            continue
        sequence_info_map.update(get_sequence_info_map(train_sequnece_path_ls[i], "train_" + os.path.basename(train_sequnece_path_ls[i])))


    # get one sample instance info
    sampled_instance_id = 'train_000027_1'
    rgb_path = sequence_info_map[sampled_instance_id]['rgb_path']
    cam_k = sequence_info_map[sampled_instance_id]['seqeunce_camera_info']['cam_K']

    for i in range(len(sequence_info_map[sampled_instance_id]['scene_gt_pose_map'])):
        rot, trans, obj_id = sequence_info_map[sampled_instance_id]['scene_gt_pose_map'][i].values()
        transform_mat = get_transform_matrix(rot, trans)
        gt_bbox = sequence_info_map[sampled_instance_id]['sequence_gt_meta_info_map'][i]['bbox_obj']
        obj_paths = glob(os.path.join("/","home","hoyeon", "6dpose", "bop_data", "ycbv", "models", "*.ply"))
        texture_paths = [x.replace(".ply", ".png") for x in obj_paths]
        obj_paths, texture_paths = sorted(obj_paths)[obj_id - 1:obj_id], sorted(texture_paths)[obj_id - 1:obj_id]

        renderer = ycb_render.ycb_renderer.YCBRenderer(
            640, 
            480, 
            obj_paths,
            texture_paths
        )

        img = cv2.imread(rgb_path)
        rgb_tensor_1, mask_tensor, img, img_mask = renderer.get_deepim_input_set(img, gt_bbox, cam_k, transform_mat)

        # for check
        rgb_tensor_1 = gu.torch_to_numpy(rgb_tensor_1.permute(1,2,0))
        mask_tensor = gu.torch_to_numpy(mask_tensor.permute(1,2,0))
        img = gu.torch_to_numpy(img.permute(1,2,0))
        img_mask = gu.torch_to_numpy(img_mask.permute(1,2,0))

        gu.imshow(rgb_tensor_1)
        gu.imshow(mask_tensor)
        gu.imshow(img)
        gu.imshow(img_mask)
    exit()
    while 1:
        rgb_tensor_1, mask_tensor = renderer.get_one_obj_rendered_img_with_mask(

            0,
            cam_k,
            rot,
            trans
        )
        import numpy as np
        import gen_utils as gu
        import cv2

        img = cv2.imread(rgb_path)
        img = np.array(img)
        rgb_tensor_1 = gu.torch_to_numpy(rgb_tensor_1)[:,:,:3]
        mask_tensor = gu.torch_to_numpy(mask_tensor)[:,:,:3]
        
        cv2.imshow('test', cv2.cvtColor(rgb_tensor_1, cv2.COLOR_RGB2BGR))
        q = cv2.waitKey(0)
        if q == ord('b'):
            trans += np.array([0,0,-10])
        elif q == ord('t'):
            trans += np.array([0,0,10])
        elif q == ord('a'):
            trans += np.array([10,0,0])
        elif q==ord('d'):
            trans += np.array([-10,0,0])
        elif q==ord('w'):
            trans += np.array([0,-10,0])
        elif q==ord('s'):
            trans += np.array([0,10,0])

        print(trans)
        cv2.destroyAllWindows()

        obj_paths, texture_paths = sorted(obj_paths)[obj_id - 1:obj_id], sorted(texture_paths)[obj_id - 1:obj_id]
        renderer = ycb_render.ycb_renderer.YCBRenderer(
            640, 
            480, 
            obj_paths,
            texture_paths
        )

        rgb_tensor_1, mask_tensor, center = renderer.get_one_obj_rendered_img_with_mask(
            0,
            cam_k,
            transform_mat
        )
        
        rgb_tensor_1 = gu.torch_to_numpy(rgb_tensor_1)[:,:,:3]
        rgb_tensor_1 = draw_circle(rgb_tensor_1, center)
        mask_tensor = gu.torch_to_numpy(mask_tensor)[:,:,:3]
        gu.imshow(rgb_tensor_1)

        zoom_bbox = renderer.get_zoom_bbox(center, gt_bbox, mask_tensor) 
        

        img = cv2.imread(rgb_path)
        img = np.array(img)
        img_copy = img.copy()
        img_copy[rgb_tensor_1!=0] = rgb_tensor_1[rgb_tensor_1!=0]
        gu.imshow(img_copy)