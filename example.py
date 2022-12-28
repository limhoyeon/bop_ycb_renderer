import torch
from glob import glob
import json
import os
import ycb_render.ycb_renderer
import numpy as np
import utils.gen_utils as gu
import utils.ycb_utils as yu
import cv2

if __name__ == "__main__":
    
    obj_parent_path = "/home/hoyeon/6dpose/bop_data_loader/bop_ycb_renderer/sample_data/models_fine" # path to sample object
    
    obj_paths = glob(obj_parent_path+"/*.ply")
    texture_paths = [x.replace(".ply", ".png") for x in obj_paths]
    obj_paths, texture_paths = sorted(obj_paths), sorted(texture_paths)

    sample_data_sequence_parent_path = "/home/hoyeon/6dpose/bop_data_loader/bop_ycb_renderer/sample_data/train_real" # path to sample data 
    # sequence means one video(img sequence).
    sequnece_path_ls = glob(sample_data_sequence_parent_path + "/*")
    sequence_info_map = {}
    for i in range(len(sequnece_path_ls)):
        sequence_info_map.update(yu.get_sequence_info_map(sequnece_path_ls[i], "test_" + os.path.basename(sequnece_path_ls[i])))

    # get one sample scene info. scene info means rgb, depth img path and meta data(gt bbox, pose) 
    sampled_scene_id = 'test_000027_1'
    rgb_path = sequence_info_map[sampled_scene_id]['rgb_path']
    cam_k = sequence_info_map[sampled_scene_id]['seqeunce_camera_info']['cam_K']

    for item_idx in range(len(sequence_info_map[sampled_scene_id]['scene_gt_pose_map'])):
        # item_idx means first item in the img of one sample scene

        rot, trans, obj_id = sequence_info_map[sampled_scene_id]['scene_gt_pose_map'][item_idx].values()
        transform_mat = yu.get_transform_matrix(rot, trans)

        gt_bbox = sequence_info_map[sampled_scene_id]['sequence_gt_meta_info_map'][item_idx]['bbox_obj']

        # set vertices & texture of object
        renderer = ycb_render.ycb_renderer.YCBRenderer(
            640, 
            480, 
            obj_paths[obj_id - 1:obj_id],
            texture_paths[obj_id - 1:obj_id]
        )

        img = cv2.imread(rgb_path)

        # rendering part!! // get data as DEEPIM input format
        rgb_tensor_1, mask_tensor, img, img_mask = renderer.get_deepim_input_set(img, gt_bbox, cam_k, transform_mat)

        # for result check
        rgb_tensor_1 = gu.torch_to_numpy(rgb_tensor_1.permute(1,2,0))
        mask_tensor = gu.torch_to_numpy(mask_tensor.permute(1,2,0))
        img = gu.torch_to_numpy(img.permute(1,2,0))
        img_mask = gu.torch_to_numpy(img_mask.permute(1,2,0))

        gu.imshow(rgb_tensor_1)
        gu.imshow(mask_tensor)
        gu.imshow(img)
        gu.imshow(img_mask)
