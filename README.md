# bop_ycb_renderer

### installation
```
git clone https://github.com/limhoyeon/bop_ycb_renderer.git
cd bop_ycb_renderer/ycb_render
sudo apt-get install libassimp-dev
pip install -r requirement.txt
export LD_LIBRARY_PATH=/usr/lib/nvidia-<vvv>:$LD_LIBRARY_PATH
python setup.py develop
```

### how to start
In example.py, change obj_parent_path and sample_data_sequence_parent_path
```
obj_parent_path = "/home/hoyeon/6dpose/bop_data_loader/bop_ycb_renderer/sample_data/models_fine" # path to sample object

obj_paths = glob(obj_parent_path+"/*.ply")
texture_paths = [x.replace(".ply", ".png") for x in obj_paths]
obj_paths, texture_paths = sorted(obj_paths), sorted(texture_paths)

sample_data_sequence_parent_path = "/home/hoyeon/6dpose/bop_data_loader/bop_ycb_renderer/sample_data/train_real" # path to sample data 
```

and run with
```
python example.py
```


### reference
1. ycb_renderer - https://github.com/NVlabs/PoseRBPF/blob/master/ycb_render/README.md
2. bop_toolkit - https://github.com/thodan/bop_toolkit
