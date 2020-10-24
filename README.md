# CVPR2020 Pose-guided Visible Part Matching for Occluded Person ReID
This is the pytorch implementation `on windows` of the CVPR2020 paper *"Pose-guided Visible Part Matching for Occluded Person ReID"*

# Contact
Email: nickhuang1996@126.com

## Modification
1. I fixed the way to load the model weights, or you will have severe errors when model weights are loaded:  
```UnicodeDecodeError: 'utf-8' codec can't decode byte 0xbc in position 0: invalid start byte```
2. I fixed the [rank_cy.pyx](torchreid/metrics/rank_cylib/rank_cy.pyx) and compiled successfully, so you just make this repository and test models.
3. I fixed the [setup.py](setup.py) so you can use cython to compile:  
`with open('README.md', encoding='latin1') as f:`

## Supplementary Specification
### Compile
- You need to download `Visual Studio 2017` or `Visual Studio 2019` to compile this repository.
- Run `python set.up develop`. If you compile successfully, the cmd is:
```
D:\project\Pycharm\PVPM-master>python setup.py develop
D:\project\Pycharm\PVPM-master\torchreid\metrics\rank.py:17: UserWarning: Cython evaluation (very fast so highly recommended) is unavailable, now use python evaluation.
  'Cython evaluation (very fast so highly recommended) is '
Compiling torchreid/metrics/rank_cylib/rank_cy.pyx because it changed.
[1/1] Cythonizing torchreid/metrics/rank_cylib/rank_cy.pyx
D:\software\Anaconda3\lib\site-packages\Cython\Compiler\Main.py:367: FutureWarning: Cython directive 'language_level' not set, using 2 for now (Py2). This will change in a later release! File: D:\project\Pycharm\PVPM-master\torchreid\metrics\rank_cylib\rank_cy.pyx
  tree = Parsing.p_module(s, pxd, full_module_name)
running develop
running egg_info
creating torchreid.egg-info
writing torchreid.egg-info\PKG-INFO
writing dependency_links to torchreid.egg-info\dependency_links.txt
writing requirements to torchreid.egg-info\requires.txt
writing top-level names to torchreid.egg-info\top_level.txt
writing manifest file 'torchreid.egg-info\SOURCES.txt'
reading manifest file 'torchreid.egg-info\SOURCES.txt'
writing manifest file 'torchreid.egg-info\SOURCES.txt'
running build_ext
building 'torchreid.metrics.rank_cylib.rank_cy' extension
creating build
creating build\temp.win-amd64-3.7
creating build\temp.win-amd64-3.7\Release
creating build\temp.win-amd64-3.7\Release\torchreid
creating build\temp.win-amd64-3.7\Release\torchreid\metrics
creating build\temp.win-amd64-3.7\Release\torchreid\metrics\rank_cylib
C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.27.29110\bin\HostX86\x64\cl.exe /c /nologo /Ox /W3 /GL /DNDEBUG /MT -ID:\software\Anaconda3\lib\site-packages\numpy\core\include -ID:\software\Anaconda3\include -ID:\software\Anaconda3\include "-IC:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.27.29110\ATLMFC\include" "-IC:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.27.29110\include" "-IC:\Program Files (x86)\Windows Kits\10\include\10.0.18362.0\ucrt" "-IC:\Program Files (x86)\Windows Kits\10\include\10.0.18362.0\shared" "-IC:\Program Files (x86)\Windows Kits\10\include\10.0.18362.0\um" "-IC:\Program Files (x86)\Windows Kits\10\include\10.0.18362.0\winrt" "-IC:\Program Files (x86)\Windows Kits\10\include\10.0.18362.0\cppwinrt" /Tctorchreid/metrics/rank_cylib/rank_cy.c /Fobuild\temp.win-amd64-3.7\Release\torchreid/metrics/rank_cylib/rank_cy.obj
rank_cy.c
D:\software\Anaconda3\lib\site-packages\numpy\core\include\numpy\npy_1_7_deprecated_api.h(14) : Warning Msg: Using deprecated NumPy API, disable it with #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
torchreid/metrics/rank_cylib/rank_cy.c(3810): warning C4244: “=”: 从“__int64”转换到“float”，可能丢失数据
torchreid/metrics/rank_cylib/rank_cy.c(4523): warning C4244: “=”: 从“double”转换到“float”，可能丢失数据
torchreid/metrics/rank_cylib/rank_cy.c(4553): warning C4244: “=”: 从“double”转换到“float”，可能丢失数据
torchreid/metrics/rank_cylib/rank_cy.c(5645): warning C4244: “=”: 从“__int64”转换到“float”，可能丢失数据
torchreid/metrics/rank_cylib/rank_cy.c(5808): warning C4244: “=”: 从“double”转换到“float”，可能丢失数据
torchreid/metrics/rank_cylib/rank_cy.c(5858): warning C4244: “=”: 从“double”转换到“float”，可能丢失数据
creating D:\project\Pycharm\PVPM-master\build\lib.win-amd64-3.7
creating D:\project\Pycharm\PVPM-master\build\lib.win-amd64-3.7\torchreid
creating D:\project\Pycharm\PVPM-master\build\lib.win-amd64-3.7\torchreid\metrics
creating D:\project\Pycharm\PVPM-master\build\lib.win-amd64-3.7\torchreid\metrics\rank_cylib
C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.27.29110\bin\HostX86\x64\link.exe /nologo /INCREMENTAL:NO /LTCG /nodefaultlib:libucrt.lib ucrt.lib /DLL /MANIFEST:EMBED,ID=2 /MANIFESTUAC:NO /LIBPATH:D:\software\Anaconda3\libs /LIBPATH:D:\software\Anaconda3\PCbuild\amd64 "/LIBPATH:C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.27.29110\ATLMFC\lib\x64" "/LIBPATH:C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.27.29110\lib\x64" "/LIBPATH:C:\Program Files (x86)\Windows Kits\10\lib\10.0.18362.0\ucrt\x64" "/LIBPATH:C:\Program Files (x86)\Windows Kits\10\lib\10.0.18362.0\um\x64" /EXPORT:PyInit_rank_cy build\temp.win-amd64-3.7\Release\torchreid/metrics/rank_cylib/rank_cy.obj /OUT:build\lib.win-amd64-3.7\torchreid\metrics\rank_cylib\rank_cy.cp37-win_amd64.pyd /IMPLIB:build\temp.win-amd64-3.7\Release\torchreid/metrics/rank_cylib\rank_cy.cp37-win_amd64.lib
  正在创建库 build\temp.win-amd64-3.7\Release\torchreid/metrics/rank_cylib\rank_cy.cp37-win_amd64.lib 和对象 build\temp.win-amd64-3.7\Release\torchreid/metrics/rank_cylib\rank_cy.cp37-win_amd64.exp
正在生成代码
已完成代码的生成
copying build\lib.win-amd64-3.7\torchreid\metrics\rank_cylib\rank_cy.cp37-win_amd64.pyd -> torchreid\metrics\rank_cylib
Creating d:\software\anaconda3\lib\site-packages\torchreid.egg-link (link to .)
torchreid 0.8.1 is already the active version in easy-install.pth

Installed d:\project\pycharm\pvpm-master
Processing dependencies for torchreid==0.8.1
Searching for torchvision==0.4.2
Best match: torchvision 0.4.2
Adding torchvision 0.4.2 to easy-install.pth file

Using d:\software\anaconda3\lib\site-packages
Searching for torch==1.4.0
Best match: torch 1.4.0
Adding torch 1.4.0 to easy-install.pth file
Installing convert-caffe2-to-onnx-script.py script to D:\software\Anaconda3\Scripts
Installing convert-caffe2-to-onnx.exe script to D:\software\Anaconda3\Scripts
Installing convert-onnx-to-caffe2-script.py script to D:\software\Anaconda3\Scripts
Installing convert-onnx-to-caffe2.exe script to D:\software\Anaconda3\Scripts

Using d:\software\anaconda3\lib\site-packages
Searching for scipy==1.2.1
Best match: scipy 1.2.1
Adding scipy 1.2.1 to easy-install.pth file

Using d:\software\anaconda3\lib\site-packages
Searching for six==1.12.0
Best match: six 1.12.0
Adding six 1.12.0 to easy-install.pth file

Using d:\software\anaconda3\lib\site-packages
Searching for Pillow==5.4.1
Best match: Pillow 5.4.1
Adding Pillow 5.4.1 to easy-install.pth file

Using d:\software\anaconda3\lib\site-packages
Searching for h5py==2.9.0
Best match: h5py 2.9.0
Adding h5py 2.9.0 to easy-install.pth file

Using d:\software\anaconda3\lib\site-packages
Searching for Cython==0.29.6
Best match: Cython 0.29.6
Adding Cython 0.29.6 to easy-install.pth file
Installing cygdb-script.py script to D:\software\Anaconda3\Scripts
Installing cygdb.exe script to D:\software\Anaconda3\Scripts
Installing cython-script.py script to D:\software\Anaconda3\Scripts
Installing cython.exe script to D:\software\Anaconda3\Scripts
Installing cythonize-script.py script to D:\software\Anaconda3\Scripts
Installing cythonize.exe script to D:\software\Anaconda3\Scripts

Using d:\software\anaconda3\lib\site-packages
Searching for numpy==1.16.2
Best match: numpy 1.16.2
Adding numpy 1.16.2 to easy-install.pth file
Installing f2py-script.py script to D:\software\Anaconda3\Scripts
Installing f2py.exe script to D:\software\Anaconda3\Scripts

Using d:\software\anaconda3\lib\site-packages
Finished processing dependencies for torchreid==0.8.1

``` 

## Dependencies
-Python2.7 or Python>=3.6\
-Pytorch>=1.0\
-Numpy

## Related Project
Our code is based on [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid). We adopt [openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) to extract pose landmarks and part affinity fields.

## Dataset Preparation
Download the raw datasets [Occluded-REID, P-DukeMTMC-reID](https://github.com/tinajia2012/ICME2018_Occluded-Person-Reidentification_datasets), and [Partial-Reid](https://pan.baidu.com/s/1VhPUVJOLvkhgbJiUoEnJWg) (code:zdl8) which is released by [Partial Person Re-identification](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/Zheng_Partial_Person_Re-Identification_ICCV_2015_paper.html). Instructions regarding how to prepare [Market1501](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf) datasets can be found [here](https://kaiyangzhou.github.io/deep-person-reid/datasets.html). And then place them under the directory like:

```
PVPM_experiments/data/
├── ICME2018_Occluded-Person-Reidentification_datasets
│   ├── Occluded_Duke
│   └── Occluded_REID
├── Market-1501-v15.09.15
└── Partial-REID_Dataset
```

## Pose extraction
Install openopse as described [here](https://github.com/CMU-Perceptual-Computing-Lab/openpose).\
Change path to your own dataset root and run sh files in /scripts:
```
sh openpose_occluded_reid.sh
sh openpose_market.sh
``` 
Extracted Pose information can be found [here](https://pan.baidu.com/s/1Majze1iFo7FytREijmQO5A)(code:iwlz)

## To Train PCB baseline

``` 
python scripts/main.py --root PATH_TO_DATAROOT \
 -s market1501 -t market1501\
 --save-dir PATH_TO_EXPERIMENT_FOLDER/market_PCB\
 -a pcb_p6 --gpu-devices 0 --fixbase-epoch 0\
 --open-layers classifier fc\
 --new-layers classifier em\
 --transforms random_flip\
 --optim sgd --lr 0.02\
 --stepsize 25 50\
 --staged-lr --height 384 --width 192\
 --batch-size 32 --base-lr-mult 0.5
```
## To train PVPM
```
python scripts/main.py --load-pose --root PATH_TO_DATAROOT
 -s market1501\
 -t occlusion_reid p_duke partial_reid\
 --save-dir PATH_TO_EXPERIMENT_FOLDER/PVPM\
 -a pose_p6s --gpu-devices 0\
 --fixbase-epoch 30\
 --open-layers pose_subnet\
 --new-layers pose_subnet\
 --transforms random_flip\
 --optim sgd --lr 0.02\
 --stepsize 15 25 --staged-lr\
 --height 384 --width 128\
 --batch-size 32\
 --start-eval 20\
 --eval-freq 10\
 --load-weights PATH_TO_EXPERIMENT_FOLDER/market_PCB/model.pth.tar-60\
 --train-sampler RandomIdentitySampler\
 --reg-matching-score-epoch 0\
 --graph-matching
 --max-epoch 30
 --part-score
```
Trained PCB model and PVPM model can be found [here](https://pan.baidu.com/s/16lr8m-wv-XOXACqIthC8lw)(code:64zy)

## EXTRA
## To test
- Set `--workers` to 0, or `pin_memory error` will be occured.
- `-evaluate` must be here.
- `--load_weights` should be  path of download models.

Then you can `python scripts/main.py`
# Evaluation Results
- Pretrain_PCB_model.pth.tar-60
    
    [Market1501 to Market1501](#Market1501 to Market1501)
- PVPM_model.pth.tar-30
    
    [Market1501 to Occlusion_reid and partial_reid](#Market1501 to Occlusion_reid and partial_reid)
    
## Pretrain_PCB_model.pth.tar-60
### Market1501 to Market1501
```
--root
PATH_TO_DATAROOT
-s
market1501
-t
market1501
--save-dir
PATH_TO_EXPERIMENT_FOLDER/market_PCB
-a
pcb_p6
--gpu-devices
0
--fixbase-epoch
0
--open-layers
classifier
fc
--new-layers
classifier
em
--transforms
random_flip
--evaluate
--start-eval
60
--load-weights
PATH_TO_EXPERIMENT_FOLDER/market_PCB/Pretrain_PCB_model.pth.tar-60
--optim
sgd
--lr
0.02
--stepsize
25
50
--staged-lr
--height
384
--width
192
--batch-size
128
--base-lr-mult
0.5
--workers
0
```
### Results
```
** Arguments **
adam_beta1: 0.9
adam_beta2: 0.999
app: image
arch: pcb_p6
base_lr_mult: 0.5
batch_size: 128
combineall: False
cuhk03_classic_split: False
cuhk03_labeled: False
dist_metric: euclidean
eval_freq: -1
evaluate: True
fixbase_epoch: 0
gamma: 0.1
gpu_devices: 0
graph_matching: False
height: 384
label_smooth: False
load_pose: False
load_weights: D:/weights_results/PVPM/pretrained_models/Pretrain_PCB_model.pth.tar-60
loss: softmax
lr: 0.02
lr_scheduler: multi_step
margin: 0.3
market1501_500k: False
max_epoch: 60
momentum: 0.9
new_layers: ['classifier', 'em']
no_pretrained: False
normalize_feature: False
num_att: 6
num_instances: 4
open_layers: ['classifier', 'fc']
optim: sgd
part_score: False
pooling_method: avg
print_freq: 20
ranks: [1, 3, 5, 10, 20]
reg_matching_score_epoch: 5
rerank: False
resume: 
rmsprop_alpha: 0.99
root: D:/datasets/ReID_dataset
sample_method: evenly
save_dir: D:/weights_results/PVPM/market_PCB
seed: 1
seq_len: 15
sgd_dampening: 0
sgd_nesterov: False
sources: ['market1501']
split_id: 0
staged_lr: True
start_epoch: 0
start_eval: 60
stepsize: [25, 50]
targets: ['market1501']
train_sampler: RandomSampler
transforms: ['random_flip']
use_att_loss: False
use_avai_gpus: False
use_cpu: False
use_metric_cuhk03: False
visrank: False
visrank_topk: 20
weight_decay: 0.0005
weight_t: 1
weight_x: 0
width: 192
workers: 0


Collecting env info ...
** System info **
PyTorch version: 1.4.0
Is debug build: No
CUDA used to build PyTorch: 10.1

OS: Microsoft Windows 10 企业版
GCC version: Could not collect
CMake version: Could not collect

Python version: 3.7
Is CUDA available: Yes
CUDA runtime version: 10.1.105
GPU models and configuration: Could not collect
Nvidia driver version: Could not collect
cuDNN version: Could not collect

Versions of relevant libraries:
[pip] numpy==1.16.2
[pip] numpydoc==0.8.0
[pip] torch==1.4.0
[pip] torchreid==0.8.1
[pip] torchstat==0.0.7
[pip] torchsummary==1.5.1
[pip] torchvision==0.4.2
[conda] blas                      1.0                         mkl  
[conda] mkl                       2019.3                      203  
[conda] mkl-service               1.1.2            py37hb782905_5  
[conda] mkl_fft                   1.0.10           py37h14836fe_0  
[conda] mkl_random                1.0.2            py37h343c172_0  
[conda] torch                     1.4.0                    pypi_0    pypi
[conda] torchreid                 0.8.1                     dev_0    <develop>
[conda] torchstat                 0.0.7                    pypi_0    pypi
[conda] torchsummary              1.5.1                    pypi_0    pypi
[conda] torchvision               0.4.2                    pypi_0    pypi
        Pillow (5.4.1)

Building train transforms ...
+ resize to 384x192
+ random flip
+ to torch tensor of range [0, 1]
+ normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
Building test transforms ...
+ resize to 384x192
+ to torch tensor of range [0, 1]
+ normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
=> Loading train (source) dataset
=> Loaded Market1501
  ----------------------------------------
  subset   | # ids | # images | # cameras
  ----------------------------------------
  train    |   751 |    12936 |         6
  query    |   750 |     3368 |         6
  gallery  |   751 |    15913 |         6
  ----------------------------------------
=> Loading test (target) dataset
=> Loaded Market1501
  ----------------------------------------
  subset   | # ids | # images | # cameras
  ----------------------------------------
  train    |   751 |    12936 |         6
  query    |   750 |     3368 |         6
  gallery  |   751 |    15913 |         6
  ----------------------------------------


  **************** Summary ****************
  train            : ['market1501']
  # train datasets : 1
  # train ids      : 751
  # train images   : 12936
  # train cameras  : 6
  test             : ['market1501']
  *****************************************


Building model: pcb_p6
Successfully loaded pretrained weights from "D:/weights_results/PVPM/pretrained_models/Pretrain_PCB_model.pth.tar-60"
Building softmax-engine for image-reid
##### Evaluating market1501 (source) #####
Extracting features from query set ...
Done, obtained 3368-by-12288 matrix
Extracting features from gallery set ...
Done, obtained 15913-by-12288 matrix
Speed: 0.0929 sec/batch
Computing distance matrix with metric=euclidean ...
Computing CMC and mAP ...
** Results **
mAP: 73.4%
CMC curve
Rank-1  : 90.6%
Rank-3  : 95.0%
Rank-5  : 96.6%
Rank-10 : 97.6%
Rank-20 : 98.5%

```
## PVPM_model.pth.tar-30
### Market1501 to Occlusion_reid and partial_reid
```
--evaluate
--start-eval
30
--load-pose
--root
D:/datasets/ReID_dataset
-s
market1501
-t
occlusion_reid
partial_reid
--save-dir
D:/weights_results/PVPM/occlusion_reid_and_partial_reid
-a
pose_p6s
--gpu-devices
0
--fixbase-epoch
30
--open-layers
pose_subnet
--new-layers
pose_subnet
--transforms
random_flip
--optim
sgd
--lr
0.02
--stepsize
15
25
--staged-lr
--height
384
--width
128
--batch-size
128
--start-eval
20
--eval-freq
10
--load-weights
D:/weights_results/PVPM/pretrained_models/PVPM_model.pth.tar-30
--train-sampler
RandomIdentitySampler
--reg-matching-score-epoch
0
--graph-matching
--max-epoch
30
--part-score
--workers
0
```
### Results
```
** Arguments **
adam_beta1: 0.9
adam_beta2: 0.999
app: image
arch: pose_p6s
base_lr_mult: 0.1
batch_size: 128
combineall: False
cuhk03_classic_split: False
cuhk03_labeled: False
dist_metric: euclidean
eval_freq: 10
evaluate: True
fixbase_epoch: 30
gamma: 0.1
gpu_devices: 0
graph_matching: True
height: 384
label_smooth: False
load_pose: True
load_weights: D:/weights_results/PVPM/pretrained_models/PVPM_model.pth.tar-30
loss: softmax
lr: 0.02
lr_scheduler: multi_step
margin: 0.3
market1501_500k: False
max_epoch: 30
momentum: 0.9
new_layers: ['pose_subnet']
no_pretrained: False
normalize_feature: False
num_att: 6
num_instances: 4
open_layers: ['pose_subnet']
optim: sgd
part_score: True
pooling_method: avg
print_freq: 20
ranks: [1, 3, 5, 10, 20]
reg_matching_score_epoch: 0
rerank: False
resume: 
rmsprop_alpha: 0.99
root: D:/datasets/ReID_dataset
sample_method: evenly
save_dir: D:/weights_results/PVPM
seed: 1
seq_len: 15
sgd_dampening: 0
sgd_nesterov: False
sources: ['market1501']
split_id: 0
staged_lr: True
start_epoch: 0
start_eval: 20
stepsize: [15, 25]
targets: ['occlusion_reid', 'partial_reid']
train_sampler: RandomIdentitySampler
transforms: ['random_flip']
use_att_loss: False
use_avai_gpus: False
use_cpu: False
use_metric_cuhk03: False
visrank: False
visrank_topk: 20
weight_decay: 0.0005
weight_t: 1
weight_x: 0
width: 128
workers: 0


Collecting env info ...
** System info **
PyTorch version: 1.4.0
Is debug build: No
CUDA used to build PyTorch: 10.1

OS: Microsoft Windows 10 企业版
GCC version: Could not collect
CMake version: Could not collect

Python version: 3.7
Is CUDA available: Yes
CUDA runtime version: 10.1.105
GPU models and configuration: Could not collect
Nvidia driver version: Could not collect
cuDNN version: Could not collect

Versions of relevant libraries:
[pip] numpy==1.16.2
[pip] numpydoc==0.8.0
[pip] torch==1.4.0
[pip] torchreid==0.8.1
[pip] torchstat==0.0.7
[pip] torchsummary==1.5.1
[pip] torchvision==0.4.2
[conda] blas                      1.0                         mkl  
[conda] mkl                       2019.3                      203  
[conda] mkl-service               1.1.2            py37hb782905_5  
[conda] mkl_fft                   1.0.10           py37h14836fe_0  
[conda] mkl_random                1.0.2            py37h343c172_0  
[conda] torch                     1.4.0                    pypi_0    pypi
[conda] torchstat                 0.0.7                    pypi_0    pypi
[conda] torchsummary              1.5.1                    pypi_0    pypi
[conda] torchvision               0.4.2                    pypi_0    pypi
        Pillow (5.4.1)

Building train transforms ...
+ resize to 384x128
+ random flip
+ to torch tensor of range [0, 1]
+ normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
Building test transforms ...
+ resize to 384x128
+ to torch tensor of range [0, 1]
+ normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
=> Loading train (source) dataset
=> Loaded Market1501
  ----------------------------------------
  subset   | # ids | # images | # cameras
  ----------------------------------------
  train    |   751 |    12936 |         6
  query    |   750 |     3368 |         6
  gallery  |   751 |    15913 |         6
  ----------------------------------------
=> Loading test (target) dataset
=> Loaded Occluded_REID
  ----------------------------------------
  subset   | # ids | # images | # cameras
  ----------------------------------------
  train    |     0 |        0 |         0
  query    |   200 |     1000 |         1
  gallery  |   200 |     1000 |         1
  ----------------------------------------
=> Loaded Paritial_REID
  ----------------------------------------
  subset   | # ids | # images | # cameras
  ----------------------------------------
  train    |     0 |        0 |         0
  query    |    60 |      300 |         1
  gallery  |    60 |      300 |         1
  ----------------------------------------


  **************** Summary ****************
  train            : ['market1501']
  # train datasets : 1
  # train ids      : 751
  # train images   : 12936
  # train cameras  : 6
  test             : ['occlusion_reid', 'partial_reid']
  *****************************************


Building model: pose_p6s
Successfully loaded pretrained weights from "D:/weights_results/PVPM/pretrained_models/PVPM_model.pth.tar-30"
Building softmax-engine for image-reid
##### Evaluating occlusion_reid (target) #####
Extracting features from query set ...
Done, obtained 1000-by-2048 matrix
Extracting features from gallery set ...
Done, obtained 1000-by-2048 matrix
Speed: 1.1385 sec/batch
Computing distance matrix with metric=euclidean ...
Computing CMC and mAP ...
** Results **
mAP: 60.2%
CMC curve
Rank-1  : 67.2%
Rank-3  : 77.6%
Rank-5  : 82.4%
Rank-10 : 88.3%
Rank-20 : 92.3%
##### Evaluating partial_reid (target) #####
Extracting features from query set ...
Done, obtained 300-by-2048 matrix
Extracting features from gallery set ...
Done, obtained 300-by-2048 matrix
Speed: 0.9049 sec/batch
Computing distance matrix with metric=euclidean ...
Computing CMC and mAP ...
** Results **
mAP: 71.4%
CMC curve
Rank-1  : 75.3%
Rank-3  : 85.0%
Rank-5  : 88.7%
Rank-10 : 92.3%
Rank-20 : 96.3%

```

# Citation
If you find this code useful to your research, please cite the following paper:
>@inproceedings{gao2020pose,  
  title={Pose-guided Visible Part Matching for Occluded Person ReID},  
  author={Gao, Shang and Wang, Jingya and Lu, Huchuan and Liu, Zimo},  
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},  
  pages={11744--11752},  
  year={2020}  
}





