# Real-Time Exposure Correction via Collaborative Transformations and Adaptive Sampling (CVPR 2024)

PyTorch Implementation of paper: Real-Time Exposure Correction via Collaborative Transformations and Adaptive Sampling.


### Setup
- Install the conda environment
```
conda create -n cotf python=3.7
conda activate cotf
```
- Install Pytroch
```
# CUDA 10.2
conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=10.2 -c pytorch
```
- Install other requirements, such as basicsr.
```
pip install -r requirements.txt
```
### Dataset
Please refer to the link below to download the dataset, place it in the . /datasets folder.
Organize the data like the example in ./datasets/example.
Then remember to modify the dataset options in OPTIONS accordingly.
- LCDP https://www.whyy.site/paper/lcdp
- MSEC https://github.com/mahmoudnafifi/Exposure_Correction
- SICE https://github.com/KevinJ-Huang/ExposureNorm-Compensation
### Run
- Testing. The testing configuration is in options/test/.
We put all the pre-training weights in the ./experiments.
After running, the results of the visualization will be saved in . /results.
```
python test.py -opt options/test/test_lcdp.yml
python test.py -opt options/test/test_msec.yml
python test.py -opt options/test/test_sice.yml
```
- Training. The training configuration is in options/train/.
```
python train.py -opt options/train/train_lcdp.yml
```

### Citation
Please cite the following paper if you feel our work useful to your research:
```
@InProceedings{li2024cotf,
    title     = {Real-Time Exposure Correction via Collaborative Transformations and Adaptive Sampling},
    author    = {Ziwen Li, Feng Zhang, Meng Cao, Jinpu Zhang, Yuanjie Shao, Yuehuan Wang, Nong Sang},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year      = {2024}
}
```