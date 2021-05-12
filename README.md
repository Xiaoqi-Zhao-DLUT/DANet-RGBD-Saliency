<p align="center">

  <img src="./Image/DANet_logo.png" alt="Logo" width="210" height="auto">


  <h3 align="center">A Single Stream Network for Robust and Real-time RGB-D Salient Object Detection</h3>

  <p align="center">
    Xiaoqi Zhao, Lihe Zhang, Youwei Pang, Huchuan Lu, Lei Zhang
    <br />
    <a href="https://arxiv.org/pdf/2007.06811.pdf"><strong>⭐ arXiv »</strong></a>
    <br />
  </p>
</p>

The official repo of the ECCV 2020 paper A Single Stream Network for Robust and Real-time RGB-D Salient Object Detection.  
## Saliency map
[Google Drive](https://drive.google.com/file/d/1CHCLvM1wUl2O4AVMel5WHf8icexojwHY/view?usp=sharing ) / [BaiduYunPan(3m9i)](https://pan.baidu.com/s/1_sOsCOgZwFNtPdXypFJHog)  
## Related Works
* (ECCV 2020 Oral) Suppress and Balance: A Simple Gated Network for Salient Object Detection: https://github.com/Xiaoqi-Zhao-DLUT/GateNet-RGB-Saliency
* (ECCV 2020) Hierarchical Dynamic Filtering Network for RGB-D Salient Object Detection: https://github.com/lartpang/HDFNet
* (CVPR 2020) Multi-scale Interactive Network for Salient Object Detection: https://github.com/lartpang/MINet

## Network
![](./Image/Network.png)

## Module
![](./Image/Module.png)

## Quantitative comparison
![](./Image/Quantitative_comparison.png)

## Visual comparison
![](./Image/visual_4channel.png)

![](./Image/visual_deda.png)  

## Trained Model
You can download the trained VGG16-model(DUT-RGBD or NJUD&NLPR) at [BaiduYunPan(5uhd)](https://pan.baidu.com/s/1XJziVUSlRynU_yUHA86cpg).
## Requirement
* Python 3.7
* PyTorch 1.5.0
* torchvision
* numpy
* Pillow
* Cython
## Training
1.Set the path of training sets in config.py  
2.Run train.py
## Testing
1.Set the path of testing sets in config.py    
2.Run generate_salmap.py (can generate the predicted saliency maps)  
3.Run generate_visfeamaps.py (can visualize feature maps)  
4.Run test_metric_score.py (can evaluate the predicted saliency maps in terms of fmax,fmean,wfm,sm,em,mae). You also can use the toolkit released by us:https://github.com/lartpang/Py-SOD-VOS-EvalToolkit.

## BibTex
```
@inproceedings{DANet,
  title={A Single Stream Network for Robust and Real-time RGB-D Salient Object Detection},
  author={Zhao, Xiaoqi and Zhang, Lihe and Pang, Youwei and Lu, Huchuan and Zhang, Lei},
  booktitle=ECCV,
  year={2020}
}
```
