# ILDM

Industrial Large-Scale Diagnosis Model with Personalized Deployment for Non-IID Diagnosis Tasks

## System
```
├── Dataset/                */Used datasets
│   ├── Gear/        
│   ├── HIT/            
│   ├── Motor/              
│   └── UEA/                  
├── Python/
│   └── Code/                   */Codebase 
│       ├── Auxiliary/          */Some affiliate code
│       ├── Net/                */Model
│       └── Package/            */Functions
│   └── ExCode/                 */InternVL
│   └── Results/                */Results
│   └── Main.py                 */Main program    

```

## Datasets
The datasets involved in this paper are all public datasets. Due to the size of the datasets, they have not been uploaded. The download paths are in the follows.

## Acknowledgment
- TimesNet: https://github.com/thuml/Time-Series-Library/blob/main/models/TimesNet.py
- InternVL: https://github.com/OpenGVLab/InternVL/tree/main
- OnlineLabelSmoothing: https://github.com/zhangchbin/OnlineLabelSmoothing
- UEA dataset: https://www.timeseriesclassification.com/dataset.php
- HIT dataset: https://drive.google.com/drive/folders/1Km1Go4ilB_bI033SBJ7eJ0uCzbqEqbgt?usp=sharing
- Motor dataset: https://gitlab.com/power-systems-technion/motor-faults/-/tree/main
- WT dataset: https://github.com/Liudd-BJUT/WT-planetary-gearbox-dataset

## Citation
If you find this repository useful, please cite:
```
@ARTICLE{11023071,
  author={Jia, Xinming and Qin, Na and Huang, Deqing and Du, Jiahao and Wang, Tianwei},
  journal={IEEE Sensors Journal}, 
  title={Industrial Large-Scale Diagnostic Model with Lightweight Customized Deployment for Distributed Multiple Non-IID Diagnostic Tasks}, 
  year={2025},
  volume={},
  number={},
  pages={1-13},
  doi={10.1109/JSEN.2025.3574226}}
```
