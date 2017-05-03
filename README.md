# Venue-Category-Estimation-from-Micro-videos
We released the dataset for venue category estimation from micro-videos.

# Announcement


# Introduction
The late 2012 has seen a dramatic shift in the way Internet users digest videos: micro-videos spread rapidly across various online
flagship platforms, such as Instagram,Snapchat, and Vine. In this repository, we relased a rich set of feature extracted from 
micro-videos which crawled from Vine. Our dataset is consisting of 270,145 micro-videos distributed in 188 Foursquare
venue categories.

# Contents
  1 The labels (from 1-188) and their corresponding venue cateogries classes;
  2 Feature_set1: alexnet_visual_feature (4096 dim) + stacked_denosing_autoencode_feature(200 dim) + paragraph_textual_feature(100 dim)
  3 Feature_set2: inception_v3_visual_feature (2048 dim)
  4 Feature_set3: vgg16_visual_feature (1000 dim)
  5 Feature_set4: resnet50_visual_feature(2048 dim)
  
# Citation
Please cite it as...
```
@inproceedings{Zhang2016Shorter,
  title={Shorter-is-Better: Venue Category Estimation from Micro-Video},
  author={Zhang, Jianglong and Nie, Liqiang and Wang, Xiang and He, Xiangnan and Huang, Xianglin and Chua, Tat Seng},
  booktitle={ACM on Multimedia Conference},
  pages={1415-1424},
  year={2016},
}
```
