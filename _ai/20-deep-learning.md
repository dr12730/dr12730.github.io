---
title: "Deep Learning 深度学习"
image: 
  path: /images/ai/deep-learning.jpeg
  thumbnail: /images/ai/deep-learning.jpeg
---

# faster RCNN 原码解析

## 1 训练 faster RCNN 

Faster rcnn 的 `train_val.py` 程序的主干如下，它主要是负责对 `fasterRCNN` 网络进行训练：

