---
title: "faster RCNN 原码解析"
date: 2019-06-06 20:12:18 +0800
description: 
author: wilson
image: 
  path: /images/ai/computer-vision.jpeg
  thumbnail: /images/ai/computer-vision.jpeg
categories: 
    - ai
tags:
    - code 
	- note
---

# faster RCNN 原码解析

## 1 训练 faster RCNN 

Faster rcnn 的 `train_val.py` 程序的主干如下，它主要是负责对 `fasterRCNN` 网络进行训练：

```python
# 1. 解析参数
args = parse_args()
# 2. 通过voc整合多个训练集数据
imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
# 3. 数据加载器
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,                            										sampler=sampler_batch, num_workers=args.num_workers)
# 4. 构建faster rcnn网络
fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
# 5. 训练rcnn网络
for epoch in range(args.start_epoch, args.max_epochs + 1):
    # 5.1 将faster rcnn网络的head部分冻结
    fasterRCNN.train()
    # 5.2 衰减学习率
    if epoch % (args.lr_decay_step + 1) == 0:
        adjust_learning_rate(optimizer, args.lr_decay_gamma)
        lr *= args.lr_decay_gamma
    # 5.3 批训练网络    
   	for data in dataloaer:
        # 5.3.1 批训练数据
        im_data.resize_(data[0].size()).copy_(data[0])
        im_info.resize_(data[1].size()).copy_(data[1])
        gt_boxes.resize_(data[2].size()).copy_(data[2])
        num_boxes.resize_(data[3].size()).copy_(data[3])
        
        fasterRCNN.zero_grad()
        # 5.3.2 用faster rcnn预估对象类别和边界框
        rois, cls_prob, bbox_pred, \
      	rpn_loss_cls, rpn_loss_box, \
      	RCNN_loss_cls, RCNN_loss_bbox, \
      	rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)
        # 5.3.3 计算损失
        loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
           	+ RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
        # 5.3.4 反向传播，调整faster rcnn网络     
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

误差函数的计算

```python
# _fasterRCNN.forward()得到
# rcnn 的分类误差
RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)
# rcnn 的回归误差
RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, 
                                 rois_inside_ws, rois_outside_ws)

# 在 _fasterRCNN.forward()中，有：
rois, rpn_loss_cls, rpn_loss_bbox 
	= self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)
# self.RCNN_rpn = _RPN(self.dout_base_model)

# _RPN.forward()得到
# rpn 的分类误差
self.rpn_loss_cls = F.cross_entropy(rpn_cls_score, rpn_label)
# 计算rpn的边界误差，请注意在这里用到了inside和outside_weights
self.rpn_loss_box = _smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, 
                                    rpn_bbox_inside_weights,
                                    rpn_bbox_outside_weights, sigma=3, dim=[1,2,3])
```



## 2 通过voc整合多个训练集数据

首先 `combined_roidb` 函数的架构如下：

```python
def combined_roidb(imdb_names, training=True):
    roidbs = [get_roidb(s) for s in imdb_names.split('+')]
  	roidb = roidbs[0] # len(roidb) = 1002
  	
    if len(roidb) > 1:
        # 如果roidbs多于1个，则进行roidb融合, 本例len(roidbs) = 1
        pass
    else:
        # 通过名字得到imdb
		imdb = get_imdb(imdb_names)
        
    if training:
        # 过滤掉没有前景的roidb
        roidb = filter_roidb(roidb)
        
    # 根据宽高比对roidb进行从低到高的排序
    ratio_list, ratio_index = rank_roidb_ratio(roidb)
    
    return imdb, roidb, ratio_list, ratio_index
```

### 2.1 get_roidb(s) 的解析

`imdb_names` 来自入参 `args.imdb_name = voc_2007_trainval`

```python
def get_roidb(imdb_name):
    # 从 voc 中获取图像数据库信息
    imdb = get_imdb(imdb_name)

    # 设置proposal方法，这里是gt（config.py）
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)

    # 得到用于训练的roidb
    # 从imdb中抽取roidb，并给roidb加入一些属性：
    # 所属图像的id，图像路径，图像的(w, h)，
    # 每幅图中每个roi与真实框的重大重合度值 roidb['max_overlaps']
    # 每幅图中每个roi的最大真实重合框所属的类别 roidb['max_classes']
    roidb = get_training_roidb(imdb)
    return roidb
```

这里的 `get_imdb(imdb_names)` 是返回一个voc实例：

```python
def get_imdb(name):
    return __sets[name]()

>>> imdb: <datasets.pascal_voc.pascal_voc object at 0x7f60e737d860>
```

```python
  def get_training_roidb(imdb):
    if cfg.TRAIN.USE_FLIPPED:
      imdb.append_flipped_images() # 在imdb类中，实现水平翻转
    prepare_roidb(imdb)
    return imdb.roidb
```

```python
def prepare_roidb(imdb):
    '''
    计算出 roidb 的一些常用属性
    '''
    roidb = imdb.roidb
    sizes = [Image.open(imdb.image_path_at(i)).size for i in range(imdb.num_images)]
    
    for i in range(len(imdb.image_index)):
        roidb[i]['img_id'] = imdb.image_id_at(i)
        roidb[i]['image'] = imdb.image_path_at(i)
        if not (imdb.name.startswith('coco')):
            roidb[i]['width'] = sizes[i][0]
            roidb[i]['height'] = sizes[i][1]
        # roidb[i]：第i幅图像的所有roi，roidb[i]['gt']roi与第i类别gt的重合度
        # gt_overlaps 维度是 [roi_box_num, class_num]
        gt_overlaps = roidb[i]['gt_overlaps'].toarray()
        max_overlaps = gt_overlaps.max(axis=1) # 第i幅图每个roi与真实框的最大重合度
        max_classes = gt_overlaps.argmax(axis=1) # 第i幅图每个roi最大重合度所属的类别
    	roidb[i]['max_classes'] = max_classes # 每个roi最大真实框所属的类别
    	roidb[i]['max_overlaps'] = max_overlaps # 第i图每个roi与真实框的最大重合度
```

## 3 冻结部分resnet网络

因为只需要训练部分resnet网络，所以要对网络的head部分进行冻结

```python
class resnet(_fasterRCNN):
	def train(self, mode=True):
    	nn.Module.train(self, mode)
        if mode:
            self.RCNN_base.eval()
            self.RCNN_base[5].train()
            self.RCNN_base[6].train()
            self.RCNN_base.apply(set_bn_eval)
            self.RCNN_top.apply(set_bn_eval)
```

```python
def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval() 
```

## 4 faster rcnn 的训练

接下来是最复杂的faster rcnn 部分，用faster rcnn预估对象类别和边界框

```python
rois, cls_prob, bbox_pred, \
rpn_loss_cls, rpn_loss_box, \
RCNN_loss_cls, RCNN_loss_bbox, \
rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)
```

```python
class resnet(_fasterRCNN):
	def __init__(self, classes, num_layers=101, pretrained=False, class_agnostic=False):
        self.model_path = 'data/pretrained_model/resnet101_caffe.pth'
        self.dout_base_model = 1024
        self.pretrained = pretrained
        self.class_agnostic = class_agnostic

        _fasterRCNN.__init__(self, classes, class_agnostic)
        
    def _head_to_tail(self, pool5):
        fc7 = self.RCNN_top(pool5).mean(3).mean(2)
        return fc7
```

下面重要的是 `_fasterRCNN` 类，其初始化如下，主要是定义了 `rpn` 网络和 `roi pool` 层

```python
class _fasterRCNN(nn.Module):
    def __init__(self, classes, class_agnostic):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0
        # 定义 rpn 网络
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
		# 定义 roi pool 层
        self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0)
```

最重要的是它的 `forward` 函数，大致架构如下（省去了部分细节）：

```python
# class _fasterRCNN(nn.Module):
def forward(self, im_data, im_info, gt_boxes, num_boxes): 
    # 将图像数据馈送到基础模型以获得基础特征图，RCNN_base是在resnet类的_init_modules中定义的
    # self.RCNN_base = nn.Sequential(resnet.conv1, resnet.bn1,resnet.relu,
    #  				resnet.maxpool,resnet.layer1,resnet.layer2,resnet.layer3)
    base_feat = self.RCNN_base(im_data)

    # 基础特征图送到 RPN 得到 ROIS
    rois, rpn_loss_cls, rpn_loss_bbox 
    = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

    # 如果是在训练 用ground truth回归
    if self.training:
        roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
        rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

        rois_label = Variable(rois_label.view(-1).long())
        rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
        rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
        rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))

    rois = Variable(rois)
    # 利用ROI pooling从基础特征图中提取候选特征图 proposal feature
    pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))

    # pooling后的候选特征图到top模块
    # self.RCNN_top = nn.Sequential(resnet.layer4) 与layer1相同，也是卷积操作
    # fc7 = self.RCNN_top(pool5).mean(3).mean(2)
    pooled_feat = self._head_to_tail(pooled_feat)

    # 计算bounding box的偏移 self.RCNN_bbox_pred = nn.Linear(2048, 4)
    bbox_pred = self.RCNN_bbox_pred(pooled_feat)
    
    if self.training and not self.class_agnostic:
        # 根据roi标签选择相应的列
        bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
        bbox_pred_select = torch.gather(bbox_pred_view, 1, \
                                        rois_label.view(rois_label.size(0), 1, 1)\
                                        .expand(rois_label.size(0), 1, 4))
        bbox_pred = bbox_pred_select.squeeze(1)
    bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

    # 计算对象分类概率 self.RCNN_cls_score = nn.Linear(2048, self.n_classes)
    cls_score = self.RCNN_cls_score(pooled_feat)
    cls_prob = F.softmax(cls_score, 1)
    cls_prob = cls_prob.view(batch_size, rois.size(1), -1)

    if self.training:
        # 分类损失
        RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)
        # 回归损失
        RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, 
                                         rois_inside_ws, rois_outside_ws)

    return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label

```

由上可以知道，`fast rcnn` 的训练分成如下几步：

1. 由 `resnet` 网络得到基础特征图

   ```python
   base_feat = self.RCNN_base(im_data)
   ```
   
2. 基础特征图送到 RPN 得到 ROIS

   ```python
   rois, rpn_loss_cls, rpn_loss_bbox 
   = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)
   ```
   
3. 对 rois 进行精简处理

   ```python
   roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
   rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data
   ```
   
4. 利用 ROI Pooling 方法，从基础特征图中选取 ROI 部分的池化图

   ```python
   pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))
   ```
   
5. 将池化后的候选池化图送入top网络，获得候选区域特征图

   ```python
   # self.RCNN_top = nn.Sequential(resnet.layer4)
   # fc7 = self.RCNN_top(pool5).mean(3).mean(2)
   pooled_feat = self._head_to_tail(pooled_feat)
   ```
   
6. 利用候选区域特征图预估候选框的偏移

   ```python
   bbox_pred = self.RCNN_bbox_pred(pooled_feat)
   # self.RCNN_bbox_pred = nn.Linear(2048, 4)
   ```
   
7. 利用候选区域特征图预估物体类别

   ```python
   cls_score = self.RCNN_cls_score(pooled_feat)
   cls_prob = F.softmax(cls_score, 1)
   # self.RCNN_cls_score = nn.Linear(2048, self.n_classes)
   ```
   
8. 计算分类误差和回归误差

   ```python
   RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)
   RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, 
                                    rois_inside_ws, rois_outside_ws)
   ```

我们继续追进去，看看具体细节的实现

#### 基础特征图送到 RPN 得到 ROIS 是如何实现的？

```python
rois, rpn_loss_cls, rpn_loss_bbox 
= self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)
# self.RCNN_rpn = _RPN(self.dout_base_model)
```

##### RPN的初始化，大体架构：

```python
class _RPN(nn.Module):
    def __init__(self, din):
        #得到输入特征图的深度
        self.din = din 
        #anchor的等级 __C.ANCHOR_SCALES = [8,16,32]
        self.anchor_scales = cfg.ANCHOR_SCALES 
        #anchor的宽高比 __C.ANCHOR_RATIOS = [0.5,1,2]
        self.anchor_ratios = cfg.ANCHOR_RATIOS 
        #特征步长 __C.FEAT_STRIDE = [16, ]
        self.feat_stride = cfg.FEAT_STRIDE[0]

        # 定义处理输入特征图的convrelu层
        self.RPN_Conv = nn.Conv2d(self.din, 512, 3, 1, 1, bias=True)
        # 通过 RPN_Conv生成的512特征，分别用于分类和回归

        # 定义背景和前景分类得分层 nc: nclass  2(bg/fg) * 9 (anchors)
        # 前景／背景的类别得分， 网络输入是512 输出是参数个数nc
        self.nc_score_out = len(self.anchor_scales) * len(self.anchor_ratios) * 2
        self.RPN_cls_score = nn.Conv2d(512, self.nc_score_out, 1, 1, 0)

        # 定义anchor的偏移预测层，回归到边界框  4(coords) * 9 (anchors)
        self.nc_bbox_out = len(self.anchor_scales) * len(self.anchor_ratios) * 4
        self.RPN_bbox_pred = nn.Conv2d(512, self.nc_bbox_out, 1, 1, 0)

        # 定义anchor目标层 _AnchorTargetLayer, 产生分类的真值rpn_labels
        self.RPN_anchor_target = _AnchorTargetLayer(self.feat_stride, 
                                                    self.anchor_scales, self.anchor_ratios)        
        
        # 定义候选区域层 _ProposalLayer, 参数：特征步长、等级、比例
        self.RPN_proposal = _ProposalLayer(self.feat_stride, 
                                           self.anchor_scales, self.anchor_ratios)

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0
```

这里出现了几个关键层：`_ProposalLayer` 候选区域层和 `_AnchorTargetLayer` 锚框目标层，让我们来看看它们是什么

1. 锚框目标层 _AnchorTargetLayer

```python
class _AnchorTargetLayer(nn.Module):
    """
    利用gtbox，将固定锚框生成锚框目标，作为候选区域网络的拟合目标：(类别，框坐标)
    1. 先要在原始图上均匀的划分出AxHxW个 anchors（A=9，H是feature map的高度，W是宽度）
    2. 删除不在图像中的 anchors
    3. 将与gtbox有最大IoU的anchor标记为前景
    4. IoU > 0.7 的anchor标记为前景，label = 1
    5. IoU < 0.3的标记为背景: label=0
    6. 计算从anchors变到gtbox要修正的变化量，记为 bbox_targets
    7. 只有前景类内部权重才非0，参与回归
    8. 外部权重初始化 bbox_outside_weights[labels == 1] = positive_weights
    9. 将label立体化为特征图的长宽，这样每个label对应特征图的一个像素, 也就是给每一个anchor都打上前景或背景的label。
    有了labels，你就可以对RPN进行训练使它对任意输入都具备识别前景、背景的能力

    返回值：
        outputs = [labels, bbox_targets, bbox_inside_weights, bbox_outside_weights]
        labels：特征图上每个点的标签，背景或前景
        bbox_targets：特征图对应的anchor要变成真实的gt boxes的改变量(dx, dy, dw, dh)
        bbox_weights：权值
    """
    def __init__(self, feat_stride, scales, ratios):
        # 省略部分代码
        self._anchors = torch.from_numpy(
            # 以(0, 0, 15, 15)为基础生成9个框(0.5, 1, 2)宽高比 * (8, 16, 32)等级
            generate_anchors(scales=np.array(anchor_scales), 	
                             ratios=np.array(ratios))
        ).float()
    
    def forward(self, input):
      """
      省略
      """
```

2. 候选区域层 ProposalLayer

是对 `input` 中的所有17100个锚框目标进行精简，根据这17100个框的类别概率大小进行删减，再通过MNS删减，从而得到最好的锚框

```python
class _ProposalLayer(nn.Module):
    def __init__(self, feat_stride, scales, ratios): # 参数： 特征步长 等级 比例
        super(_ProposalLayer, self).__init__()
        # 得到特征步长
        self._feat_stride = feat_stride
        # 以(0,0, 15,15)窗口为基础，生成9个锚框，它们也是以(左上角，右下角)的格式排列成行
        self._anchors = torch.from_numpy(generate_anchors(scales=np.array(scales),
            ratios=np.array(ratios))).float()
        self._num_anchors = self._anchors.size(0)
        
    def forward(self, input):
        '''
        1. 通过 feature_stride * H * W 复制self._anchors生成anchors
        2. anchors变动bbox_deltas，变成候选框(proposals)
        3. 移除四个角不在图像边界内的候选框
        4. 对分数(概率)从大到小进行排序
        5. 在NMS之前接受最高的 pre_nms_topN 个提议区
        6. 将门槛为0.7的NMS应用于剩余提议
        7. 在NMS之后接受 after_nms_topN 候选框
        8. return 顶级候选框
        
        input = (rpn_cls_prob.data, rpn_bbox_pred.data, im_info, cfg_key)
        output = (batch_no, proposals)  第1列为batch号，后面为候选框坐标
        '''
        # 前景概率
        scores = input[0][:, self._num_anchors:, :, :]
        # 偏移 17100 偏移
        bbox_deltas = input[1]
        # 图像信息 (h, w, scale)
        im_info = input[2]
        # 是training还是test
        cfg_key = input[3]
```

##### RPN 的前向传播

```python
def forward(self, base_feat, im_info, gt_boxes, num_boxes):
    #features信息包括 (batch_size，data_height，data_width，num_channels)
    #即批尺寸，特征数据高度，特征数据宽度，特征的通道数。
    batch_size = base_feat.size(0)

	# 1. 对resnet抓取的基础特征进行分析，得到分类概率
    rpn_conv1 = F.relu(self.RPN_Conv(base_feat), inplace=True) 
    rpn_cls_score = self.RPN_cls_score(rpn_conv1) # 得到RPN分类得分
    rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2) # 2 - 前景/背景
    # 用softmax函数得到前景和背景概率
    rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, 1)
    # 前景背景分类，2个参数 nc_score_out = 2(bg/fg) * 9 (anchors)
    rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out)

    # 2. 利用基础特征预估边界框偏移
    rpn_bbox_pred = self.RPN_bbox_pred(rpn_conv1)

    cfg_key = 'TRAIN' if self.training else 'TEST'
    # 提取候选区域  self.RPN_proposal = _ProposalLayer(input) 
    # 对input中的所有17100个锚框目标按概率大小和NMS方案进行精简
    # 其实 RPN_proposal 应该叫 RPN_proposal_reduce
    rois = self.RPN_proposal((rpn_cls_prob.data, rpn_bbox_pred.data,
                              im_info, cfg_key))

    self.rpn_loss_cls = 0
    self.rpn_loss_box = 0

    # 生成训练标签并构建rpn损失
    if self.training:
        # self.RPN_anchor_target = _AnchorTargetLayer()
        # 利用gtbox，将固定锚框变成锚框目标(类别，框坐标)，作为候选区域网络的拟合对象
        # 其实 anchor_target 应该叫 proposal_target
        rpn_data = self.RPN_anchor_target((rpn_cls_score.data, gt_boxes, im_info, num_boxes))

        # 计算分类损失
        # 返回rpn网络判断的anchor前后景分数
        rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1)\
        .contiguous().view(batch_size, -1, 2)
        # 返回每个anchor属于前景还是后景的ground truth
        rpn_label = rpn_data[0].view(batch_size, -1)
        
        # rpn_keep = rpn_label中不为-1的索引，-1是要忽略的标签
        rpn_keep = Variable(rpn_label.view(-1).ne(-1).nonzero().view(-1))
        # 返回得分中前景和背景的得分和label
        rpn_cls_score = torch.index_select(rpn_cls_score.view(-1,2), 0, rpn_keep)
        rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep.data)
        rpn_label = Variable(rpn_label.long())
        # 计算rpn的分类误差
        self.rpn_loss_cls = F.cross_entropy(rpn_cls_score, rpn_label)
        # 统计前景数目
        fg_cnt = torch.sum(rpn_label.data.ne(0))
        
        # 计算回归损失        
        rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]
        # 在训练计算边框误差时有用，仅对未超出图像边界的anchor有用
        rpn_bbox_inside_weights = Variable(rpn_bbox_inside_weights)
        rpn_bbox_outside_weights = Variable(rpn_bbox_outside_weights)
        rpn_bbox_targets = Variable(rpn_bbox_targets)

        # 计算rpn的边界误差，请注意在这里用到了inside和outside_weights
        self.rpn_loss_box = _smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, 
                                            rpn_bbox_inside_weights,
                                            rpn_bbox_outside_weights, sigma=3, dim=[1,2,3])

        return rois, self.rpn_loss_cls, self.rpn_loss_box
```

这样就得到了faster训练所需的rpn误差值



## 附录

### _smooth_l1_loss

```python
def _smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, 
                    bbox_outside_weights, sigma=1.0, dim=[1]):

    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = torch.abs(in_box_diff)
    smoothL1_sign = (abs_in_box_diff < 1. / sigma_2).detach().float()
    in_loss_box = torch.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = out_loss_box
    for i in sorted(dim, reverse=True):
      loss_box = loss_box.sum(i)
    loss_box = loss_box.mean()
    return loss_box
```



### resnet.layer4

```python
(layer4): Sequential(
      (0): Bottleneck(
        (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, 
                           affine=True, track_running_stats=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), 
                        stride=(2, 2), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, 
                           affine=True, track_running_stats=True)
        (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, 
                           affine=True, track_running_stats=True)
        (relu): ReLU(inplace)
        (downsample): Sequential(
          (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, 
                           affine=True, track_running_stats=True)
        )
      )
      (1): Bottleneck(
        (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, 
                           affine=True, track_running_stats=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), 
                        stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, 
                           affine=True, track_running_stats=True)
        (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, 
                           affine=True, track_running_stats=True)
        (relu): ReLU(inplace)
      )
      (2): Bottleneck(
        (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, 
                           affine=True, track_running_stats=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), 
                        padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, 
                           track_running_stats=True)
        (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, 
                           track_running_stats=True)
        (relu): ReLU(inplace)
      )
    )
```

