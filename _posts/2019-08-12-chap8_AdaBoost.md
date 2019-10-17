---
title: 第8章 提升方法
date: 2019-08-12 19:46:46 +0800
description:
image:
  path: /assets/images/posts/2019-08-12-schap8/cover.jpg
  thumbnail: /assets/images/posts/2019-08-12-schap8/thumb.jpg
categories:
  - it
tags:
  - 统计学习方法
---

<!-- vim-markdown-toc GFM -->

* [1 前言](#1-前言)
    * [1.1 本意概要](#11-本意概要)
    * [1.2 目录](#12-目录)
* [2 读书笔记](#2-读书笔记)
    * [2.1 提升方法 AdaBoost](#21-提升方法-adaboost)
        * [2.1.1 基本思路](#211-基本思路)
        * [2.1.2 AdaBoost 算法](#212-adaboost-算法)
        * [2.1.3 应用](#213-应用)
        * [2.1.3 代码实战](#213-代码实战)
    * [2.2 AdaBoost 算法训练误差分析](#22-adaboost-算法训练误差分析)
        * [2.2.1 AdaBoost 的训练误差边界](#221-adaboost-的训练误差边界)
    * [2.3 AdaBoost 算法的解释](#23-adaboost-算法的解释)
        * [2.3.1 前向分步算法](#231-前向分步算法)
            * [算法](#算法)
        * [2.3.2 前向分步算法与 AdaBoost](#232-前向分步算法与-adaboost)
    * [3 编程](#3-编程)

<!-- vim-markdown-toc -->

# 1 前言

## 1.1 本意概要

1. **提升方法**是将弱学习算法提升为强学习算法的统计学习方法。提升方法通过反复修改训练数据权值分布，构建一系列基本分类器（弱分类器），然后将这些弱分类器线性组合成一个强分类器。

2. AdaBoost 模型是弱分类器的线性组合：

   $$ f(x) = \sum\limits_{m=1}^M \alpha_m G_m(x)$$

3. AdaBoost 每次迭代提高前一轮分类器错误分类数据的权值，降低正确分类的数据权值。最后，将基本分类器线性组合，对于分类误差率小的基本分类器给予大的权值，给分类误差率大的基本分类器以小的权值

4. AdaBoost 每次迭代可以减少它在训练数据集上的分类误差率

5. AdaBoost 是前身分步算法的一个实现。模型是加法模型，损失函数是指数损失，算法是前身分步算法。在每步中极小化损失函数 $(\beta_m, \gamma_m) = arg \min\limits_{\beta, \gamma} \sum\limits_{i=1}^N L(y_i, f_{m-1}(x_i) + \beta b(x_i; \gamma))$ 得到参数 $\beta_m, \gamma_m$

6. 提升树是统计学习中最有效的方法之一。以分类树和回归树为基本分类器

## 1.2 目录

1. 提升方法  
   1.1 提升方法的基本思路  
   1.2 AdaBoost 算法  
   1.3 AdaBoost 的例子
2. AdaBoost 算法的训练误差分析
3. AdaBoost 算法的解释  
   3.1 前向分步算法  
   3.2 前向分步算法与 AdaBoost
4. 提升树  
   4.1 提升树模型  
   4.2 提升树算法  
   4.3 梯度提升

---

# 2 读书笔记

提升方法是一种常见的统计学习方法，它的过程是通过改变训练样本的权重，训练多个分类器，最后把这些分类器线性组合起来，得到性能更高的分类器。

## 2.1 提升方法 AdaBoost

### 2.1.1 基本思路

- 对于复杂任务，三个臭皮匠顶个诸葛亮。

- **强可学习**是指对于一个类别（或概念）如果存在一个多项式的学习算法能够拟合它，并且正确率高

- **弱可学习**是指对于一个类别，算法学习的正确率仅比随机猜测略好

- **提升方法**是改变训练数据的概率分布（训练数据的权值分布），调用弱学习算法学习一系列弱分类器

提升方法的理论基础来源于**强可学习等价于弱可学习**，而发现弱可学习算法是非常容易的。那么，如何把弱可学习算法提升到强可学习算法呢？提升方法采用的是从弱可学习算法出发，反复学习，得到一系列的弱分类器（基础分类器），然后组合它们，构成一个强分类器。

### 2.1.2 AdaBoost 算法

- 输入：

  - 数据集 $T = \{(x_1, y_1), (x_2, y_2), ..., (x_N, y_N)\}$
  - 弱学习算法

- 输出

  - 最终分类器 $G(x)$

- 流程

  1. 初始化训练数据集的权值分布

     $D_1 = (w_{11}, ..., w_{1i}, ..., w_{1N})$，其中 $w_{1i} = \dfrac{1}{N}, i = 1, 2, ..., N$

  2. 对于 $m = 1, 2, ...., M$ 的每一次迭代，AdaBoost 学习一个基本分类器 $G_m(x)$

     - 用权值分布 $D_m$ 的训练数据集学习，得到基本分类器 $G_m(x): X \to \{-1, +1\}$

     - 计算 $G_m(x)$ 在训练数据集上的 **加权分类误差率**

       $$e_m = \sum\limits_{i=1}^{N} w_{mi} I(G_m(x_i) \neq y_i)$$

       其中：$w_{m,i}$ 是第 i 个样本的权值

     - 计算 $G_m(x)$ 的系数 $\alpha_m = \dfrac{1}{2} \ln \dfrac{1 - e_m}{e_m}$

     - 更新训练数据集的权值分布

       $$\begin{aligned}D_{m+1} &= (w_{m+1, 1}, ..., w_{m+1, i, }..., w_{m+1, N}) \\ w_{m+1, i} &= \dfrac{w_{mi}}{Z_m} e^{-\alpha_m y_i G_m(x_i)} \\ Z_m &= \sum\limits_{i=1}^{N} w_{mi} e^{-\alpha_m y_i G_m(x_i)} \end{aligned}$$

     - 构建基本分类器的线性组合 $f(x) = \sum\limits_{m=1}^N \alpha_m G_m(x)$，得到最终分类器：

       $$G(x) = sign \left( f(x) \right) = sign \left( \sum\limits_{m=1}^N \alpha_m G_m(x)\right)$$


### 2.1.3 应用

给定训练集数据如下，用 AdaBoost 算法学习一个强分类器：

| 序号 | 1 | 2 | 3  | 4  | 5  | 6 | 7 | 8 | 9 | 10 |
|:---: |---|---|----|----|----|---|---|---|---|----|
| x    | 0 | 1 | 2  | 3  | 4  | 5 | 6 | 7 | 8 | 9  |
| y    | 1 | 1 | -1 | -1 | -1 | 1 | 1 | 1 | 1 | -1 |

1. 给每一个训练样本一个权值

    $$D_1 = (w_{1,1}, \dots, w_{1, i}, \dots, w_{1, M})$$ 

    其中 $w_{1, i} = \dfrac{1}{N} = \dfrac{1}{10}$ 

2. 对于第一次迭代，产出第一个基础分类器 $G_1(x)$

    - 首先，定义弱分类器 $G(x)$ 为如下形式：

        $G(x) = \left\{\begin{aligned} 1, && x \le v\\ -1, && x > v\end{aligned}\right.$ 或 $G(x) = \left\{\begin{aligned} 1, && x > v\\ -1, && x \le v\end{aligned}\right.$

    - 然后，$v = [0.5, 1.5, 2.5, \dots, 10.5]$，看哪个 $G(x)$ 和 v 的组合得到的误差率最小。这里遍历得到

        $$G(x) = \left\{\begin{aligned} 1, && x \le 2.5\\ -1, && x > 2.5\end{aligned}\right.$$

        分类误差率最小，而且 $e_1 = 0.3$

    - 计算这个基本分类器  $G_1(x)$ 的系数

        $$\alpha_1 = \dfrac{1}{2} \log \dfrac{1 - e_1}{e_1} = 0.4236$$

    - 更新训练数据集的权值分布

        $$D_1 = (w_{2,1}, \dots, w_{2,i}, \dots, w_{2, M})$$

        其中 $w_{2,i} = \dfrac{w_{1,i}}{Z_1} e^{-\alpha_i y_i G_1(x)}$

        计算得到 $D_1 = (0.07143, 0.07143, \dots)$

    - 那么，$f_1(x) = 0.4236 G_1(x)$，分类器为 $sign[f_1(x)]$
    
3. 对于第二次迭代，产出第二个基础分类器

    - 在权值分布为 $D_2$ 的训练集上，阈值为 $v = 8.5$ 是误差率最优，所以基本分类器为

        $$G_2(x) = \left\{\begin{aligned} 1, && x \le 8.5\\ -1, && x > 8.5\end{aligned}\right.$$

    - $G_2(x)$ 在训练集上的加权误差率为 $e_2 = 0.2143$

    - 计算这个基本分类器  $G_2(x)$ 的系数

        $$\alpha_2 = \dfrac{1}{2} \log \dfrac{1 - e_2}{e_2} = 0.6496$$

    - 更新训练数据集的权值分布 $D_3$

        $$D_3 = (w_{3,1}, \dots, w_{3,i}, \dots, w_{3, M})$$

    - 那么，$f_2(x) = 0.4236 G_1(x) + 0.6496 G_2(x)$，分类器为 $sign[f_2(x)]$

4. $m = 3, 4, \dots$ 以此类推

    AdaBoost 的训练误差是以指数级下降的

### 2.1.3 代码实战

```python
import os
import numpy as np
import matplotlib.pyplot as plt

DATA_FILE = './Machine-Learning/AdaBoost_Project2/horseColicTraining2.txt'
TEST_FILE = './Machine-Learning/AdaBoost_Project2/horseColicTest2.txt'

def load_data(filename):
    '''
    从文件中读取数据和标签 
    '''
    dirpath = os.path.relpath('.')
    filepath = os.path.join(dirpath, filename)
    data = []
    labels = []

    with open(filepath) as fr:
        lines = [line.strip().split('\t') for line in fr.readlines()]
        #nftrs = len(lines[0]) - 1
        for line in lines:
            d = line[:-1]
            l = line[-1]
            data.append(d)
            labels.append(l)
    return np.array(data, dtype=np.float), np.array(labels, dtype=np.float, ndmin=2)


def stumpClassify(data, dimen, thresh_val, thresh_ineq):
    '''
    :data: 训练数据
    :dimen: 特征
    :thresh_val: 阈值
    :thresh_ineq: 符号

    :ret: 分类结果
    '''
    ret = np.ones((data.shape[0], 1))
    if thresh_ineq == 'lt':
        ret[data[:, dimen] <= thresh_val] = -1
    else:
        ret[data[:, dimen] > thresh_val] = -1
    return ret


def buildStump(data, labels, D):
    '''
    找到数据集上的最佳单层决策树

    Parameters:
        data: 训练集数据
        labels: 标签
        D: 样本权重

    Returns:
        bestStump: 最佳单层决策树信息
        minError: 最小误差
        bestClasEst: 最佳的分类结果

    ------------------------------------------------
    1. 对于训练数据集中的每一个特征：
        1.1 对于大于和小于（等于）两种情况:
            1.1.1 计算分类的阈值
            1.1.2 计算分类结果
            1.1.3 计算加权误差率
            1.1.4 计算最小误差率及对应的基本分类器
    ------------------------------------------------
    '''
    # m=5, n=2
    m, n = np.shape(data)
    numSteps = 10.0
    bestStump = {}
    # (5, 1)全零列矩阵
    bestClasEst = np.zeros((m, 1))
    # 最小误差初始化为正无穷大inf
    minError = float('inf')
    # 遍历所有特征
    for i in range(n):
        # 找到(每列)特征中的最小值和最大值
        rangeMin = data[:, i].min()
        rangeMax = data[:, i].max()
        # 计算步长
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps) + 1):
            # 大于和小于的情况均遍历，lt:Less than  gt:greater than
            for inequal in ['lt', 'gt']:
                # 计算阈值
                threshVal = (rangeMin + float(j) * stepSize)
                # 计算分类结果
                predictedVals = stumpClassify(data, i, threshVal, inequal)
                # 初始化误差矩阵
                errArr = np.ones((m, 1))
                # 分类正确的，赋值为0
                labels = labels.reshape(-1, 1)
                errArr[predictedVals == labels] = 0
                # 计算误差
                weightedError = D.T.dot(errArr)
                print("基本分类器 特征: %d, 阈值: %.2f, 符号: %s, 加权误差: %.3f" % (i, threshVal, inequal, weightedError))
                # 找到误差最小的分类方式
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst


def adaClassify(datToClass, classifierArr):
    '''
    函数说明：AdaBoost分类函数
    
    Parameters:
        datToClass - 待分类样例
        classifierArr - 训练好的分类器
        
    Returns:
        分类结果
    '''
    data = datToClass
    m = np.shape(data)[0]
    aggClassEst = np.zeros((m, 1))
    for i in range(len(classifierArr)):
        # 遍历所有分类器进行分类
        classEst = stumpClassify(data, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        # print(aggClassEst)
    return np.sign(aggClassEst)



def adaBoostTrainDS(dataArr, classLabels, numIt=60):
    '''
    函数说明：使用AdaBoost进行优化
    
    Parameters:
        dataArr - 数据矩阵
        classLabels - 数据标签
        numIt - 最大迭代次数
        
    Returns:
        weakClassArr - 存储单层决策树的list
        aggClassEsc - 训练的label
    '''
    weakClassArr = []
    # 获取数据集的行数
    m = np.shape(dataArr)[0]
    # 样本权重，每个样本权重相等，即1/n
    D = np.ones((m, 1)) / m
    # 初始化为全零列
    aggClassEst = np.zeros((m, 1))
    # 迭代
    for i in range(numIt):
        # 构建单层决策树
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        # print("D:", D.T)
        # 计算弱学习算法权重alpha，使error不等于0，因为分母不能为0
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
        # 存储弱学习算法权重
        bestStump['alpha'] = alpha
        # 存储单层决策树
        weakClassArr.append(bestStump)
        # 打印最佳分类结果
        # print("classEst: ", classEst.T)
        # 计算e的指数项
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)
        # 计算递推公式的分子
        D = np.multiply(D, np.exp(expon))
        # 根据样本权重公式，更新样本权重
        D = D / D.sum()
        # 计算AdaBoost误差，当误差为0的时候，退出循环
        # 以下为错误率累计计算
        aggClassEst += alpha * classEst
        # print("aggClassEst: ", aggClassEst.T)
        # 计算误差
        aggErrors = np.multiply(np.sign(aggClassEst) != (classLabels).T, np.ones((m, 1)))
        errorRate = aggErrors.sum() / m
        # print("total error:", errorRate)
        if errorRate == 0.0:
            # 误差为0退出循环
            break
    return weakClassArr, aggClassEst


if __name__ == '__main__':
    dataArr, LabelArr = load_data(DATA_FILE)
    weakClassArr, aggClassEst = adaBoostTrainDS(dataArr, LabelArr)
    testArr, testLabelArr = load_data(TEST_FILE)
    print(weakClassArr)
    predictions = adaClassify(dataArr, weakClassArr)
    errArr = np.mat(np.ones((len(dataArr), 1)))
    print('训练集的错误率:%.3f%%' % float(errArr[predictions != np.mat(LabelArr).T].sum() / len(dataArr) * 100))
    predictions = adaClassify(testArr, weakClassArr)
    errArr = np.mat(np.ones((len(testArr), 1)))
    print('测试集的错误率:%.3f%%' % float(errArr[predictions != np.mat(testLabelArr).T].sum() / len(testArr) * 100))
```
运行结果：

```shell
基本分类器 特征: 0, 阈值: 0.90, 符号: lt, 加权误差: 0.502
基本分类器 特征: 0, 阈值: 0.90, 符号: gt, 加权误差: 0.498
基本分类器 特征: 0, 阈值: 1.00, 符号: lt, 加权误差: 0.497
基本分类器 特征: 0, 阈值: 1.00, 符号: gt, 加权误差: 0.503
基本分类器 特征: 0, 阈值: 1.10, 符号: lt, 加权误差: 0.497
基本分类器 特征: 0, 阈值: 1.10, 符号: gt, 加权误差: 0.503
基本分类器 特征: 0, 阈值: 1.20, 符号: lt, 加权误差: 0.497
基本分类器 特征: 0, 阈值: 1.20, 符号: gt, 加权误差: 0.503
基本分类器 特征: 0, 阈值: 1.30, 符号: lt, 加权误差: 0.497
基本分类器 特征: 0, 阈值: 1.30, 符号: gt, 加权误差: 0.503
基本分类器 特征: 0, 阈值: 1.40, 符号: lt, 加权误差: 0.497
基本分类器 特征: 0, 阈值: 1.40, 符号: gt, 加权误差: 0.503
基本分类器 特征: 0, 阈值: 1.50, 符号: lt, 加权误差: 0.497
基本分类器 特征: 0, 阈值: 1.50, 符号: gt, 加权误差: 0.503
基本分类器 特征: 0, 阈值: 1.60, 符号: lt, 加权误差: 0.497
基本分类器 特征: 0, 阈值: 1.60, 符号: gt, 加权误差: 0.503
基本分类器 特征: 0, 阈值: 1.70, 符号: lt, 加权误差: 0.497
基本分类器 特征: 0, 阈值: 1.70, 符号: gt, 加权误差: 0.503
基本分类器 特征: 0, 阈值: 1.80, 符号: lt, 加权误差: 0.497
基本分类器 特征: 0, 阈值: 1.80, 符号: gt, 加权误差: 0.503
基本分类器 特征: 0, 阈值: 1.90, 符号: lt, 加权误差: 0.497
基本分类器 特征: 0, 阈值: 1.90, 符号: gt, 加权误差: 0.503
基本分类器 特征: 0, 阈值: 2.00, 符号: lt, 加权误差: 0.498
基本分类器 特征: 0, 阈值: 2.00, 符号: gt, 加权误差: 0.502
基本分类器 特征: 1, 阈值: 0.20, 符号: lt, 加权误差: 0.502
基本分类器 特征: 1, 阈值: 0.20, 符号: gt, 加权误差: 0.498
...

训练集的错误率:18.729%
测试集的错误率:19.403%
```

## 2.2 AdaBoost 算法训练误差分析

### 2.2.1 AdaBoost 的训练误差边界

**定理 1** AdaBoost 算法最终分类器的训练误差界为：

$$\dfrac{1}{N} \sum\limits_{i=1}^N I\bigg( G(x_i) \neq yi \bigg) \le \dfrac{1}{N} \sum\limits_i \exp\bigg(-y_i f(x_i)\bigg) = \prod_m Z_m$$

**定理 2** AdaBoost 二分类问题的训练误差边界

$$\prod_{m=1}^M Z_m = \prod_{m=1}^M 2 \sqrt{e_m(1-e_m)} = \prod_{m=1}^M \sqrt{1-4\gamma_m^2} \le \exp \left( -2 \sum\limits_{m=1}^M \gamma^2 \right)$$

这里，$\gamma_m = \dfrac{1}{2} - e_m$

## 2.3 AdaBoost 算法的解释

从前向分步算法的角度理解 AdaBoost 算法

### 2.3.1 前向分步算法

**思想：**

从前向后，每一步只学习一个基函数及其系数，逐步逼近优化目标，就可以简化优化的复杂度

#### 算法

- 输入

  - 训练数据集 $T = \{(x_1, y_1), (x_2, y_2), ..., (x_N, y_N) \}$
  - 损失函数 $L(y, f(x))$
  - 基函数集 $\{ b(x; \gamma) \}$

- 输出

  - 加法模型 $f(x) = \sum\limits_{m=1}^M \beta_m b(x; \gamma)$

- 流程

  1. 初始化 $f_0(x) = 0$

  2. 对 $m = 1, 2, ..., M$

     - 极小化损失函数得到参数 $\beta_m, \gamma_m$

       $$(\beta_m, \gamma_m) = \mathop{\arg\min}\limits_{\beta, \gamma} L\bigg( y_i, f_{m-1}(x_i) + \beta b(x; \gamma) \bigg)$$

     - 更新

       $$f_m(x) = f_{m-1}(x) + \beta_m b(x; \gamma_m)$$

  3. 得到加法模型

     $$f(x) = \sum\limits_{m=1}^M \beta_m b(x; \gamma)$$

### 2.3.2 前向分步算法与 AdaBoost

- 加法模型： $f(x) = \sum\limits_{m=1}^M \alpha_m G_m(x)$
- 损失函数：$L(y, f(x)) = \exp \bigg(-y f(x) \bigg)$

- 基本分类器：$G_m(x)$

- 流程

  1. 假设经过 $m-1$ 轮迭代前向分步算法得到 $f_{m-1}(x)$

     $$ f_{m-1}(x) = f_{m-2}(x) + \alpha_{m-1} G_{m-1}(x) = \alpha_1 G_1(x) + ··· + \alpha_{m-1} G_{m-1}(x)$$

     则第 $m$ 轮迭代该应得到

     $$f_m(x) = f_{m-1}(x) + \alpha_m G_m(x)$$

     这里的 $\alpha_m, G_m(x)$ 使 $f_m(x)$ 在训练集上的损失最小

  2. 最小化损失函数求 $G^*_m(x)$ 和 $\alpha_m$

     $$ \begin{aligned}\alpha_m, G_m(x) & = \mathop{\arg\min}\limits_{a, G} \sum\limits_{i=1}^N \exp \bigg[ -y_i \big(f_{m-1}(x_i) + \alpha G(x_i)\big)\bigg] \\ & = \arg\min\limits_{\alpha, G}\sum\limits_{i=1}^N \bar{w}_{mi} \exp \bigg[-y_i \alpha G(x_i)\bigg]\end{aligned}$$

     > - $\bar{w}_{mi} = \exp [-y_i f_{m-1}(x_i)]$，不依赖 $\alpha$ 和 $G$，所以与最小化无关。
     > - $y_i G(x) \in {-1, +1}$
     > - M1：分类正确的数据集
     > - M2：分类错误的数据集

     $$\begin{aligned} \alpha_m, G_m &= \mathop{\arg\min}\limits_{\alpha, G}\sum\limits_{i=1}^N \bar{w}_{mi} \exp [-y_i \alpha G(x_i)] \\ &= \sum\limits_{i \in M1} \bar{w}_{mi} e^{-\alpha} + \sum\limits_{i \in M2} \bar{w}_{mi} e^{-\alpha} + \sum\limits_{i \in M2} \bar{w}_{mi} \left(e^{\alpha} - e^{-\alpha} \right) \\ &= \sum\limits_{i=1}^{M} e^{-\alpha}\bar{w}_{mi} +  \left(e^{\alpha} - e^{-\alpha} \right)\sum\limits_{i \in M2} \bar{w}_{mi} I(y_i \neq G(x_i)) \end{aligned}$$

     先求：
     $$G^*_m(x) = \mathop{\arg\min}\limits_{G} \sum\limits_{i} \bar{w}_{mi} I(y_i \neq G(x_i))$$

     再求：
     $$\begin{aligned}\alpha_m &= \mathop{\arg\min}\limits_{\alpha} \sum\limits_{i} \bar{w}_{mi} \exp (\alpha y_i G^*(x_i)) \\ &= \mathop{\arg\min}\limits_{\alpha} \sum\limits_{i \in M1}\bar{w}_{mi} e^{-\alpha} + \sum\limits_{i \in M2} \bar{w}_{mi} e^{\alpha} \\ &= \left( e^{\alpha} - e^{-\alpha} \right) \sum\limits_{i} \bar{w}_{mi} I(y_i \neq G(x_i)) + e^{-\alpha} \sum\limits_{i}\bar{w}_{mi}\end{aligned}$$

     求导：

     $$\begin{aligned}\dfrac{\partial L}{\partial \alpha_m} &= \left( e^\alpha + e^{-\alpha} \right) \sum \bar{w}_i I - \sum e^{-\alpha} \bar{w}_i \\ &= \left( e^{2\alpha} +1 \right) \sum \bar{w}_i I - \sum\bar{w}_i = 0\end{aligned}$$

     所以有：

     $$\begin{aligned} \alpha &= \frac{1}{2} \ln \frac{\sum\limits_{i} \bar{w}_{mi} - \sum\limits_{i} \bar{w}_{mi} I}{\sum\limits_{i}\bar{w}_{mi}I} \\ &= \frac{1}{2} \ln \frac{1 - e_m}{e_m} \end{aligned}$$

  其中：

  $$e_m = \frac{\sum\limits_{i} \bar{w}_{mi} I}{\sum\limits_{i} \bar{w}_{mi}}$$

  $$\begin{aligned}\bar{w}_{mi} &= \exp \bigg( y_i f_{m-1}(x_i)\bigg) \\ &= \exp \bigg(-y_i \sum\limits_{j=1}^{M-1}\alpha_j G_j(x_i)\bigg) \\&= \prod_j \exp\bigg(-y_i\alpha_j G_j(x_i)\bigg)\end{aligned}$$

---

## 3 编程

实现提升算法

```python
import numpy as np

class AdaBoost:

    def __init__(self, features, labels, iter_times=3):
        self.ftrs = features
        self.labels = labels
        self.iter_times = iter_times
        wights = [1 / len(features)] * len(features)
        self.weights = np.array(wights)
        self.values_list = []

    def base_classifier(self, value, ftrs, less):
        classes = 2 * (ftrs < value) - 1
        return classes if less else -classes

    def error_ratio(self, ftrs, value, less):
        classes = self.base_classifier(value, ftrs, less)
        err_ratio = sum(self.weights * (classes != self.labels))
        return err_ratio

    def get_coff(self, ftrs, value, less):
        err_ratio = self.error_ratio(ftrs, value, less)
        coff = np.log((1 - err_ratio) / err_ratio) / 2
        return coff

    def get_best_v(self, ftrs):
        ftrs = ftrs * self.weights
        ftrs = np.sort(ftrs)
        values = [(ftrs[i]+ftrs[i+1])/2
                        for i in range(len(ftrs))
                            if i != len(ftrs)-1]
        values.append(ftrs[-1]+0.1)
        err_ratios = [self.error_ratio(ftrs, v, less)
                        for less in [True, False] for v in values]
        min_index = err_ratios.index(min(err_ratios))
        less = min_index < 10
        min_index = min_index % 10
        return self.ftrs[min_index] + 0.5, less

    def renew_weights(self, alpha, value, ftrs, less):
        G = self.base_classifier(value, ftrs, less)
        expon = -alpha * self.labels * self.base_classifier(value, ftrs, less)
        normal_coff = sum(self.weights * np.exp(expon))
        self.weights = self.weights * np.exp(expon) / normal_coff

    def train(self):
        for i in range(self.iter_times):
            value, less = self.get_best_v(self.ftrs)
            alpha = self.get_coff(self.ftrs, value, less)
            self.renew_weights(alpha, value, self.ftrs, less)
            print("[", i, "]", "\n分类点:", value, "\n系数:",
                    alpha, "\n权重:", self.weights)
            self.values_list.append((value, less))

    def sign(self, features):
        return 1 if features > 0 else -1

    def predict(self, features):
        func_list = [self.base_classifier(value, features, less)
                     for value, less in self.values_list]
        result = sum(func_list)
        return self.sign(result)

if __name__ == "__main__":
    np.set_printoptions(precision=3)
    features = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    labels = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1])
    adaboost = AdaBoost(features, labels)
    adaboost.train()
```
