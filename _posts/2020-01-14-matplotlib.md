---
title: Matplotlib 使用笔记
mode: immersive
date: 2020-01-14 13:35:19 +0800
key: matplotlib-2020-01-14
layout: article
cover: /assets/images/posts/2020-01-14-matplotlib/cover.jpg
header:
  theme: dark
  background: 'linear-gradient(135deg, rgb(34, 139, 87), rgb(139, 34, 139))'
article_header:
  show_cover: true
  type: overlay
  theme: dark
  background_image:
    gradient: 'linear-gradient(rgba(255, 255, 255, 0), rgba(255, 255, 255, 0), rgba(255, 255, 255, 0))'
    src: /assets/images/posts/2020-01-14-matplotlib/post-web.jpg
tags: [绘图, matplotlib]
---

*如何使用 matplotlib 来快速绘制图像*


<!--more-->

## 基本概念

1. `Figure fig = plt.figure()`

    fig 理解为画布，在这个画布上加入各种元素

2. `Axes ax = fig.add_subplot(111)`

    轴域，由 x, y 轴构成的区域。如果画布上只有一张图，那么轴域 axes 就只有一个；如果画布 fig 上有多个子图 subplot，那么就有多个轴域 axes

3. `Axis ax.xaxis/ax.yaxis`

    x、y 坐标轴。每个坐标轴实际上也是由竖线和数字组成的，每一个竖线其实也是一个axis的subplot，因此ax.xaxis也存在axes这个对象。对这个axes进行编辑就会修改xaxis图像上的表现。 


## 图像组成

下面是图像各个组件的名称

![图像组成](/assets/images/posts/2020-01-14-matplotlib/figure.jpg)


## 画图实战

1. 创建画布

    ```python
    fig, ax = plt.subplots(figsize=(14, 7))
    ```

    创建了大小为 (14, 7) 的画布，并把句柄给 fig，同时还在画布上创建了一个轴域 axes，赋值给 ax。今后 `fig.xxx` 是对这个画布的操作；`ax.xxx` 是对轴域的操作。

    > `fig, axes = plt.subplots(2, 1, figsize=(14, 7))`
    > 创建的两个轴域，那么用 axes[0], axes[1] 来表示两个轴域

2. 绘制数据

    我们的图像是在轴域中绘制的，所以用 `ax.plot` 来绘图

    ```python
    A = np.arange(1, 5)
    B = A ** 2
    C = A ** 3

    ax.plot(A, B)
    ax.plot(B, A)
    ```

    这样在轴域中绘制了两条曲线，剩下的是绘制辅助部分

3. 添加标题和坐标轴

    ```python
    ax.set_title("标题", fontsize=18)
    ax.set_xlabel("x 轴", fontsize=18, fontfamily='sans-serif')
    ax.set_ylabel("y 轴", fontsize='x-large', fontstyle='oblique')
    ax.legend()
    plt.show()
    ```

## 如何显示中文

1. 下载中文字体

    到 [github](https://github.com/yakumioto/YaHei-Consolas-Hybrid-1.12) 上下载 `YaHei Consolas Hybrid 1.12` 字体

2. 找到 matplotlib 配置文件路径啊

    ```bash
    >>> import matplotlib
    >>> print(matplotlib.matplotlib_fname())
    /home/wilson/anaconda3/lib/python3.7/site-packages/matplotlib/mpl-data/matplotlibrc
    ```

3. 拷贝把 YaHei Consolas Hybrid 1.12.ttf 字体

    ```bash
    $ cp YaHei\ Consolas\ Hybrid\ 1.12.ttf /home/wilson/anaconda3/lib/python3.7/site-packages/matplotlib/mpl-data/fonts/ttf
    ```

4. 修改配置文件

    修改 `/home/wilson/anaconda3/lib/python3.7/site-packages/matplotlib/mpl-data/matplotlibrc`

    去掉这3项前的注释符

    ```bash
    font.family: sans-serif
    font.sans-serif: YaHei Consolas Hybrid, ...
    axes.unicode_minus: False # 解决负号
    ```

5. 删除 matplotlib 的字体缓存

    ```bash
    $ rm -rf ~/.cache/matplotlib
    ```
