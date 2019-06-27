---
title: Linux 进阶篇
date: 2019-06-27 16:42:29 +0800
description: 
image:
    path: /assets/images/posts/2019-06-27-linux-advance/cover.jpg 
    thumbnail: /assets/images/posts/2019-06-27-linux-advance/thumb.jpg 
categories: 
    - it
tags:
    - linux
---

# 1. 编写 Shell 脚本

Shell终端解释器负责执行输入终端的各种指令，查看当前系统的命令行终端解释器指令为：

```bash
[root@linuxprobe ~]# echo $SHELL
/bin/bash
```

## 1.1 简单的脚本

首先创建一个脚本 `ex.sh`

```bash
[root@linuxprobe ~]# vim example.sh
#!/bin/bash 
#For Example BY linuxprobe.com 
pwd 
ls -al
```
- 第一行的脚本声明（#!）用来告诉系统使用哪种Shell解释器来执行该脚本
- 第二行是注释
- 第三、四行是脚本

## 1.2 接收参数

Shell脚本语言内设了用于接收参数的变量，含义如下：

|------|-------------------------------|
| 变量 | 功能                          |
|------|-------------------------------|
| `$0` | 当前Shell脚本程序的名称       |
| `$#` | 总共有几个参数                |
| `$*` | 所有位置的参数值              |
| `$?` | 显示上一次命令的执行返回值    |
| `$N` | 第N个位置的参数值，如 `$1,$2` |
|------|-------------------------------|

例如：

```bash
[root@linuxprobe ~]# vim example.sh
#!/bin/bash
echo "当前脚本名称为$0"
echo "总共有$#个参数，分别是$*。"
echo "第1个参数为$1，第5个为$5。"
[root@linuxprobe ~]# sh example.sh one two three four five six
当前脚本名称为example.sh
总共有6个参数，分别是one two three four five six。
第1个参数为one，第5个为five。
```


