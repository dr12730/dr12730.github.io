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

Shell 终端解释器负责执行输入终端的各种指令，查看当前系统的命令行终端解释器指令为：

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

- 第一行的脚本声明（#!）用来告诉系统使用哪种 Shell 解释器来执行该脚本
- 第二行是注释
- 第三、四行是脚本

## 1.2 接收参数

Shell 脚本语言内设了用于接收参数的变量，含义如下：

|------|-------------------------------|
| 变量 | 功能 |
|------|-------------------------------|
| `$0` | 当前 Shell 脚本程序的名称 |
| `$#` | 总共有几个参数 |
| `$*` | 所有位置的参数值 |
| `$?` | 显示上一次命令的执行返回值 |
| `$N` | 第 N 个位置的参数值，如 `$1,$2` |
|------|-------------------------------|

例如：

```bash
[root@linuxprobe ~]# vim example.sh
#!/bin/bash
echo "当前脚本名称为$0"
echo "总共有$#个参数，分别是$*。"
echo "第1个参数为$1，第5个为$5。"
```

执行结果：

```bash
[root@linuxprobe ~]# sh example.sh one two three four five six
当前脚本名称为example.sh
总共有6个参数，分别是one two three four five six。
第1个参数为one，第5个为five。
```

## 1.3 判断输入

Shell 脚本中的条件测试语法可以判断表达式是否成立，若条件成立则返回数字 0，否则便返回其他随机数值。条件判断句的格式为：

```bash
[ 条件表达式 ]
```

> 注意：表达式两边有空格

### 1.3.1 文件测试

| 操作符 | 作用                         |
| :----: | ---------------------------- |
|   -d   | 文件是否为目录               |
|   -f   | 是否为一般文件               |
|   -e   | 文件是否存在                 |
| -r/w/x | 当前用户是否有读/写/执行权限 |

比如，

1. 测试 `/etc/fstab` 是否为目录，并通过解释器的内设变量 `$?` 显示上一条语句执行的返回值，为 0 则目录存在，非零则不存在。

```bash
[root@linuxprobe ~]# [ -d /etc/fstab ]
[root@linuxprobe ~]# echo $?
1
```

2. 再判断 `/etc/fstab` 是否为文件，为 0 则是，非 0 则不是

```bash
[root@linuxprobe ~]# [ -f /etc/fstab ]
[root@linuxprobe ~]# echo $?
0
```

3. 判断 `/etc/cdrom` 文件是否存在，存在则输出 "存在"`。这里利用了逻辑运算`&&` 的特性

```bash
[root@linuxprobe ~]# [ -e /etc/cdrom ] && echo "存在"
```

4. 判断当前用户是否是管理员

```bash
# 前面的命令失败后，才会执行后面的命令
[root@linuxprobe ~]# [ $USER = root ] || echo "user"
[root@linuxprobe ~]# su - wilson
[root@linuxprobe ~]# [ $USER = root ] || echo "user"
user
# 逻辑非
[root@linuxprobe root]# [ $USER != root ] || echo "administrator"
```

5. 判断是否为 root 用户，是输出 root 否输出 user

```bash
[root@linuxprobe root]# [ $USER != root ] && echo "user" || echo "root"
```

### 1.3.2 整数比较语句

整数比较运算符只能对整数生效，不能面对字符串、文件。因为 `>`、`<`、`=` 都另有它用，所以只能用规范的运算符

| 操作符 | 作用           |
| :----: | -------------- |
|  -eq   | 是否等于       |
|  -ne   | 是否不等于     |
|  -gt   | 是否大于       |
|  -lt   | 是否小于       |
|  -le   | 是否等于或小于 |
|  -ge   | 是否大于或等于 |

举例：

```bash
$ [ 10 -eq 10 ]
$ echo $?
$ 0
```

获取当前系统可用的内存量信息，当可用量小于 1024 时显示内存不足：

```bash
[root@linuxprobe ~]# free -m
            total     used     free     shared     buffers     cached
Mem:        1826      1244     582      9          1           413
-/+ buffers/cache:    830 996
Swap:       2047      0        2047

[root@linuxprobe ~]# free -m | grep Mem:
Mem:        1826      1244     582      9

[root@linuxprobe ~]# free -m | grep Mem: | awk '{print $4}'
582

[root@linuxprobe ~]# FreeMem=`free -m | grep Mem: | awk '{print $4}'`
[root@linuxprobe ~]# echo $FreeMem
582
```

显示内存不足：

```bash
[root@linuxprobe ~]# [ $FreeMem -lt 1024] && echo "内存不足"
```

### 1.3.3 字符串比较

| 操作答 | 功能               |
| :----: | ------------------ |
|  `=`   | 字符串内容是否相同 |
|  `!=`  | 字符串不同         |
|  `-z`  | 字符串是否为空     |

比如判断是否定义了变量 `String`：

```bash
[root@linuxprobe ~]# [ -z $String ]
[root@linuxprobe ~]# echo $?
0
```

当前环境不是英语时，显示非英语环境：

```bash
[root@linuxprobe ~]# [ $LANG != "en.US" ] && echo "非英语环境"
```

## 1.4 流程控制语句

### 1.4.1 if 语句

if 语句由 if, then, fi 构成。
