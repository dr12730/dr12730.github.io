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
  - [linux, 编程, shell, 脚本]
---

<!-- vim-markdown-toc GFM -->

- [1. 编写 Shell 脚本](#1-编写-shell-脚本)
  - [1.1 简单的脚本](#11-简单的脚本)
  - [1.2 接收参数](#12-接收参数)
  - [1.3 判断输入](#13-判断输入)
    - [1.3.1 文件测试](#131-文件测试)
    - [1.3.2 整数比较语句](#132-整数比较语句)
    - [1.3.3 字符串比较](#133-字符串比较)
  - [1.4 流程控制语句](#14-流程控制语句)
    - [1.4.1 if 语句](#141-if-语句)
      - [语法格式](#语法格式)
      - [示例](#示例)
    - [1.4.2 for 语句](#142-for-语句)
      - [语法格式](#语法格式-1)
      - [示例](#示例-1)
    - [1.4.3 while 语句](#143-while-语句)
      - [语法格式](#语法格式-2)
      - [示例](#示例-2)
    - [1.4.4 case 语句](#144-case-语句)
      - [语法格式](#语法格式-3)
      - [示例](#示例-3)
- [2. 计划任务](#2-计划任务)
  - [2.1 临时任务](#21-临时任务)
  - [2.2 周期任务](#22-周期任务)
    - [2.2.1 格式与参数](#221-格式与参数)
    - [2.2.2 示例](#222-示例)

<!-- vim-markdown-toc -->

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

|--------|------------------------------|
| 操作符 | 作用 |
| :----: | ---------------------------- |
| -d | 文件是否为目录 |
| -f | 是否为一般文件 |
| -e | 文件是否存在 |
| -r/w/x | 当前用户是否有读/写/执行权限 |
|--------|------------------------------|

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

3. 判断 `/etc/cdrom` 文件是否存在，存在则输出 "存在"。这里利用了逻辑运算 `&&` 的特性

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

|--------|----------------|
| 操作符 | 作用 |
| :----: | -------------- |
| -eq | 是否等于 |
| -ne | 是否不等于 |
| -gt | 是否大于 |
| -lt | 是否小于 |
| -le | 是否等于或小于 |
| -ge | 是否大于或等于 |
|--------|----------------|

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

|--------|--------------------|
| 操作答 | 功能 |
| :----: | ------------------ |
| `=` | 字符串内容是否相同 |
| `!=` | 字符串不同 |
| `-z` | 字符串是否为空 |
|--------|--------------------|

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

#### 语法格式

if 作为判断语句，格式如下：

    ```bash
    if 条件判断; then
        执行语句
    elif 条件判断; then
        执行语句
    else
        执行语句
    fi
    ```

#### 示例

1.  判断 `~/workSpace/test` 目录是否存在，不存在创建
    ```bash
    #!/bin/bash
    DIR="$HOME/workSpace/test"
    if [ -e $DIR ]; then
        echo "$DIR 存在"
    else
        mkdir -p $DIR
    fi
    ```
2.  判断主机是否在线

    ```bash
    #!/bin/bash
    ping -c 3 -i 0.2 -W 3 $1 &> /dev/null
    if [ $? -eq 0]; then
        echo "$1 on-line"
    else
        echo "$1 off-line"
    fi
    ```

    > ping 的参数说明：
    >
    > - `-c` 规定尝试的次数
    > - `-i` 数据包的发送间隔
    > - `-W` 等待超时时间

    执行结果：

    ```bash
    [root@linuxprobe ~]# bash chkhost.sh 192.168.10.10
    192.168.10.10 On-line.
    [root@linuxprobe ~]# bash chkhost.sh 192.168.10.20
    192.168.10.20 Off-line.
    ```

3.  读取输入分数，判断成绩

    ```bash
    [root@linuxprobe ~]# vim chkscore.sh
    #!/bin/bash read -p "Enter your score（0-100）：" GRADE
    if [ $GRADE -ge 85 ] && [ $GRADE -le 100 ] ; then
        echo "$GRADE is Excellent"
    elif [ $GRADE -ge 70 ] && [ $GRADE -le 84 ] ; then
        echo "$GRADE is Pass"
    else
        echo "$GRADE is Fail"
    fi
    ```

    > read 读取输入，-p 显示提示信息

        ```bash
        #!/bin/bash
        [root@linuxprobe ~]# bash chkscore.sh
        Enter your score（0-100）：88
        88 is Excellent
        [root@linuxprobe ~]# bash chkscore.sh
        Enter your score（0-100）：80
        80 is Pass
        ```

### 1.4.2 for 语句

#### 语法格式

    ```bash
    #!/bin/bash
    for 变量 in 取值列表; do
        语句
    done
    ```

#### 示例

1. 根据用户列表 user.txt，读取用户输入密码，创建用户

   ```bash
   #!/bin/bash
   read -p "请输入密码：" PASSWD
   for UNAME in $(cat user.txt); do
       id $UNAME &> /dev/null
       if [ $? -eq 0 ]; then
           echo "用户已存在"
       else useradd $UNAME &> /dev/null
           echo "$PASSWD" | passwd --stdin $UNAME &> /dev/null
           if [ $? -eq 0 ]; then
               echo "$UNAME 创建成功"
           else
               echo "$UNAME 创建失败"
           fi
       fi
   done`
   ```

   > - id 用户名：查看用户信息

2. 批量查看主机在线

   ```bash
   #!/bin/bash

   HLIST=$(cat $HOME/ipaddr.txt)

   for IP in $HLIST; do
       ping -c 3 -i 0.2 -W 3 $IP &> /dev/null
       if [ $? -eq 0 ]; then
           echo "$IP 在线"
       else
           echo "$IP 不在线"
       fi
   done
   ```

### 1.4.3 while 语句

#### 语法格式

    ```bash
    while 条件判断; do
        语句
    done
    ```

#### 示例

1. 猜数字

   ```bash
   #!/bin/bash

   PRICE=$(expr $RANDOM % 1000)
   TIMES=0
   echo "请输入 0~999 之间的数字"
   while true; do
       read -p "请输入数字：" NUMBER
       let TIMES++
       if [ $NUMBER -eq $PRICE ]; then
           echo "正确，你猜了 $TIMES 次"
           exit 0
       elif [ $NUMBER -gt $PRICE ]; then
           echo "太高了"
       else
           echo "太低了"
       fi
   done
   ```

### 1.4.4 case 语句

#### 语法格式

    ```bash
    case ${VAR} in
    pattern1)
        commands1
        ;;
    pattern2)
        commands2
        ;;
    esac
    ```

注意的是, case 比较的是 pattern，然后既然是通配符，那么：

1. 切记通配符本身不能用引号括起来。
2. 而对于变量 VAR 是否使用双引号括起来都可以。
3. 另外要记住通配符(pattern)和规则表达式(regular expression)的区别。

#### 示例

    ```bash
    #!/bin/bash
    read -p "输入一个字符：" KEY
    case $KEY in
    [a-z]|[A-Z])
        echo "输入的是字母"
        ;;
    [0-9])
        echo "输入的是数字"
        ;;
    *)
        echo "其他字符"
    esac
    ```

# 2. 计划任务

计划任务可以完成周期性、规律性的工作。

## 2.1 临时任务

|-----------|--------------------|
| 命令 | 功能 |
|-----------|--------------------|
| at 时间 | 在规定时间完成任务 |
| at -l | 查看未执行任务 |
| atrm 编号 | 删除任务 |
|-----------|--------------------|

非交互式执行临时任务：

```bash
[root@linuxprobe ~]# echo "systemctl restart httpd" | at 23:30
job 4 at Mon Apr 27 23:30:00 2015
```

## 2.2 周期任务

### 2.2.1 格式与参数

周期任务用 `crond` 系统服务，格式为：crontab -e 分、时、日、月、星期 命令

|------------|--------------|
| 命令 | 功能 |
|------------|--------------|
| crontab -e | 编辑任务 |
| crontab -l | 任务列表 |
| crontab -r | 删除任务 |
| crontab -u | 编辑他人任务 |
|------------|--------------|

|------|-------------------|
| 字段 | 说明 |
|------|-------------------|
| 分 | 0~59 |
| 时 | 0~23 |
| 日 | 1~31 |
| 月 | 1~12 |
| 星期 | 0~7，0、7 均为周日 |
| 命令 | 要执行的脚本 |
|------|-------------------|

### 2.2.2 示例

每周一、三、五凌晨 3 点 25 分，用 tar 命令把数据目录打包为一个备份文件

```bash
crontab -e
25 3 * * 1,3,5 /usr/bin/tar -czvf backup.tar.gz /home/data
0 1 * * 1-5 /usr/bin/rm -rf /tmp/*
```

> **注意**：
>
> 1. `,` 表多个时间点
> 2. `-` 表时间段
> 3. `/` 表任务间隔时间，如 `*/2` 每隔 2 分钟
> 4. 必须使用绝对路径
> 5. 分必须有值，不能为空或 `*`
> 6. 日和星期不能同时有效，会冲突
