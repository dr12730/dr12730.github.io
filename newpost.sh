#!/bin/bash

date=`date +'%Y-%m-%d'`
name="new.post"

if [ $# -gt 0 ]; then
    name=$1
fi

mkdir "assets/images/posts/"$date-$name
imagedir="assets/images/posts/"$date-$name

fn="_posts/"$date-$name".md"
echo "will creat file:" $fn
pic=$picname

if [ ! -f $fn ]; then
    touch $fn
    echo "---">>$fn
    echo "title: $name">>$fn
    echo "date: "`date +'%Y-%m-%d %H:%M:%S'`" +0800" >>$fn
    echo "key: "$date-$name >>$fn
    echo "cover: /$imagedir/cover.jpg">>$fn
    echo "mode: immersive">>$fn
    echo "header:">>$fn
    echo "  theme: dark">>$fn
    echo "article_header:">>$fn
    echo "  type: overlay">>$fn
    echo "  theme: dark">>$fn
    echo "  background_color: '#203028'">>$fn
    echo "  background_image:">>$fn
    echo "    gradient: 'linear-gradient(135deg, rgba(34, 139, 87 , .4), rgba(139, 34, 139, .4))'">>$fn
    echo "    src: /$imagedir/header_image.jpg">>$fn
    echo "mathjax: false">>$fn
    echo "mathjax_autoNumber: false">>$fn
    echo "mermaid: false">>$fn
    echo "chart: false">>$fn
    echo "tags: " >>$fn
    echo "---" >>$fn
    echo " ">>$fn
    echo "本文简介">>$fn
    echo " ">>$fn
    echo "<!--more-->">>$fn
    echo "Creat sucess!"
else
    echo "Creat fail! File is existed"
fi
