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
    echo "title: ">>$fn
    echo "date: "`date +'%Y-%m-%d %H:%M:%S'`" +0800" >>$fn
    echo "description: " >>$fn
    echo "author: wilson" >>$fn
    echo "image:      " >>$fn
    echo "    path: $imagedir/cover.jpg " >>$fn
    echo "    thumbnail: $imagedir/thumb.jpg " >>$fn
    echo "categories: " >>$fn
    echo "    - " >>$fn
    echo "tags:" >>$fn
    echo "    - " >>$fn
    echo "---" >>$fn
    echo "Creat sucess!"
else
    echo "Creat fail! File is existed"
fi
