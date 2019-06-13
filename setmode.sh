#!/bin/bash

if [ "$1" == "r" ]; then
    cp _config.yml.remote _config.yml
    cp Gemfile.remote Gemfile
    echo "设置为 github remote 模式"
fi

if [ "$1" == "l" ]; then
    cp _config.yml.local _config.yml
    cp Gemfile.local Gemfile
    echo "设置为本地 local 模式"
fi
