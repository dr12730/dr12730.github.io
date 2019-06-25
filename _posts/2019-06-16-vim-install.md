---
title: VIM 之安装与配置篇
date: 2019-06-16 15:45:04 +0800
description:
image:
    path: assets/images/posts/2019-06-16-vim-install/cover.jpg
    thumbnail: assets/images/posts/2019-06-16-vim-install/thumb.jpg
categories:
    - it
tags:
    - vim
---

只删除author，是否有影响

vim 是一个优秀的文本编辑器，它可以在手不离开键盘的条件下完成大部分的工作，而且在没有图形界面的服务器端也可以使用。这篇博文主要记录在 linux 环境下 vim 的安装与配置。

# 1. 安装

按照 vim [官方网站](https://www.vim.org/download.php) 的说明，可以从 vim 的 git 官方下载、编译安装，方法如下：


```shell
git clone https://github.com/vim/vim.git
cd vim/src
make distclean
make
```

但在实践中，却会出现问题，原因是我们使用的是中文系统，因此用的是下面的方法：

## 1.1 彻底卸载操作系统自带的 vim

- 查看系统中安装的 vim 组件：
```shell
dpkg -l | grep vim
```
如果 vim 是通过 dpkg 安装的话，会列出相应的 vim 组件

- 删除对应的 vim
```shell
sudo apt-get remove --purge vim vim-tiny vim-runtime vim-common
vsudo apt-get remove --purge vim-doc vim-scripts
vsudo apt-get remove --purge gvim vim-gui-common
sudo apt-get clean
```

## 1.2 安装编译所需依赖库

```shell
sudo apt install -y libncurses5-dev libgnome2-dev libgnomeui-dev libgtk2.0-dev \
libatk1.0-dev libbonoboui2-dev libcairo2-dev libx11-dev libxpm-dev libxt-dev \
python-dev python3-dev ruby-dev lua5.1 liblua5.1-dev libperl-dev git
```
> 也许还有一些依赖，在编译时根据当时的提示择情安装

## 1.3 获取源代码并编译
按照下面的步骤依次执行

```shell
# 下载 vim 源码
git clone https://github.com/vim/vim.git

# 进入源码目录
cd vim/src

# 清理编译环境
distclean

# 配置编译选项
./configure --with-features=huge \
--enable-multibyte \
--enable-python3interp=yes \
--with-python3-config-dir=/usr/local/lib/python3.7/config-3.7m-x86_64-linux-gnu/ \
--enable-gui=gtk2 \
--enable-cscope

# 开始编译
make

# 编译成功后，安装
sudo make install
```

> 其中参数说明如下：  
> –with-features = huge：支持最大特性  
> –enable-multibyte：多字节支持可以在 Vim 中输入中文  
> –enable-rubyinterp：启用 Vim 对 ruby 编写的插件的支持  
> –enable-pythoninterp：启用 Vim 对 python2 编写的插件的支持  
> –enable-python3interp: 启用 Vim 对 python3 编写的插件的支持  
> –enable-luainterp：启用 Vim 对于 lua 编写的插件的支持  
> –enable-perlinterp：启用 Vim 对 perl 编写的插件的支持  
> –enable-cscope：Vim 对 cscope 支持  
> –with-python-config-dir=/usr/lib/python2.7/config-x86_64-linux-gnu : 指定 python 路径  
> –enable-gui = gtk2：gtk2 支持，也可以使用 gnome，表示生成 gvim  
> -prefix = / usr：编译安装路径  

至此，vim 安装完成，在终端输入 `vim` 验证安装是否完整，也可以执行 `vim --version` 查看它的版本号


# 2. 插件与配置

## 2.1 vundle 插件管理器
vim 有许多好用的插件，这些插件的管理和安装可以通过 vundle 来完成；而 vim 自身的各种特性，则由 `~/.vimrc` 来配置

关于插件的安装，推荐直接下载 [wKevin](https://github.com/wkevin/DotVim) 的配置，很方便，执行如下命令：

```shell
# 下载 DotVim 配置
git clone https://github.com/wkevin/DotVim.git

cd DotVim

# 执行部署命令
sh deploy.sh
```

下面，我们对这个部署命令 `deploy.sh` 的内容进行一下说明，因为有几个命令我稍作了修改，所以和 `deploy.sh` 有点区别：
```shell
git clone http://github.com/gmarik/vundle.git ~/.vim/bundle/vundle
ln -s $PWD/.vimrc ~/
vim +BundleInstall +BundleClean! +qa
# cp -r snippets/ ~/.vim/snippets/
cp -r colors/ ~/DotVim/colors/
ln -s $PWD/tools/callgraph /usr/local/bin/
ln -s $PWD/tools/tree2dotx /usr/local/bin/
```

**说明**：

1. 从 github 下载 vundle 插件管理器
2. 将 vim 的配置文件软连接为 DotVim 目录下的 .vimrc 配置文件
    `.vimrc` 是 vim 的配置文件，里面设置了 vim 的各类属性，比如字体、主题、自定义快捷键等等
3. 启动 vim，并执行 BundleInstall 指令，通过 vundle 安装 .vimrc 中要求安装的插件
    要安装的插件，写在 `.vimrc` 的 Plugin 后
4. colors 是各类主题颜色， snippets 是一些常用代码，因为 bundle 也会安装相关的 snipptes，所以可以不用执行它的拷贝
5. callpraph 和 tree2dotx 可以链接到 /usr/local/bin 下

这样再进入 vim 就与之前不一样了

## 2.2 vim-plug 插件管理器

-vim 
    - 安装
    ```shell
    curl -fLo ~/.vim/autoload/plug.vim --create-dirs \
    https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim
    ```
    - 管理插件
    在. vimrc 中写入要安装的插件名称
    ```shell
    call plug#begin('~/.vim/plugged')
    Plug 'vim-airline/vim-airline'
    call plug#end()
    ```
    - 安装插件
    保存 `.vimrc` 后，重进 vim，再执行 `:PlugInstall` 安装插件

- Neovim
    - 安装
    ```
        curl -fLo ~/.local/share/nvim/site/autoload/plug.vim --create-dirs \
    https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim
    ```


## 2.2 主题
通过修改 `.vimrc` 中的 `colorscheme` 可以修改主题，修改的部分如下：

```shell
set background=dark
colorscheme herald
```

**herald 主题**

![herald](/assets/images/posts/2019-06-16-vim-install/herald.png)


**zellner 主题**

![zellner](/assets/images/posts/2019-06-16-vim-install/zellner.png)

**lucius 主题**

![lucius](/assets/images/posts/2019-06-16-vim-install/lucius.png)


**moria 主题**

![moria](/assets/images/posts/2019-06-16-vim-install/moria.png)

# 3. vim8 的中文手册
感谢 yianwillis 对 vim 手册的翻译，我们可以在他的 [github](https://github.com/yianwillis/vimcdoc) 上找到对应的中文文档源文件

也可以通过 vundle 直接安装，方法是在 `.vimrc` 中添加：
```shell
Plugin "yianwillis/vimcdoc"
```

重启 Vim 后执行 `:PluginInstall`

也可以进入 yianwillis 的 [网站](http://yianwillis.github.io/vimcdoc/doc/help.html) 直接阅读手册。

现在 vim 的基本安装和配置就完成了
