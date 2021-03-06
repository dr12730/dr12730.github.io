---
layout: article
title: VIM 之安装与配置篇
date: 2019-06-16 15:45:04 +0800
cover:  assets/images/posts/2019-06-16-vim-install/cover.jpg
header:
  theme: dark
  background: 'linear-gradient(135deg, rgb(34, 139, 87), rgb(139, 34, 139))'
article_header:
  type: overlay
  theme: dark
  background_color: '#203028'
  background_image:
    gradient: 'linear-gradient(135deg, rgba(34, 139, 87 , .4), rgba(139, 34, 139, .4))'
    src: assets/images/posts/2019-06-16-vim-install/header_image.jpg
tags:
    - vim
---

<!--more-->

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


# 1.4 插件管理器

## 1.4.1 vundle 插件管理器
vim 有许多好用的插件，这些插件的管理和安装可以通过 vundle 来完成；而 vim 自身的各种特性，则由 `~/.vimrc` 来配置

关于插件的安装，推荐直接下载 [wKevin](https://github.com/wkevin/DotVim) 的配置，很方便，执行如下命令：

```python
# 下载 DotVim 配置
git clone https://github.com/wkevin/DotVim.git

cd DotVim

# 执行部署命令
sh deploy.sh
```

下面，我们对这个部署命令 `deploy.sh` 的内容进行一下说明，因为有几个命令我稍作了修改，所以和 `deploy.sh` 有点区别：
```python
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

### 1.4.2 vim-plug 插件管理器

vim-plug 是新一代的插件管理器，现在大有超越 vundle 的趋势

- vim   
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


### 1.5 主题设置

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

### 1.6 安装 vim8 的中文手册
感谢 yianwillis 对 vim 手册的翻译，我们可以在他的 [github](https://github.com/yianwillis/vimcdoc) 上找到对应的中文文档源文件

也可以通过 vundle 直接安装，方法是在 `.vimrc` 中添加：
```shell
Plugin "yianwillis/vimcdoc"
```

重启 Vim 后执行 `:PluginInstall`

也可以进入 yianwillis 的 [网站](http://yianwillis.github.io/vimcdoc/doc/help.html) 直接阅读手册。

现在 vim 的基本安装和配置就完成了

# 2 vim 的配置

通过对 vim 的配置，可以打开许多 vim 特有的功能，让使用更加方便。vimrc 是 vim 的配置文件，再它打开之前，会先加载这个文件，根据 vimrc 决定编译器自身各类属性的设置以及各种功能的开启和关闭。 


vim 的配置文件 vimrc 基本内容就是设置选项的开关状态或数值，还有自定义操作的映射，以及之后安装的各类插件的设置三大部分。


那么，现在我们来看看这个 vimrc 怎么编写。


首先，在 Linux 下 vimrc 的存放路径是 `~/.vimrc` 或者 `~/.vim/vimrc`，如果没有就自行创建一个

```python
mkdir ~/.vim      # 创建.vim 目录
vim ~/.vim/vimrc  # 打开空的 vimrc 文件
```


这样就通过 vim 打开了一个 vimrc 配置文件，下面是一个配置的部分示例：

```vimrc

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"
" =>  Vim/NeoVim 编辑器设置
"
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" 打开文件类型识别
filetype on
" 根据文件类型设置缩进
filetype indent on
filetype plugin on
filetype plugin indent on

set encoding=utf-8

" 空格 + 回车，取消搜索高亮
noremap <LEADER><CR> :nohlsearch<CR>
" 保存文件
noremap <leader>w :w<cr>
map R :source ~/.config/nvim/init.vim<CR>


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"
" =>  Vim/NeoVim 其他设置
"
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

```

#### 基本映射 map

我们在上面的 vimrc中看到了**状态开关**、**状态值**和**自定义映射**三类设置方式，这里介绍一下 vim 映射的概念。

映射就是为一些操作创建新的快捷方式，它的格式为：

```
map 新操作 旧操作
map jj <Esc> # 把 Esc 映射为 jj
```

- 用 nmap/vmap/imap 定义只在 normal/visual/insert 模式下有效的映射  
- `:vmap \ U` 在 visual 模式下把选中的文本进行大小写转换 (u/U 转换大小写)  
- `:imap <c-d> <esc>ddi` 在 insert 模式下删除一行

#### 递归与非递归映射

使用 `map` 进行的映射会有递归映射的问题：

1. `map` 是递归映射，比如 `map - dd` 和 `map \ -`，使用 `\` 后会删除一行  
2. 多个插件间的映射会混乱

解决方法：

使用非递归映射，`nnoremap/vnoremap/inoremap`，所以为了自己和插件作者，**我们在任何时候都应该使用非递归映射 `noremap`**

推荐一本书：[《笨方法学 VimScript》](https://www.kancloud.cn/kancloud/learn-vimscript-the-hard-way/49321)

了解了 vimrc 的映射，我们就可以自己编写 vimrc 的配置了，也可以参考各个大神写的配置文件，拿来自己用。

# Vim 的高阶配置

```vim
" 总是显示屏幕最下面的5行
set scrolloff=5
```
