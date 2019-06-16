---
title: VIM 之基本概念篇
date: 2019-06-16 15:55:36 +0800
description: 
image:      
    path: assets/images/posts/2019-06-16-vim-concept/cover.jpg 
    thumbnail: assets/images/posts/2019-06-16-vim-concept/cover.jpg
categories: 
    - it
tags:
    - vim
---

# 1. 撤消与回退

以删除为例，通过 `x` 可以删除字符，通过 `u` 可以撤消删除操作，通过 `CTRL-r` 可以回退一个 `u` 操作，也就是说 `u` 撤消 `x`， `CTRL-r` 撤消 `x`。的确有点绕，我们来看一个例子：

```shell
  xxx  # <-- 光标移到数字 3 处，连用 3 个 x
12345678910
12678910  # 连用 3 个 x 删除了 345
125678910 # 用 u 撤消 x 操作后，恢复最后被删除的 5
1245678910 # 连用 2 个 u 撤消 x 操作后，恢复最后被删除的 45
125678910  # 用 ctrl+r 回退一个 u
```

还有一个 `U`(行撤销)，它是取消最近在一行上的所有操作
```shell
A very intelligent turtle
  xxxx # 删除 very

A intelligent turtle
              xxxxxx  # 删除 turtle
# 删除后结果
A intelligent
# 用 "U" 恢复最近对此行的所有操作
A very intelligent turtle
```

# 2. 准确搜索
如果你输入 "/the"，你也可能找到 "there"
要找到以 "the" 结尾的单词，可以用: `/the\>`
`\>` 是一个特殊的记号，表示只匹配单词末尾。类似地，`\<` 只匹配单词的开头。 这样，要匹配一个完整的单词 "the"，只需:`/\<the\>`，这不会匹配 "there" 或者 "soothe"。注意 `*` 和 `#` 命令也使用了 "词首" 和 "词尾" 标记来匹配整个单词 (要部分匹配，使用 "g\*" 和 "g#")

还可以只匹配行首与行尾：
```shell
# "x" 标记出被 "the" 模式匹配的位置:
the solder holding one of the chips melted and the
xxx                       xxx                  xxx

#用 "/the$" 则匹配如下位置:
the solder holding one of the chips melted and the
                                               xxx
# 而使用 "/^the" 则匹配:
the solder holding one of the chips melted and the
xxx
```

匹配单个字符:
"." 字符匹配任何字符。例如，模式 "c.m" 匹配一个字符串，它的第一个字符是 c， 第二个字符是任意字符，而第三个字符是 m。例如:

```shell
# c.m 匹配结果
We use a computer that became the cummin winter.
         xxx             xxx      xxx
```

# 3. 跳转
把光标移到本行之外的操作，都是跳转（j, k 除外）。vim 会对跳转前的位置作一个标记， 可以用 \` \` 跳转回来。而 `CTRL-o` 可以跳转到较老 (old) 一点的标记处，而 `CTRL-i` 跳转到新的标记。比如如下命令：

```shell
33G   # 跳到 33 行
/^The  # 向下跳到开头为 The 的行首
CTRL-o  # 返回 33 行
CTRL-o  # 返回最初的地方
CTRL-i  # 跳到 33 行
CTRL-i  # 跳到 The 的街道
```

# 4. 重复操作
`.`是重复最后一次操作，比如：

```shell
/four<Enter>  # 找到第一个 "four" 
cwfive<Esc>   # 修改成 "five" 
n             # 找下一个 "four" 
.             # 重复修改到 "five" 的操作 
n             # 找下一个 "four" 
.             # 重复修改 如此类推...... 
```

# 5. 文本对象
在一个单词中间，要删除这个单词时，可以用 `daw` 实现。这里的 "aw" 是一个文本对象，aw 表示 "A word"，这样，"daw" 就是删除一个单词，包括后面的空格。  
用 "cis" 可以改变一个句子。看下面的句子:  

```shell
Hello there.  This 
is an example.  Just 
some text. 
```

移动到第二行的开始处。现在使用 "cis":  
```shell
Hello there.    Just 
some text. 
```

现在你输入新的句子 "Another line.":

```
Hello there.  Another line.  Just 
some text. 
```

"cis" 包括 "c" (change，修改) 操作符和 "is" 文本对象。这表示 "Inner Sentence"
(内含句子)。还有一个文本对象是 "as"，区别是 "as" 包括句子后面的空白字符而 "is"
不包括。如果你要删除一个句子，而且你还想同时删除句子后面空白字符，就用 "das"；
如果你想保留空白字符而替换一个句子，则使用 "cis"。   
`ci"`是匹配一对"中的内容


# 6. 简单映射

```shell
# 用一个大括号将一个特定的单词括起来
:map <F5> i{<Esc>ea}<Esc>
```
这个命令分解如下：  

- <F5>
    F5 功能键。这是命令的触发器。当这个键被按下时，相应的命令即被执行。
- i{<Esc>     
    插入 { 字符。<Esc> 键用于退出插入模式。
- e
    移动到词尾。
- a}<Esc>     
    插入 } 到单词尾。

> 为了不与系统的映射冲突，可以用反斜杠来定义自已的映射，比如：
> `:map \p i(<Esc>ea)<Esc>`

# 7. 使用寄存器
```shell
# 拷贝一个句子(a sentence) 到 f 寄存器
"fyas
# 拷贝三个整行到寄存器 l (l 表示 line)
"l3Y
# 要拷贝一个文本列块到寄存器 b (代表 block) 中
CTRL-Vjjww"by
# 粘贴 f 寄存器的内容
"fp
```

# 8. 分割窗口

| 序号 | 命令 | 功能 |
|:---:|---|---|
| 1 | :split | 打开新窗口 |
| 2 | CTRL-w w | 窗口间跳转 |
| 3 | :close | 关闭窗口 |
| 4 | :only | 关闭其他窗口 |
| 5 | :split two.c | 打开新窗口编辑 two.c |
| 6 | :3split a.c | 打开3行大小的窗口 |
| 7 | 4CTRL-w +/- | 增加/减小窗口4行 |
| 8 | 5CTRL-w _ | 窗口设定为5行高 |
| 9 | :vs | 垂直打开新窗口 |
| 10 | CTRL-w h/j/k/l | 窗口间跳转 |
| 11 | CTRL-w H/J/K/L | 把当前窗口放到最左/下/上/右面 |
| 12 | :qall | 退出所有窗口 |
| 13 | :splitbelow | 当前窗口下打开新窗口 |
| 14 | :splitright | 当前窗口右侧开新窗口 | 

> 另一种新开窗口的方法：  
> 前置“s”，如“:tag”跳转到一个标记，用“:stag”成分割新窗口再跳转到标记  
> 前置 CTRL-w 开新窗口，比如 `CTRL-w CTRL-^` 新开窗口编辑轮换文件  

# 9. 比较文件差异

## 9.1. 比较两个文件 

```shell
vimdiff main.c~ main.c
```
vim 用垂直窗口打开两个文件

## 9.2. 从vim中比较另一个文件

```shell
:vertical diffsplit main.c~
```
## 9.3. 合并差异

| 序号 | 命令 | 功能 |
|:---:|---|---|
| 1 | ]c | 跳转到下个修改点 |
| 2 | [c | 跳转到上个修改点 |
| 3 | :diffupdate | 更新高亮显示 |
| 4 | :dp | diff put，把左边(当前窗口)文件拷到右边 |
| 5 | :do | diff obtain，把左边窗口文本拉到右边(当前窗口) |


# 10. 标签页

| 序号 | 命令 | 功能 |
|:---:|---|---|
| 1 | :tabedit newfile | 新建 newfile 标签页 |
| 2 | gt | 跳转到 newfile 标签页, Goto Tab |
| 3 | :gt help gt | 在新标签页中打开 gt 帮助 |
| 4 | :tabonly | 仅保留当前标签页 |


# 11 大修改
## 11.1 记录与回放 - 宏

制作宏的步骤如下：  
1. "q{register}" 命令启动一次击键记录，结果保存到 {register} 指定的寄存器中。
    寄存器名可以用 a 到 z 中任一个字母表示。
2. 输入你的命令。
3. 键入 q (后面不用跟任何字符) 命令结束记录。
4. 用 "@{register}" 执行宏

对于  
```
stdio.h 
fcntl.h 
unistd.h 
stdlib.h 
```

而你想把它变成这样:

```shell
#include "stdio.h" 
#include "fcntl.h" 
#include "unistd.h" 
#include "stdlib.h" 
```

先移动到第一行，接着执行如下命令:

```shell
qa                 #    启动记录，并使用寄存器 a
^                  #    移到行首
i#include "<Esc>   #    在行首输入 #include "
$                  #    移到行末
a"<Esc>            #    在行末加上双引号 (")
j                  #    移到下一行
q                  #    结束记录
```
现在，你已经完成一次复杂的修改了。你可以通过 "3@a" 完成余下的修改。  

把光标移到相应位置输入 "@a"即可。也可以用 “@@” 完成相同操作。对于 "." 只能重复一个动作，而 @a 是一个宏，这就是它们的区别。

## 11.2 修改宏
如果有一个复杂的操作宏，我们可以对宏进行修改：

```shell
G        #  移到行尾
o<Esc>   #  建立一个空行
"np      #  拷贝 n 寄存器中的文本，你的命令将被拷到整个文件的结尾
{edits}  #  像修改普通文本一样修改这些命令
0        #  回到行首
"ny$     #  把正确的命令拷贝回 n 寄存器
dd       #  删除临时行
```

## 11.3 追加寄存器
假设寄存器 a 中记录了一个宏，但加再附加一个新命令，可以通过下面方式实现：

```shell
qA/word<Enter>q
```

qA 或 "A 表示启用 A 寄存器，但也会将后面的 `/word<Enter>` 追加到小写寄存器 a 中。  

这种方法在宏记录，拷贝和删除命令中都有效。例如，你需要把选择一些行到一个寄存器中，可以先这样拷贝第一行:

```shell
        "aY
```

然后移到下一个要拷贝的地方，执行:  
```shell
        "AY
```

如此类推。这样在寄存器 a 中就会包括所有你要拷贝的所有行。

# 11.4 替换

```shell
# %指命令作用于全部行，g指对所有匹配点起作用
:%s/Prof/Teacher/g
```

还有一个 c 选项，是替换前询问，它会打印：替换为 Teacher 么 (y/n/a/q/l/^E/^Y)?  
其中：  

| 提示符 | 说明 |
|---|---|
| y       |  Yes，是；执行替换 |
| n       |  No，否；跳过 |
| a       |  All，全部；对剩下的匹配点全部执行替换，不需要再确认 |
| q       |  Quit，退出；不再执行任何替换 |
| l       |  Last，最后；替换完当前匹配点后退出 |
| CTRL-E  |  向上滚动一行 |
| CTRL-Y  |  向下滚动一行 |

> 当要搜索“/”时，可以在前面加转义符 `\`，也可以用加号代替“/”
> `:s+one/two+one or two+`  这里的 + 就是分隔符 “/”

## 11.5 在范围中使用模式匹配
我们知道 `:.+3,$-5s/this/that/g` 是在范围 [当前行+3, 最后一行-5] 内执行替换命令。(包括第 5 行)  
假如你只想把第3章所有的 "grey" 修改成 "gray"。其它的章节不变。另外，你知道每章的开头的标志是行首的单词为 "Chapter"。下面的命令会对你有帮助:

```shell
        :?^Chapter?,/^Chapter/s=grey=gray=g
```

你可以看到这里使用了两个查找命令。第一个是 "?^Chapter?"，用于查找前一个行首的 "Chapter"，就是说 "?pattern?" 用于向前查找。同样，"/^Chapter/" 用于向后查找下一章。   
斜杠使用的混淆，在这种情况下，"=" 字符用于代替斜杠。使用斜杠或使用其它字符其实也是可以的。

我们还可以用标记来指定范围，比如已通过 "ms" 和 "me" 来标记了开始和结尾，那么可以用 `:'t,'b` 来指定范围

还可以用可视模式选中行，然后输入 ":" 启动命令模式，会看到 `'<, '>`，它们是可视模式的开始和结尾标记，之后再输入剩下的命令。这两个标记一直有效，我们甚至可以用 `:\`>,$` 来选择结尾到文件未的部分。

我们还可以指定当前向下多少行，比如输入 `5:`，则会得到 `:.,.+4` 的结果，然后继续输入命令

另外，`:g` 命令可以找到一个匹配点，并在那里执行一条指令，形式一般是：

```shell
:[range]global/{pattern}/{command}
```

比如 `:g+//+s/foobar/barfoo/g` 这个命令用 ":g" 开头，然后是一个匹配模式，由于模式中包括正斜杠，我们用加号作分隔符，后面是一个把 "foobar" 替换成 "barfoo" 的替换命令。全局命令的默认范围是整个文件，所以这个例子中没有指定范围。

## 11.5 可视列块模式
### 11.5.1 插入文本

用 `CTRL-v` 选择矩形文本块，再用 `I` 输入文本，文本将插在每行的行首；用可视模块选择 long，然后可以进行修改、插入、删除等操作

```
        This is a long line 
        short 
        Any other long line 
```

### 11.5.2 平移

">" 命令把选中的文档向右移动一个 "平移单位"，中间用空白填充。平移的起始点是可视列块的左边界。还是用上面的例子，">" 命令会导致如下结果:

```
        This is a         long line 
        short 
        Any other         long line 
```

平移的距离由 'shiftwidth' 选项定义。例如，要每次平移 4 个空格，可以用这个命令:

```
        :set shiftwidth=4
```

## 11.6 读写文件的一部分

### 11.6.1 读取文件

| 序号 | 命令 | 功能 |
|:---:|---|---|
| 1 | `:read {filename}` | 将文件插入到光标后面|
| 2 | `:$read {file}` | 将文件插入本文的最后 |
| 3 | `:0read {file}`| 将文件插入本文开头 |
| 4 | `:read !ls` | 把 ls 的结果插入本文 |
| 5 | `:0read !date -u` | 这将用 UTC 格式把当前的时间插入到文件开头 |


### 11.6.1 保存部分内容

| 序号 | 命令 | 功能 |
|:---:|---|---|
| 1 | `:.,$write tempo` | 写入当前位置到文件末的全部行到文件 "tempo" 中 |
| 2 | `:.write collection` | 把当前行写入文件 collection |
| 3 | `:.write >>collection` | ">>" 通知 Vim 把内容添加到文件 "collection" 的后面 |
| 4 | `:write !wc` | 把文本写入到命令，wc是字符统计的程序 |

### 11.6.2 排版

| 序号 | 命令 | 功能 |
|:---:|---|---|
| 1 | `:.,$write tempo` | 写入当前位置到文件末的全部行到文件 "tempo" 中 |
| 2 | `:set textwidth=72` | 自动换行 |
| 3 | `gqap` | 排版段落对象，`gq`是排版指令 |

> "一段" 与下一段的分割符是一个空行  
> 只包括空白字符的空白行不能分割 "一段"。这很不容易分辨

### 11.6.3 使用外部程序

使用外部程序的格式为：`!{motion}{program}`，比如 `!5Gsort<Enter>`，其中 `!` 告诉vim要执行过滤操作，然后 Vim 编辑器等待一个 "动作" 命令来告诉 它要过滤哪部分文本。"5G" 命令告诉 Vim 移到第 5 行。于是，Vim 知道要处理的是第 1 行 (当前行) 到第 5 行间的内容。


# 12. 崩溃与恢复

对于一个异常关闭的文件，可以用下面的方式恢复：

```shell
# 将当前打开的内容写入recovered备份文件中
:write help.txt.recovered
# 编辑这个文件
:edit #
# diff split help.txt 与help.txt文件比较
:diffsp help.txt
```

vim会自动打开一个交换文件，可以通过 `vim -r` 查询交换文件所在的目录

# 13. 小窍门

## 13.1 替换

### 13.1.1 在单个文件中的替换

| 序号 | 命令 | 功能 |
|:---:|---|---|
| 1 | `:%s/four/4/g` | 在全文中用一个单词替换另一个单词 |
| 2 | `:%s/\\<four/4/g` | 指定匹配单词开头 |

### 13.1.2 在多个文件中

```shell
vim *.cpp     #  启动 Vim，用当前目录的所有 C++ 文件作为文件参数。
              #  启动后你会停在第一个文件上。
qq            #  用 q 作为寄存器启动一次记录。
:%s/\<GetResp\>/GetAnswer/g  #  在第一个文件中执行替换。
:wnext        #  保存文件并移到下一个文件。
q             #  中止记录。
@q            #  回放 q 中的记录。这会执行又一次替换和":wnext"。
              #  你现在可以检查一下记录有没有错。
999@q         #  对剩下的文件执行 q 中的命令
```

这里有一个陷阱: 如果有一个文件不包含 "GetResp"，Vim 会报错，而整个过程会中止，要避免这个问题，可以在替换命令后面加一个标记:

```shell
        :%s/\<GetResp\>/GetAnswer/ge
```

> `%`范围前缀表示在所有行中执行替换

"e" 标记通知 ":substitute" 命令找不到不是错误

### 13.2 排序

```makefile
        OBJS = \ 
                version.o \ 
                pch.o \ 
                getopt.o \ 
                util.o \ 
                getopt1.o \ 
                inp.o \ 
                patch.o \ 
                backup.o 
```

通过命令：

```shell
        /^OBJS   # 光标移到开头
        j        # 下一行
        :.,/^$/-1!sort
```

这会先移到 "OBJS" 开头的行，向下移动一行，然后一行行执行过滤，直到遇到一个空行。你也可以先选中所有需要排序的行，然后执行 "!sort"。那更容易一些，但如果有很多行就比较麻烦。 

> 这里的 `.,/^$/-1` 是指从当前行到一个空白行 `/^$`(正则式匹配行首尾) 的前一行 `-1`

# 14 命令行的命令

命令行的简写可以在[这里](http://yianwillis.github.io/vimcdoc/doc/quickref.html#option-list)查到

# 15 在代码间移动

## 15.1 标签

标签就是一个标识符被定义的地方，标签列表保存在一个标签文件中。`ctags *.c` 为当前目录下的所有c文件建立标签。然后就可以用`:tag function`跳转到一个函数定义的地方。`CTRL-]` 命令会跳转到当前光标下单词的标签。几次跳转后，可以用 `:tags`显示你经过的标签列表`CTRL-T` 命令跳转到上一个标签，而`:tag`可以跳转到标签列表的最上面一个，你可以在前面加上要向前跳转的标签个数。比如: ":3tag"。

## 15.2 分割窗口

`:stag tagname` 使用 ":split" 命令将窗口分开然后再用 ":tag" 命令。

`CTRL-W ]`分割当前窗口并跳转到光标下的标签

## 15.3 多个标签文件 

多个目录则在每一个目录下创建一个标签文件，但Vim 只能跳转到那个目录下的标签。通过设定 'tags' 选项，你可以使用多个相关的标签文件：

```shell
:set tags=./tags,./../tags,./*/tags
```

让 Vim 找到当前文件所在目录及其父目录和所有一级子目录下的标签文件

```shell
:set tags=~/proj/**/tags
```

这个指令可以查找整个目录树下标签文件

## 15.4 预览窗口

`:ptag write_char`：打开一个预览窗口来显示函数 "write_char"

`CTRL-W }`：预览窗口中得到光标下函数的定义

`:pclose`：关闭预览窗口

`:pedit defs.h`：预览窗口中编辑一个指定的文件

# 16 配置

设置属性用 `set`，比如 `set number` 是让vim显示行号，`set nonumber`关闭行号
映射是`map`和`noremap`(非递归映射)
`let mapleader=" "`，让vim的leader变为空格，
`noremap \<LEADER\>\<CR\> :nohlsearch`：让空格+回国取消搜索高亮，vim的leader默认为反斜杠
`:colorscheme \<tab\>`：选择主题  


| 序号 | 属性项 | 功能 |
|:---:|---|---|
| 1 | relativenumber | 相对行号 |
| 2 | number | 行号 |
| 3 | cursorline | 鼠标线 |
| 4 | wrap | 自动换行 |
| 5 | wildmenu | 输入命令时，`tab`显示候选菜单 |
| 6 | syntax | 语法高亮 |
| 7 | hlsearch | 搜索高亮 |
| 8 | incsearch |  一面输入一面高亮 |
| 9 | ignorecase | 忽略大小写 |
| 10 | smartcase | 智能大小写 |
