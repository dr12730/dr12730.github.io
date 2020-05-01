# SD卡启动详解

## 主流的外存设备介绍

内存和外存的区别：一般是把这种RAM(random access memory,随机访问存储器，特点是任意字节读写，掉电丢失)叫内存，把ROM（read only memory，只读存储器，类似于Flash SD卡之类的，用来存储东西，掉电不丢失，不能随机地址访问，只能以块为单位来访问）叫外存

### 软盘、硬盘、光盘、CD、磁带

- 存储原理大部分为**磁**存储
  - 缺点是读写速度、可靠性等。
  - 优点是技术成熟、价格便宜。广泛使用在桌面电脑中，在嵌入式设备中几乎无使用。
- 现代存储的发展方向是Flash存储，闪存技术是利用**电学原理**来存储1和0，从而制成存储设备。所以闪存设备没有物理运动（硬盘中的磁头），所以读写速度可以很快，且无物理损耗。

### 纯粹的Flash：NandFlash、NorFlash

最早出现的、最原始的Flash颗粒组成芯片。也就是说NandFlash、NorFlash芯片中只是对存储单元做了最基本的读写接口，然后要求外部的SoC来提供Flash读写的控制器以和Flash进行读写时序。

- 缺陷：
  - 读写接口时序比较复杂。
  - 内部无坏块处理机制，需要SoC自己来管理Flash的坏块；
  - 各家厂家的Flash接口不一致，甚至同一个厂家的不同型号、系列的Flash接口都不一致，这就造成产品升级时很麻烦。

NandFlash分MLC和SLC两种。SLC技术比较早，可靠性高，缺点是容量做不大（或者说容量大了太贵，一般SLC Nand都是512MB以下）；MLC技术比较新，不成熟，可靠性差，优点是容量可以做很大很便宜，现在基本都在发展MLC技术。

### SD卡、MMC卡、MicroSD、TF卡

- 这些卡其实内部就是Flash存储颗粒，比直接的Nand芯片多了**统一的外部封装和接口**
- 卡都有**统一的标准**，譬如SD卡都是遵照SD规范来发布的。这些规范规定了SD卡的读写速度、读写接口时序、读写命令集、卡大小尺寸、引脚个数及定义。这样做的好处就是不同厂家的SD卡可以**通用**。

### iNand、MoviNand、eSSD

- 电子产品如手机、相机等，前些年趋势是用SD卡/TF卡等扩展存储容量；但是近年来的趋势是直接内置大容量Flash芯片而不是外部扩展卡。外部扩展卡时间长了卡槽可能会接触不良导致不可靠。
- 现在主流的发展方向是使用iNand、MoviNand、eSSD（还有别的一些名字）来做电子产品的存储芯片。这些东西的**本质还是NandFlash**，内部由Nand的存储颗粒构成，再集成了块设备管理单元，综合了SD卡为代表的各种卡的优势和原始的NandFlash芯片的优势。
- 优势：
  - 向SD卡学习，有**统一的接口标准**（包括引脚定义、物理封装、接口时序）。
  - 向原始的Nand学习，以**芯片的方式来发布**而不是以卡的方式；
  - 内部内置了Flash**管理模块**，提供了诸如**坏块管理**等功能，让Nand的管理容易了起来。

## SD卡的特点和背景知识

### SD卡和MMC卡的关系

- **MMC**标准比SD标准**早**，SD标准兼容MMC标准。
- MMC卡可以被SD读卡器读写，而SD卡不可以被MMC读卡器读写。

### SD卡和Nand、Nor等Flash芯片差异

SD卡/MMC卡等卡类有**统一**的接口标准，而Nand芯片没有统一的标准（各家产品会有差异）

### SD卡与MicroSD的区别

体积大小区别而已，传输与原理完全相同，

## SD卡的编程接口

### SD卡的物理接口

![sd](./images/sd.jgp)

SD卡由9个针脚与外界进行物理连接，这9个脚中有2个地，1个电源，6个信号线。

![microsd](./images/microsd.jgp)

微型SD卡（Micro SD）也称为TF卡，它的体积比普通SD卡小很多，并且只有8个引脚

### SD协议与SPI协议

SD卡与SRAM/DDR/SROM之类的东西的不同：

- SRAM/DDR/SROM之类的存储芯片是总线式的，只要连接上初始化好之后就可以由SoC直接以地址方式来访问；
- 但是SD卡不能直接通过接口给地址来访问，它的访问需要按照一定的接口协议（时序）来访问。

SD卡虽然只有一种物理接口，但是却支持两种读写协议：SD协议和SPI协议。

### SPI协议特点（低速、接口操作时序简单、适合单片机）

- SPI协议是**单片机**中广泛使用的一种通信协议，并不是为SD卡专门发明的。
- SPI协议相对SD协议来说**速度比较低**
- SD卡支持SPI协议，就是为了单片机方便使用

### SD协议特点（高速、接口时序复杂，适合有SDIO接口的SoC）

- SD协议是专门用来和SD卡通信的
- SD协议要求SoC中有**SD控制器**，运行在高速率下，要求SoC的主频不能太低

### S5PV210的SD/MMC控制器

(1) 数据手册Section8.7，为SD/MMC控制器介绍。

(2) SD卡内部除了存储单元Flash外，还有SD卡管理模块，我们SoC和SD卡通信时，通过9针引脚以SD协议/SPI协议向SD卡管理模块发送命令、时钟、数据等信息，然后从SD卡返回信息给SoC来交互。工作时每一个任务（譬如初始化SD卡、譬如读一个块、譬如写、譬如擦除····）都需要一定的时序来完成（所谓时序就是先向SD卡发送xx命令，SD卡回xx消息，然后再向SD卡发送xx命令····）

## S5PV210的SD卡启动详解

### SD卡启动的好处

- 可以在不借用专用烧录工具（类似Jlink）的情况下对SD卡进行刷机，然后刷机后的SD卡插入卡槽，SoC既可启动；
- 可以用SD卡启动进行量产刷机（量产卡）。像我们X210开发板，板子贴片好的时候，内部iNand是空的，此时直接启动无启动；板子出厂前官方刷机时是把事先做好的量产卡插入SD卡卡槽，然后打到iNand方式启动；因为此时iNand是空的所以第一启动失败，会转而第二启动，就从外部SD2通道的SD卡启动了。启动后会执行刷机操作对iNand进行刷机，刷机完成后自动重启（这回重启时iNand中已经有image了，所以可以启动了）。刷机完成后SD量产卡拔掉，烧机48小时，无死机即可装箱待发货。

### SD卡启动的难点（SRAM、DDR、SDCard）

- SRAM、DDR都是**总线式**访问的，SRAM不需初始化既可直接使用，而DDR需要初始化后才能使用，但是总之CPU可以直接和SRAM/DRAM打交道；
- 而**SD卡**需要时序访问，CPU不能直接和SD卡打交道；
- NorFlash读取时可以总线式访问，所以Norflash启动非常简单，可以直接启动，但是SD/NandFlash不行。

以前只有Norflash可以作为启动介质，台式机笔记本的BIOS就是Norflash做的。后来三星在2440中使用了SteppingStone的技术，让Nandflash也可以作为启动介质。SteppingStone（翻译为启动基石）技术就是在SoC内部内置4KB的SRAM，然后开机时SoC根据OMpin判断用户设置的启动方式，如果是NandFlash启动，则SoC的启动部分的硬件直接从外部NandFlash中读取开头的4KB到内部SRAM作为启动内容。

启动基石技术进一步发展，在6410芯片中得到完善，在210芯片时已经完全成熟。210中有96KB的SRAM，并且有一段iROM代码作为BL0，BL0再去启动BL1（210中的BL0做的事情在2440中也有，只不过那时候是硬件自动完成的，而且体系没有210中这么详细）。

### S5PV210的启动过程回顾

- 210启动首先执行内部的iROM（也就是BL0），BL0会判断OMpin来决定从哪个设备启动，
- 如果启动设备是SD卡，则BL0会从SD卡读取前16KB（不一定是16，反正16是工作的）到SRAM中去启动执行（这部分就是BL1，这就是steppingstone技术）
- BL1执行之后剩下的就是软件的事情了

### SD卡启动流程

- 整个镜像大小小于16KB
  - 整个镜像作为BL1被steppingstone直接硬件加载执行
- 整个镜像大小大于16KB
  - 把整个镜像分为2部分
  - 第一部分16KB大小，第二部分是剩下的大小
  - 第一部分作为BL1启动。去初始化DRAM并且将第二部分加载到DRAM中去执行

### iROM究竟是怎样读取SD卡/NandFlash的

三星在iROM中事先内置了一些代码去初始化外部SD卡/NandFlash，并且内置了读取各种SD卡/NandFlash的代码在iROM中。BL0执行时就是通过调用这些device copy function来读取外部SD卡/NandFlash中的BL1的。

> 《iROM application note》：
>
> S5PV210内部有一个ROM代码的块复制功能，用于boot-u设备。因此，开发人员可以不需要实现设备复制功能。这些内部功能可以将存储设备中的任何数据复制到SDRAM中。用户可以在完全结束内部ROM启动过程后使用这些功能。

| 地址       | 名称            | 功能                                                         |
| ---------- | --------------- | ------------------------------------------------------------ |
| 0xD0037F98 | CopySDMMCtoMem  | 该内部功能可以将SD/MMC设备中的任何数据复制到SDRAM中。用户可以在IROM启动过程完全结束后使用该功能。 |
| 0xD0037F9C | CopyMMC4_3toMem | 该内部功能可以将eMMC设备中的任何数据复制到SDRAM中。用户可以在IROM启动过程完全结束后使用该功能。 |

下面这个函数工作在 EPLL 时钟源 20MHz 频率下	

```c
/**
* This Function copy MMC(MoviNAND/iNand) Card Data to memory.
* Always use EPLL source clock.
* This function works at 20Mhz.
* @param int blockNum：SD卡在 SoC 上的编号，通道号
* @param u32 StartBlkAddress : Source card(MoviNAND/iNand MMC)) Address.(It must block address.) SD卡的起始块地址
* @param u16 blockSize : Number of blocks to copy. 要拷贝的块数
* @param u32* memoryPtr : Buffer to copy from. 拷贝到DDR的地址
* @param bool with_init : determined card initialization. 是否初始化SD卡
* @return bool(u8) - Success or failure.
*/
#define CopySDMMCtoMem(z,a,b,c,e)(((bool(*)(int, unsigned int, unsigned short, unsigned int*, bool))(*((unsigned int *)0xD0037F98)))(z,a,b,c,e))

typedef bool (*tFunc_SDMMCtoMem)(int, unsigned int, unsigned short, unsigned int*, bool)

tFunc_SDMMCtoMem pFunc = (tFunc_SDMMCtoMem)(*(unsigned int *)0xD0037F98);
```

### 扇区和块的概念

早期的**块设备**就是**软盘硬盘**这类磁存储设备，这种设备的存储单元不是以字节为单位，而是以**扇区**为单位。磁存储设备**读写的最小单元**就是扇区，不能只读取或写部分扇区。这个限制是磁存储设备本身物理方面的原因造成的，也成为了我们编程时必须遵守的规律。

一个扇区有多个字节（一般是**512字节**）。早期的磁盘扇区是512字节，实际上后来的磁盘扇区可以做的比较大（譬如1024字节，譬如2048字节，譬如4096字节），但是因为原来最早是512字节，很多的软件（包括操作系统和文件系统）已经默认了512这个数字，因此后来的硬件虽然物理上可能支持更大的扇区，但是实际上一般还是兼容512字节扇区这种操作方法。

一个扇区可以看成是一个块block（块的概念就是：不是一个字节，是多个字节组成一个共同的操作单元块），所以就把这一类的设备称为**块设备**。常见的块设备有：磁存储设备硬盘、软盘、DVD和Flash设备（U盘、SSD、SD卡、NandFlash、Norflash、eMMC、iNand）

linux里有个mtd驱动，就是用来管理这类块设备的。

磁盘和Flash以块为单位来读写，就决定了我们启动时device copy function只能以整块为单位来读取SD卡。

## S5PV210的SD卡启动实战

### 总体思路

- 将我们的代码分为2部分：
  - 第一部分BL1小于等于16KB，第二部分为任意大小
- iROM代码执行完成后从SD卡启动会自动读取BL1到SRAM中执行；
- BL1执行时负责初始化DDR，然后手动将BL2从SD卡copy到DDR中正确位置，
- 然后BL1远跳转到BL2中执行BL2.

### 程序安排

- 文件夹 BL1完成

  - 关看门狗、设置栈、开iCache、初始化DDR、
  - 从SD卡复制BL2到DDR中特定位置，跳转执行BL2

- BL1在SD卡中必须从Block1开始（Block0不能用，这个是三星官方规定的），长度为16KB内，我们就定为16KB（也就是32个block）。三星的 《iRom Application Note》中规定，说明 iROM 中固化的拷贝程序固定是从这个地址开始拷贝的

  ![block](./images/block.jpg)

- BL1理论上可以从33扇区开始，但是实际上为了安全都会留一些空扇区作为隔离，譬如可以从45扇区开始，长度由自己定（实际根据自己的BL2大小来分配长度，我们实验时BL2非常小，因此我们定义BL2长度为16KB，也就是32扇区）。

- DDR初始化好之后，整个DDR都可以使用了，这时在其中选择一段长度足够BL2的DDR空间即可。我们选0x23E00000（uboot 用了这个地址，而且因为我们BL1中只初始化了DDR1，地址空间范围是0x20000000～0x2FFFFFFF）。

### BL2远跳转

因为我们BL1和BL2其实是2个独立的程序，链接时也是独立分开链接的，所以不能像以前一样使用 `ldr pc, =main` 这种方式来通过链接地址实现元跳转到BL2.

我们的解决方案是使用地址进行强制跳转。因为我们知道BL2在内存地址 `0x23E00000` 处，所以直接去执行这个地址即可。

### 代码分为2部分启动（上一节讲的）的缺陷

代码分为2部分，这种技术叫分散加载。这种分散加载的方法可以解决问题，但是比较麻烦。

分散加载的缺陷：

- 第一，代码完全分2部分，完全独立，代码编写和组织上麻烦；
- 第二，无法让工程项目兼容SD卡启动和Nand启动、NorFlash启动等各种启动方式。

### uboot中的做法

程序代码仍然包括BL1和BL2两部分，但是组织形式上不分为2部分而是作为一个整体来组织。它的实现方式是：

- iROM启动然后从SD卡的扇区1开始读取16KB的BL1然后去执行BL1，BL1负责初始化DDR，
- 然后从SD卡中读取**整个程序**（BL1+BL2）到DDR中，
- 然后从DDR中执行（利用 `ldr pc, =main` 这种方式以远跳转从SRAM中运行的BL1跳转到DDR中运行的BL2）

### uboot的SD卡启动细节

- uboot编译好之后有200多KB，超出了16KB。uboot的组织方式就是前面16KB为BL1，剩下的部分为BL2.
- uboot在烧录到SD卡的时候，先截取uboot.bin的前16KB（实际脚本截取的是8KB）烧录到SD卡的block1～bolck32；然后将整个uboot烧录到SD卡的某个扇区中（譬如49扇区）
- 实际uboot从SD卡启动
  - iROM先执行，根据OMpin判断出启动设备是SD卡
  - 从SD卡的block1开始读取16KB（8KB）到SRAM中执行BL1
  - BL1执行时负责初始化DDR，并且从SD卡的49扇区开始复制整个uboot到DDR中指定位置（0x23E00000）去备用
  - 然后BL1继续执行直到 `ldr pc, =main` 时BL1跳转到DDR上的BL2中接着执行uboot的第二阶段
- uboot中的这种启动方式比上节讲的分散加载的好处在于：能够兼容各种启动方式

### 为什么uboot方式可以直接跳转

分散加载的 BL1 和 BL2 相当于两个独立的程序。好比你有一个程序 BL1，从别人那里拿了一个程序 BL2，你自然不知道 BL2 中的链接地址是怎么分配的，只能听别人告诉你的跳转到 0x23E00000 去执行他的代码。如果他修改了他的链接地址，你也必须相应修改这个 0x23E00000，不然没法执行他的程序。

而 uboot 的方法，相当于只有一个程序 BL，程序自身当然知道内部各个符号的链接地址。与分散加载不同的是，你把这个程序 BL 的前 16KB 放到 SRAM 中执行，这个 BL 会把自己拷贝到一个新地址（链接地址）处，然后 `ldr pc, =main` 就可以跳转到 DDR 的 main 地址处执行。因为链接地址在 PC 端编译时在 `link.lds` 中就已经指定了，编译完后就固定下来。这也要求在 DDR 中放置的地址（运行地址）和编译完后的链接地址一致，否则也不能正确执行。

## 解决X210开发板的软开关按键问题

### X210开发板的软启动电路详解

![pwrkey](./images/pwrkey.jpg)

- 210供电需要的电压比较稳定，而外部适配器的输出电压不一定那么稳定，因此板载了一个文稳压器件MP1482.这个稳压芯片的作用就是外部适配器电压在一定范围内变化时稳压芯片的输出电压都是5V。
- MP1482芯片有一个EN（Enable）引脚，这个引脚可以让稳压芯片输出或关闭输出。EN为高电平时有输出电压，EN引脚为低电平时稳压芯片无输出
- 两个因素可以影响EN引脚的电平
  - 第一个是POWER按键（SW1），POWER按键按下时EN为高电平，POWER按键弹起时EN为低电平
  - 第二个是POWER_LOCK（EINT0）引脚，这个引脚为POWER_LOCK模式下高电平，则EN为高；若这个引脚为EINT0模式或者为POWER_LOCK模式但输出为低电平，则EN为低
- 图中还有EINT1引脚（输出），这个引脚的作用是用来做中断，提供给CPU用来唤醒的

### 为什么要软启动

一般的电路设计都是用拨码开关来做电源开关的（打到一侧则接通，打到另一侧则关闭）。这种方式的优点是设计简单，缺点是电路太简单，整个主板要么有电要么没电无法做休眠模式、低功耗模式等。软启动电路是比较接近于实际产品的，其他开发板的硬开关其实是简化版的，和实际产品还有差异。

### 开发板供电置锁原理和分析

- 软开关在设计时有一个**置锁电路**，用EINT0（也就是GPH0_2）引脚来控制的

- EINT0这个引脚是有复用设计（两个完全不相干的功能挤在同一个引脚上，同时我们只能让这个引脚用于其中一种功能，这就叫复用）的，一个是GPIO（也就是GPH0_2引脚）、一个是PS_HOLD_CONTROL。（注意：EINT0功能算是GPIO下的一个子功能）

- PS_HOLD在Section2.4 Power Management章节下的4.10.5.8节下

  PS_HOLD_CONTROL: `0xE010_E81C`

  | PS_HOLD_CONTROL | Bit     | 描述                                                         | 初始值 |
  | --------------- | ------- | ------------------------------------------------------------ | ------ |
  | 保留            | [31:12] | 保留                                                         | 0x5    |
  | 保留            | [11:10] | 保留                                                         | 0x0    |
  | DIR             | [9]     | 方向(Direction) 0-输入；1-输出                               | 1      |
  | DATA            | [8]     | 驱动值（0-低；1-高）                                         | 0      |
  | 保留            | [7:1]   | 保留                                                         | 0      |
  | PS_HOLD_OUT_EN  | [0]     | 焊盘 XEINT[0] 由该寄存器值控制，当该字段为'1'时，GPIO 章的 XEINT[0]控制寄存器的值被忽略。(0：禁用，1：启用) | 0      |

  PS_HOLD（复用XEINT[0]）引脚的值在任何电源模式下都会保持不变。该寄存器处于存活区域，仅通过XnRESET或断电复位。

  - bit0, 0表示这个引脚为GPIO功能，1表示这个引脚为PS_HOLD功能
  - bit9，0表示这个引脚方向为输入，1表示这个引脚方向为输出
  - bit8，0表示这个引脚输出为低电平，1表示输出为高电平
  - 软启动置锁，则需要将bit0、8、9都置为1即可

- 开发板置锁后，POWER按键已经失效，关机时需要按下复位按键

## Linux 的 dd 命令

**dd**是一个[Unix](https://baike.baidu.com/item/Unix)和[类Unix](https://baike.baidu.com/item/类Unix)[系统](https://baike.baidu.com/item/系统)上的命令，主要功能为转换和复制文件。

参数说明：

- if=文件名：输入文件名，默认为标准输入。即指定源文件。
- of=文件名：输出文件名，默认为标准输出。即指定目的文件。
- ibs=bytes：一次读入bytes个字节，即指定一个块大小为bytes个字节。
- obs=bytes：一次输出bytes个字节，即指定一个块大小为bytes个字节。
- bs=bytes：同时设置读入/输出的块大小为bytes个字节。
- cbs=bytes：一次**转换**bytes个字节，即指定转换缓冲区大小。
- skip=blocks：从**输入文件**开头跳过blocks个块后再开始复制。
- seek=blocks：从**输出文件**开头跳过blocks个块后再开始复制。
- count=blocks：仅拷贝blocks个块，块大小等于ibs指定的字节数。
- flag参数说明：
  -  append      追加模式(仅对输出有意义；隐含了conv=notrunc)
  -  direct         使用直接I/O 存取模式
  - directory    除非是目录，否则 directory 失败
  - dsync         使用同步I/O 存取模式
  - sync           与上者类似，但同时也对元数据生效

1. 将 BL1 拷贝 16 kB 到 SD 卡起始地址为 Block1

   - 16kB = 16384 Byte = 32 x 512 Byte。所以从 Block1 到 Block32 共 32 个 block

   - linux 命令：

     ```bash
     dd iflag=dsync oflag=dsync if=./BL1/BL1.bin of=/dev/sdb seek=1
     ```

2. 将 BL2 拷贝到 SD 卡的 Block45 

   ```bash
   dd iflag=dsync oflay=dsync if=./BL2/BL2.bin of/dev/sdb seek=45
   ```

   