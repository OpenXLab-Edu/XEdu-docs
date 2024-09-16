# 如何快速获得XEdu



最便捷使用XEdu的方法是使用浦育平台。浦育平台为使用者提供了专属的“XEdu容器”，你可以把“XEdu容器”想象成一台部署好XEdu和配套AI开发工具的远程电脑。要知道，很多初学者就因为环境搭建的问题，而始终被挡在AI世界之外。除了浦育平台，浙大的Mo平台、OpenHydra云平台也提供支持XEdu的容器，相信未来会有更多的云算力供应商会提供内置XEdu的容器供我们选择。

XEdu当然可以在本地运行。为了满足不同用户的需求，XEdu安装方式分为一键安装包安装、pip安装、Docker安装和Openhydra安装等。对于中小学的初学者，强烈推荐选择XEdu一键安装包（CPU版本），它在绝大多数的机房都能正常运行，包括信创电脑。

## 在浦育平台使用XEdu

### 1. 打开浦育平台，注册登录账号

[浦育平台openinnolab
](https://www.openinnolab.org.cn/pjedu/home
)：[https://www.openinnolab.org.cn/pjedu/home
](https://www.openinnolab.org.cn/pjedu/home
)

### 2. 克隆一个XEdu容器的项目

建议初学者从克隆他人的项目开始学习XEdu，这样省去选择“工具类型”的麻烦。克隆项目的步骤如下：

“项目”->"搜索XEdu"->"挑选喜欢的项目"->"克隆"

项目中出现“XEdu”“MMEdu”“BaseNN”等名词的，大都属于“XEdu”项目，使用的是配置好XEdu环境的容器。当然，在XEdu文档中出现的浦育平台项目，也肯定都使用了XEdu环境。

![](../images/how_to_quick_start/openinnolab0.jpg)

这样你就得到了一个XEdu项目，跟随项目就可以快速入门XEdu了。

![](../images/how_to_quick_start/openinnolab1.jpg)

### 3. 新建一个XEdu容器的项目

在浦育平台新建项目时，会有多个工具类型供你选择，请初学者使用“Notebook编程”中的“XEdu”。新建项目的步骤如下：

“工具”->"人工智能工坊"->"Notebook编程"->"XEdu"

![](../images/how_to_quick_start/openinnolab2.jpg)

## 在本地使用XEdu

### 准备工作

下载工具：XEdu一键安装包

下载方式

飞书网盘：[https://p6bm2if73b.feishu.cn/file/boxcn7ejYk2XUDsHI3Miq9546Uf?from=from_copylink](https://p6bm2if73b.feishu.cn/file/boxcn7ejYk2XUDsHI3Miq9546Uf?from=from_copylink)

硬件要求：准备win10及以上的电脑，将一键安装包安装到纯英文路径下。飞书网盘里的一件安装包会不定期更新，可时常到网盘中查看与下载最新版。

### 安装步骤

第一步：双击运行“XEdu v1.6.7d.exe”文件，将自解压为XEdu文件夹。

![](../images/about/XEDUinstall1.png)

第二步：打开XEdu简介.pdf，快速了解一键安装包的使用。

第三步：快速测试XEdu示例代码。

打开根目录下的"jupyter编辑器.bat"，即自动启动浏览器并显示界面，如下图所示。

![](../images/about/XEDUinstall3.png)

此时可打开"demo"文件夹中的ipynb文件，如"MMEdu\_cls\_notebook.ipynb"。选中代码单元格，点击常用工具栏"运行"按钮，就可以运行单元格中的代码，单元格左侧\[\*\]内的星号变为数字，表示该单元格运行完成。按步骤即可测试体验XEdu代码。

![](../images/about/jupyter1.png)

更多一键安装包的使用请前往[XEdu一键安装包说明](https://xedu.readthedocs.io/zh-cn/master/about/installation.html#id2)。