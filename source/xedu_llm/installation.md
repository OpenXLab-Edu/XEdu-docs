# XEduLLM安装

### 安装及更新方法

你可以通过pip命令来安装XEduLLM。

`pip install xedu-python` 或 `pip install XEdu-python`

更新库文件：

```python
pip install --upgrade XEdu-python
```

XEdu-python已经内置在XEdu的一键安装包中，解压后即可使用。

### 安装常见问题解答

1.系统原因

强烈建议win10及以上，win7不推荐。

2.依赖库问题

常见出问题的是onnxruntime库，建议手动pip安装，再进行库排查。

### XEduLLM安装包版

为了大大降低老师们的使用成本，XEdu团队上线了XEduLLM安装包版，无需编写过多代码，即可完成基本使用。

Windows安装包：[https://p6bm2if73b.feishu.cn/file/PTB8binCMozK1Rx8iPQcnVrynWd](https://p6bm2if73b.feishu.cn/file/PTB8binCMozK1Rx8iPQcnVrynWd)

Mac/源码版本：[https://p6bm2if73b.feishu.cn/file/CMOqbJzKeo7lgRxmqufcugoEnCe](https://p6bm2if73b.feishu.cn/file/CMOqbJzKeo7lgRxmqufcugoEnCe)

命令行启动版本（Mac/Linux）：`bash <(curl https://cdn.openinnolab.org.cn/res/api/v1/file/creator/b5121868-53ab-4ea8-b75f-8458b768b9e4&name=xedu)`

- 注：如果运行源码，需要提前在本地python中安装依赖库：`xedu-python gradio requests`，并将对应python路径配置进环境变量，调试过程中的错误将输出到`log.txt`。

- [源码开源仓库](https://github.com/EasonQYS/XEduLLM-tools)，欢迎提交PR增加新功能。

**视频版使用指引**

（点击视频进入跳转观看）

<iframe src="//player.bilibili.com/player.html?isOutside=true&aid=112994499431048&bvid=BV12tWsePEmA&cid=500001656107517&p=1" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"></iframe>
