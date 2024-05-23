# 在Mind+中使用XEduHub

Mind+中也上线了XEduHub积木块，使用积木也可以玩XEduHub。使用Mind+V1.7.2及以上版本，在python模式用户库中加载此扩展。

Gitee链接：[https://gitee.com/liliang9693/ext-xedu-hub](https://gitee.com/liliang9693/ext-xedu-hub)

## 使用说明

### 第一步：加载积木库

- 如果联网情况下，打开Mind+用户库粘贴本仓库的链接即可加载：

![](../images/xeduhub/mind1.png)

- 如果电脑未联网，则可以下载本仓库的文件，然后打开Mind+用户库选择导入用户库，选择`.mpext`文件即可。

### 第二步：安装python库

打开库管理，输入xedu-python运行，提示successfully即可。

注：WARNING是提醒，可以忽略；请及时更新xedu-python用户库，以获得更稳定、更强大的模型部署使用体验。

![](../images/xeduhub/mind2.png)

### 第三步：开始编程！

至此，即可拖动积木块开始快乐编程啦，根据任务类别，可以分为两类：预置任务和通用任务。其中，预置任务指各种内置模型的常见任务，通用任务包含“XEdu”的MMEdu、BaseNN和BaseML等各种工具训练的模型。通用任务也支持其他的ONNX，但前提是需要知道输入的数据格式，需要做前处理。在Mind+中编写预制任务的程序非常简单，例如pose_body任务的运行示例如下：

![](../images/xeduhub/mind3.png)

大家可以举一反三尝试编写各种预制任务的代码，针对MMEdu、BaseNN、BaseML等工具训练及转换并导出的模型，额外将模型文件上传再指定即可。

用这套积木块基本可以完成XEduHub的所有任务，可以做各种小任务，也可以做复杂任务。使用积木完成对一张图片借助XEduHub的相关模型进行人体画面提取、关键点识别，再用BaseNN训练并转换的ONNX模型完成分类模型推理的示例如下。

![](../images/xeduhub/mind4.png)