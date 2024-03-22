# BaseNN安装或下载

你可以通过pip命令来安装BaseNN。

`pip install basenn` 或 `pip install BaseNN`

更新库文件：

`pip install --upgrade BaseNN`

可以在命令行输入BaseNN查看安装的路径，在安装路径内，可以查看提供的更多demo案例。

BaseNN已经内置在XEdu的一键安装包中，解压后即可使用。

如果在使用中出现类似报错：`**AttributeError**: partially initialized module 'cv2' has no attribute 'gapi_wip_gst_GStreamerPipeline' (most likely due to a circular import)` 

可尝试通过运行`pip install --upgrade opencv-python`解决。

库文件源代码可以从[PyPi](https://pypi.org/project/BaseNN/#files)下载，选择tar.gz格式下载，可用常见解压软件查看源码。
