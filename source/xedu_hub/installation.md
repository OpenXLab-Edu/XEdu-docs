# XEduHub安装

### 安装及更新方法

你可以通过pip命令来安装XEduHub。

`pip install xedu-python` 或 `pip install XEdu-python`

更新库文件：

```python
pip install --upgrade XEdu-python
```

XEdu-python已经内置在XEdu的一键安装包中，解压后即可使用。

### 安装常见问题解答

1.系统原因

建议win10及以上。

2.依赖库问题

常见出问题的是onnxruntime，建议手动pip安装，再进行库排查。

最简测试代码：

```python
# 导入 onnxruntime 库
import onnxruntime

# 打印版本信息
print(f"onnxruntime version: {onnxruntime.__version__}")
```

如果安装正确，您将看到类似以下的输出：

```
onnxruntime version: 1.9.0
```

