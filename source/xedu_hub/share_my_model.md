# 生态共建：分享我的模型方案
相信你已经领略到XEduHub的统一代码格式带来的便利了。但是如果我们有自己发现的好玩的模型，能不能也运行在XEduHub之上呢？答案是“必须可以”！

你可以在[魔搭社区](https://modelscope.cn/home)等平台上按照格式上传模型，然后只需要向好友分享这个名称即可在XEduHub工具中载入模型和相应的处理流程。

但是这样只有知道这个名称的好友能用到这个模型，为了让更多人知道你贡献的好玩的模型，我们诚邀您在上传模型之后，在这里填写[问卷](https://aicarrier.feishu.cn/share/base/form/shrcnZqkQ3n3EVkE9vEMhkMYAsf)登记您的模型，这样，就可以让您的模型加入[生态共建模型仓库](https://aicarrier.feishu.cn/wiki/VdWkwyNvtiUyAlkrOkEcZgBznOb?from=from_copylink)，让更多人看到并用上您的模型！

## 体验好友的模型
我们的生态伙伴上传了一个模型到[魔搭社区](https://modelscope.cn/models/fhl123/mobileone_test)，那么我们就可以使用这个模型啦！模型仓库名称为`fhl123/mobileone_test`，这个名称作为`repo`的值传入，创建一个模型，然后输入的数据为图片路径，最后我们就可以看到结果输出啦！
```python
from XEdu.hub import Workflow as wf
img_path = 'ele.jpg'
model = wf(repo='fhl123/mobileone_test')
result = model.inference(data=img_path)
print(result)
```
### 原理分析
为什么填写`repo`就可以了呢？这是因为仓库里存放了模型和模型数据前后处理函数。模型文件为onnx后缀，前后处理函数则定义在`data_process.py`文件中。

## 分享我的模型
现在我们已经感受到了开源共享模型的乐趣，那么如何分享我自己的模型呢？就跟随继续往下看吧！

### 第一步：准备模型
例如我们用MMEduCls训练好了一个“猫狗分类”的模型，名为`cat_dog.onnx`，并且也准备好了一段推理代码如下：
```python
def inference(img_path):
    from XEdu.hub import Workflow as wf
    model = wf(task='mmedu',checkpoint='cat_dog.onnx')
    result = model.inference(data=img_path)
    return result
```
确保模型配合代码可以正常运行即可。

### 第二步：上传模型
这里以[魔搭社区](https://modelscope.cn/home)为例，首先需要通过网页注册/登录，然后点击[创建模型](https://modelscope.cn/models/create)，填写相关信息，其中`是否公开`选择“`公开模型`”。点击创建后再上传其他文件。

点击“模型文件”，点击“添加文件”，然后上传模型文件，文件名没有严格要求（上传位置选择“根目录”，且需要填写“文件信息”）。

点击“添加文件”，继续添加文件，现在我们添加一个文件名为`data_process.py`的文件，内容即为刚才我们定义的函数，注意函数名务必为`inference()`。

### 第三步：测试模型
接下来，我们测试一个模型是否可以顺利运行。下面代码中的`XXXXXXXXXX`替换为你的仓库名称。
```python
from XEdu.hub import Workflow as wf
img_path = 'cat.jpg'
model = wf(repo='XXXXXXXXXX')
result = model.inference(data=img_path)
print(result)
```
看起来运行一切正常，那么就ok啦！如果有问题的话，我们就需要调整一下代码，确保可以运行后，再分享出来。

### 填写生态共建问卷登记模型
为了让更多人知道你贡献的好玩的模型，我们诚邀您在上传模型之后，在这里填写[问卷](https://aicarrier.feishu.cn/share/base/form/shrcnZqkQ3n3EVkE9vEMhkMYAsf)登记您的模型，这样，就可以让您的模型加入[生态共建模型仓库](https://aicarrier.feishu.cn/wiki/VdWkwyNvtiUyAlkrOkEcZgBznOb?from=from_copylink)，让更多人看到并用上您的模型！

