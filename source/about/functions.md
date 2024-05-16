# XEdu的常见函数

## XEdu.utils中的函数
在XEdu-python库中，我们封装了一系列数据处理函数，可以帮助你方便地完成AI推理和部署。这些函数被封装在XEdu.utils中，你可以这样引入它们：
```python
from XEdu.utils import *
```
或者具体写明引入的函数
```python
from XEdu.utils import softmax, cosine_similarity, get_similarity, visualize similarity
```
下面对函数展开使用介绍。
### softmax
1. 函数说明

softmax函数是一个常用的非线性函数，它用于将一个numpy数组映射到0到1之间的数值，同时所有数值之和为1。神经网络最终输出的结果是一串数字，如果想要把数字映射为各类概率，那么使用softmax函数再好不过了。

2. 使用示例
```python
from XEdu.utils import *
import numpy as np
data = np.array([[1,2],[3,3]])
output = softmax(data)
print(output)
# [[0.2689414213699951, 0.7310585786300049], [0.5, 0.5]]
```
在这个例子中，需要处理两组数据，[1,2]和[3,3]，对于第一组数据，按照softmax算法（一种指数算法）进行映射，得到输出是[0.2689414213699951, 0.7310585786300049]，而第二组数据两个数值相等，得到就是平均分配的[0.5, 0.5]。每一组数据经过处理之后的加和都是一。

3. 参数说明

输入参数：

`x`：numpy array，对数据尺寸没有要求。

输出参数：

list，形状与输入相同，数组映射到0到1之间的数值，同时所有数值之和为1。

4. 函数实现揭秘
```python
def softmax(x):
    x1 = x - np.max(x, axis = 1, keepdims = True) #减掉最大值防止溢出    
    x1 = np.exp(x1) / np.sum(np.exp(x1), axis = 1, keepdims = True)
    return x1.tolist()
```

### cosine_similarity
1. 函数说明
该函数可以比较两个embedding序列的相似度，这里的相似度是以余弦相似度为计算指标的，在高中我们就学习过余弦定理，这里的余弦相似度公式也是类似的，具体计算可以参考[这里](https://zhuanlan.zhihu.com/p/43396514)。

我们在`wf(task='embedding_image')`或者`wf(task='embedding_text')`的任务中，对数据进行embedding操作之后，可以计算不同数据之间的相似度，就可以使用该函数。

- embedding会在[图像嵌入和文本嵌入](https://xedu.readthedocs.io/zh/master/xedu_hub/introduction.html#id92)中用到，具体案例可参见：[教程1-7](https://www.openinnolab.org.cn/pjlab/project?id=65518e1ae79a38197e449843&sc=62f33550bf4f550f3e926cf2#public)

2. 使用示例
```python
from XEdu.hub import Workflow as wf # 导入库
from XEdu.utils import *
txt_emb = wf(task='embedding_text')# 实例化模型
txts1 = ['cat','dog','room','elephant'] # 指定文本1
txts2 = ['a cat','a dog','a room','an elephant'] # 指定文本2
txt_embeddings1 = txt_emb.inference(data=txts1) # 获得向量1
txt_embeddings2 = txt_emb.inference(data=txts2) # 获得向量2
print(txt_embeddings1) # 打印向量1，向量2
# [[ 0.20516919 -0.03279374 -0.06166159 ... ]...[ -0.22821501  0.08871169 0.08685149]]
print(txt_embeddings1.shape)
# (4, 512)
output = cosine_similarity(txt_embeddings1,txt_embeddings2)
print(output)
# [[0.94926983 0.86368805 0.7956152  0.8016052 ]
# [0.89295036 0.9511493  0.8203819  0.82089627]
# [0.8249735  0.8343273  0.97274196 0.76703286]
# [0.81858265 0.8157523  0.7587172  0.9856292 ]]
```
在这个例子中，我们使用了`wf(task='embedding_image')`文本embedding操作，针对两组文本进行处理，每个字符串处理为512个特征，txts1有4个单词，所以txt_embeddings1.shape是(4,512)。对两组文本转换出的向量进行相似度比较，可以得到一个比较矩阵，代表每两个字符串之间的相似度，我们可以看到对角线上的词相似度是最高的。

下面这个例子将让你有更好的理解：
```python
from XEdu.hub import Workflow as wf # 导入库
from XEdu.utils import *
txt_emb = wf(task='embedding_text')# 实例化模型
txts1 = ['cat','dog'] # 指定文本1
txts2 = ['a cat','a dog','a room','an elephant'] # 指定文本2
txt_embeddings1 = txt_emb.inference(data=txts1) # 获得向量1
txt_embeddings2 = txt_emb.inference(data=txts2) # 获得向量2
print(txt_embeddings1.shape)
# (2, 512) 两组文本中的字符串数量无需一致，但都会转换为512个特征
output = cosine_similarity(txt_embeddings1,txt_embeddings2) # 计算向量1和向量2的余弦相似度
print(output)
# [[0.94926983 0.86368805 0.7956152  0.8016052 ]
# [0.89295036 0.9511493  0.8203819  0.82089627]]
print(softmax(output))
# [[0.27485617995262146, 0.25231191515922546, 0.23570789396762848, 0.2371240258216858], 
# [0.25507545471191406, 0.2703610360622406, 0.23722068965435028, 0.2373427450656891]]
visualize_similarity(output,txts1,txts2) # 可视化相似度矩阵
```

3. 参数说明

`embeddings_1`：一个numpy数组，数据维度为(N, D)，表示N个具有D维的embedding；

`embeddings_2`：另一个numpy数组，数据维度为(M, D)，表示M个具有D维的embedding；

4. 函数实现揭秘

该函数实际是利用了numpy的矩阵乘法运算符`@`，numpy的矩阵乘法运算符`@`可以直接实现两个矩阵的点积，从而计算两个embedding序列的余弦相似度。最终输出的结果尺度为(N, M)。
```python
def cosine_similarity(embeddings_1: np.ndarray, embeddings_2: np.ndarray) -> np.ndarray:
    """Compute the pairwise cosine similarities between two embedding arrays.

    Args:
        embeddings_1: An array of embeddings of shape (N, D).
        embeddings_2: An array of embeddings of shape (M, D).

    Returns:
        An array of shape (N, M) with the pairwise cosine similarities.
    """

    for embeddings in [embeddings_1, embeddings_2]:
        if len(embeddings.shape) != 2:
            raise ValueError(
                f"Expected 2-D arrays but got shape {embeddings.shape}."
            )

    d1 = embeddings_1.shape[1]
    d2 = embeddings_2.shape[1]
    if d1 != d2:
        raise ValueError(
            "Expected second dimension of embeddings_1 and embeddings_2 to "
            f"match, but got {d1} and {d2} respectively."
        )

    def normalize(embeddings):
        return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    embeddings_1 = normalize(embeddings_1)
    embeddings_2 = normalize(embeddings_2)

    return embeddings_1 @ embeddings_2.T
```

5. 更多用法

图片之间也可以计算相似度，给定的列表中，需要指明各图片的文件所在路径。
```python
from XEdu.hub import Workflow as wf # 导入库
from XEdu.utils import *
img_emb = wf(task='embedding_image') # 实例化模型
image_embeddings1 = img_emb.inference(data='demo/cat.png') # 模型推理
image_embeddings2 = img_emb.inference(data='demo/dog.png') # 模型推理
output = cosine_similarity(image_embeddings1,image_embeddings2) # 计算向量1和向量2的余弦相似度
print(output)
print(softmax(output))
visualize_similarity(output,['demo/cat.png','demo/dog.png'],['demo/cat.png','demo/dog.png'])
```
Q: 当我们提取完了特征能做什么任务呢？
A: 零样本分类！

Q: 什么是零样本分类呢？
A: 举个例子，现在我们想要分类图片中的猫是黑色的还是黄色的，按照图像分类的方式，我们需要收集数据集，并且标注数据集，再进行模型训练，最后才能使用训练出来的模型对图像进行分类。而现在，我们使用的“图像特征提取”和“文本特征提取”只需通过特征向量就可以进行分类，避免了大量的标注工作。

假设我们已经通过图像特征提取和文本特征提取把cat.jpg,'a black cat','a yellow cat'分别变成了3堆数字（3个512维向量），但是很显然，我们看不懂这些数字，但是计算机可以！
通过让计算机将数字进行运算，即将图像和文本的特征向量作比较，就能看出很多信息，这也叫计算向量之间相似度。

下面就示范使用cosine_similarity比较两个embedding序列的相似度，也可以直接使用get_similarity函数，选择method='cosine'来实现。尝试一下，在没有训练集的情况下，仅通过图像特征提取（图像->向量），以及文本特征提取（文本->向量），通过计算向量相似度的方式，能够有什么惊喜呢？
```python
from XEdu.utils import * # 导入库
logits = cosine_similarity(image_embeddings, txt_embeddings) # 计算余弦相似度
print(logits) # 输出相似度计算结果
# [[0.48788464069366455, 0.5121153593063354]]
visualize_similarity(logits,[images],texts) # 可视化相似度矩阵
```
![](../../images/about/imbedding1.png)

### get_similarity
1. 函数说明

上面的函数cosine_similarity能够计算两个embedding向量的余弦相似度，而get_similarity则提供了更丰富的选择，该函数可以选择相似度的比较算法，可选'cosine', 'euclidean', 'manhattan', 'chebyshev', 'pearson'，默认是'cosine'（method='cosine'）。

- embedding会在[图像嵌入和文本嵌入](https://xedu.readthedocs.io/zh/master/xedu_hub/introduction.html#id92)中用到，具体案例可参见：[教程1-7](https://www.openinnolab.org.cn/pjlab/project?id=65518e1ae79a38197e449843&sc=62f33550bf4f550f3e926cf2#public)

2. 使用示例

```python
from XEdu.utils import * # 导入库
logits = get_similarity(image_embeddings, txt_embeddings,method='cosine') # 计算余弦相似度
print(logits) # 输出相似度计算结果
# [[0.48788464069366455, 0.5121153593063354]]
```
可以看出，使用这个函数是对前面cosine_similarity和softmax的统一封装，这里经历了计算相似度，然后进行归一化的过程。

3. 参数说明

输入参数：

`embeddings_1`：一个numpy数组，数据维度为(N, D)，表示N个具有D维的embedding；

`embeddings_2`：另一个numpy数组，数据维度为(M, D)，表示M个具有D维的embedding；

`method`：计算方法，可选'cosine', 'euclidean', 'manhattan', 'chebyshev', 'pearson'，默认是'cosine'（method='cosine'）；

`use_softmax`：是否进行归一化，默认为True，即进行归一化。

输出参数：

list，形状与输入相同，数组映射到0到1之间的数值，同时所有数值之和为1。

4. 函数实现揭秘

该函数实际是利用了numpy的矩阵乘法运算符`@`，numpy的矩阵乘法运算符`@`可以直接实现两个矩阵的点积，从而计算两个embedding序列的余弦相似度。最终输出的结果尺度为
输入还可以指定计算方法method，可选'cosine', 'euclidean', 'manhattan', 'chebyshev', 'pearson'，默认是'cosine'（method='cosine'）。

对于相似度计算结果可选择是否进行归一化，默认是进行归一化（use_softmax=True）。

```python
def get_similarity(embeddings_1: np.ndarray, embeddings_2: np.ndarray,method:str='cosine',use_softmax:bool=True) -> np.ndarray:
    """Compute pairwise similarity scores between two arrays of embeddings.
    Args:
        embeddings_1: An array of embeddings of shape (N, D) or (D,).
        embeddings_2: An array of embeddings of shape (M, D) or (D,).
        method: The method used to compute similarity. Options are 'cosine', 'euclidean', 'manhattan', 'chebyshev', 'pearson'. Default is 'cosine'.
        use_softmax: Whether to apply softmax to the similarity scores. Default is True.

    Returns:
        An array with the pairwise similarity scores. If both inputs are 2-D,
            the output will be of shape (N, M). If one input is 1-D, the output
            will be of shape (N,) or (M,). If both inputs are 1-D, the output
            will be a scalar.
    """
    if embeddings_1.ndim == 1:
        # Convert to 2-D array using x[np.newaxis, :]
        # and remove the extra dimension at the end.
        return get_similarity(
            embeddings_1[np.newaxis, :], embeddings_2
        )[0]

    if embeddings_2.ndim == 1:
        # Convert to 2-D array using x[np.newaxis, :]
        # and remove the extra dimension at the end.
        return get_similarity(
            embeddings_1, embeddings_2[np.newaxis, :]
        )[:, 0]
    if method == 'cosine':
        similarity =  cosine_similarity(embeddings_1, embeddings_2) * 100
    elif method == 'euclidean':
        distance = np.array([[np.linalg.norm(i - j) for j in embeddings_2] for i in embeddings_1]) * 100
        sigma = np.mean(distance)  # Or choose sigma in some other way
        similarity = np.exp(-distance ** 2 / (2 * sigma ** 2)) * 100
    elif method == 'pearson':
        similarity = np.array([[np.corrcoef(i, j)[0,1] for j in embeddings_2] for i in embeddings_1]) * 100
    else:
        raise ValueError(
            f"Expected method to be cosine,euclidean and pearson but got {method}."
        )
    if use_softmax:
        return softmax(similarity)
    else:
        return similarity

```

### cosine_similarity 函数和get_similarity函数的联系

get_similarity 函数实际上是对 cosine_similarity 函数的扩展和泛化。它不仅支持余弦相似度，还支持其他距离测量方法，并提供了可选的 softmax 应用，使其功能更为丰富和灵活。在 get_similarity 中使用 'cosine' 方法时，它会调用 cosine_similarity 函数来计算余弦相似度，同时还有是否进行归一化的处理。因此 cosine_similarity 可以视为 get_similarity 的一个特定实现。

### visualize_similarity
1. 函数说明

2. 使用示例

3. 参数说明

