# XEdu的常见函数

## XEdu.utils中的函数
在XEdu-python库中，我们封装了一系列数据处理函数，可以帮助你方便地完成AI推理和部署。这些函数被封装在XEdu.utils中，你可以这样引入它们：
```python
from XEdu.utils import *
```
或者具体写明引入的函数
```python
from XEdu.utils import softmax, cosine_similarity, get_similarity
```
下面对函数展开介绍。
### softmax
softmax是一个常见的非线性函数，神经网络最终输出的结果是一串数字，如果想要把数字映射为各类概率，那么使用softmax函数再好不过了。向函数输入一个numpy数组，即可得到输出的结果。
- softmax的计算公式为：$$\sigma(z_i) = \frac{e^{z_{i}}}{\sum_{j=1}^K e^{z_{j}}} \ \ \ for\ i=1,2,\dots,K$$
```python
def softmax(x):
    """Apply the softmax operation to the input array.

    Args:
        x: The input numpy array, which can be of any shape.

    Returns:
        A numpy array after the softmax operation, with the same shape as the input array. Each element is between 0 and 1, and the sum of all elements in the same row is 1.
    """
    x1 = x - np.max(x, axis = 1, keepdims = True) #减掉最大值防止溢出    
    x1 = np.exp(x1) / np.sum(np.exp(x1), axis = 1, keepdims = True)
    return x1.tolist()
```

### cosine_similarity
该函数可以比较两个embedding序列的相似度，这里的相似度是以余弦相似度为计算指标的，其公式为：$$Cosine(x,y) = \frac{x \cdot y}{|x||y|}$$。

假设输入的待比较embedding序列尺度分别为(N, D)和(M, D)，则输出的结果尺度为(N, M)。

- embedding会在[图像嵌入和文本嵌入](https://xedu.readthedocs.io/zh/master/xedu_hub/introduction.html#id84)中用到，具体案例可参见：[教程1-7](https://www.openinnolab.org.cn/pjlab/project?id=65518e1ae79a38197e449843&sc=62f33550bf4f550f3e926cf2#public)

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

### get_similarity
该函数可以比较两个embedding序列的相似度，

假设输入的待比较embedding序列尺度分别为(N, D)和(M, D)，则输出的结果尺度为(N, M)。当然，也可以输入一维序列，假设尺度一个是(N, D)或(M, D)，另一个是(D,)，输出则为(N,)或(M,)，假设两个输入都是一维的，则输出是一个数值。

输入还可以指定计算方法method，可选'cosine', 'euclidean', 'manhattan', 'chebyshev', 'pearson'，默认是'cosine'（method='cosine'）。

对于相似度计算结果可选择是否进行归一化，默认是进行归一化（use_softmax=True）。
- embedding会在[图像嵌入和文本嵌入](https://xedu.readthedocs.io/zh/master/xedu_hub/introduction.html#id84)中用到，具体案例可参见：[教程1-7](https://www.openinnolab.org.cn/pjlab/project?id=65518e1ae79a38197e449843&sc=62f33550bf4f550f3e926cf2#public)

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