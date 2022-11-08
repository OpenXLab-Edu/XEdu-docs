图像分类模型LeNet-5
===================

`Backpropagation Applied to Handwritten Zip Code
Recognition <https://ieeexplore.ieee.org/document/6795724>`__

简介
----

LeNet是一种用于手写体字符识别的非常高效的卷积神经网络，通常LeNet指LeNet-5。

网络结构
--------

LeNet的网络结构示意图如下所示：

.. figure:: ../../images/dl_library/LeNet5.jpg


即输入手写字符图片经过C1，C2，C3这3个卷积神经网络后，再经过两层全连接神经网络，输出最终的分类结果。

优点
----

-  简单，易于初学者学习与使用
-  运行速度快，对硬件设备没有要求

适用领域
--------

-  手写体字符识别

参考文献
--------

.. code:: plain

   @ARTICLE{6795724,
     author={Y. {LeCun} and B. {Boser} and J. S. {Denker} and D. {Henderson} and R. E. {Howard} and W. {Hubbard} and L. D. {Jackel}},
     journal={Neural Computation},
     title={Backpropagation Applied to Handwritten Zip Code Recognition},
     year={1989},
     volume={1},
     number={4},
     pages={541-551},
     doi={10.1162/neco.1989.1.4.541}}
   }
