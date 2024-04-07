# 用AI解决问题的一般步骤

人工智能教育区别于传统编程教育的最重要标志，那就是要训练AI模型。用AI解决问题，核心工作是训练一个具备某种智能的AI模型。

## Step1: 发现问题
发现生活中的问题，并转换为AI可解决的问题

## Step2: 寻找可行解决方案
分析是需要找一个现成的模型，还是要训练一个个性化模型。

## Step3: 采集数据
分析问题，采集针对本问题的各种数据，并对数据进行探索和特征提取。使得数据方便输入模型。

### 专题1：从采集的数据到ImageNet数据。
当我们采集到数据后，要准备一份可以训练的，包含training_set、val_set和test_set的规范数据集，可以用下面的代码进行转换：
```python
import os, random, shutil
target_path = './target_path/'
origin_path = './datasets/'
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1
random_list = True

classes = os.listdir(origin_path)
print(classes)

os.makedirs(os.path.join(target_path,'training_set'))
os.makedirs(os.path.join(target_path,'val_set'))
os.makedirs(os.path.join(target_path,'test_set'))

f = open(os.path.join(target_path,'classes.txt'),'w')
for i, c in enumerate(classes):
    f.write(c+' '+str(i)+'\n')
    os.makedirs(os.path.join(target_path,'training_set',c))
    os.makedirs(os.path.join(target_path,'val_set',c))
    os.makedirs(os.path.join(target_path,'test_set',c))
f.close()

for c in classes:
    image_path = os.path.join(origin_path, c)
    images = os.listdir(image_path)
    num = len(images)
    if random_list == True:
        random.shuffle(images)
    for i,pic in enumerate(images):
        _,ext = os.path.splitext(pic)
        if ext !='.jpg':
            continue
        o_path = os.path.join(image_path,pic)
        if i <= num*train_ratio:
            t_path = os.path.join(target_path, 'training_set',c,pic)
        elif i <= num*(train_ratio+val_ratio):
            t_path = os.path.join(target_path, 'val_set',c,pic)
        else:
            t_path = os.path.join(target_path, 'test_set',c,pic)
        shutil.move(o_path,t_path)



```

## Step4: 模型训练
将按照格式要求整理好的数据输入选择的模型，进行训练。如果效果不够好，调整超参数后，继续训练。或者寻找预训练模型，在此基础上调优。
- 方式一：[MMEdu](../mmedu.html)
- 方式二：[BaseNN](../basenn.html)
- 方式三：[BaseML](../baseml.html)
## Step5: 模型验证
验证模型的效果。
- 方式一：[MMEdu](../mmedu/quick_start.html#id7)
- 方式二：[BaseNN](../basenn/quick_start.html#id10)
- 方式三：[BaseML](../baseml/quick_start.html#id9)
## Step6: 模型应用
使用规范的方式应用模型，可以添加相关的多模态操作，使项目更丰富，得以真正解决现实问题。
- 方式一：[XEduHub](../xedu_hub.html)
- 方式二：[BaseDeploy](../basedt.html)
