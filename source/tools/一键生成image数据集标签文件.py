# 本文件可以放在数据集的根目录下运行
import os
# 如果不是在数据集根目录下，可以指定路径
set_path = './' 

templist = os.listdir(set_path +'training_set')
# 处理mac的特殊文件夹
classes = []
for line in templist:
    if line[0] !='.':
        classes.append(line)
    
with open(set_path +'classes.txt','w') as f:
    for line in classes: 
        str_line = line +'\n'
        f.write(str_line) # 文件分行写入，即类别名称

val_dir = set_path +'val_set/'  # 指定验证集文件路径
# 打开指定文件，写入标签信息
with open(set_path +'val.txt', 'w') as f:
    for cnt in range(len(classes)):
        t_dir = val_dir + classes[cnt]  # 指定验证集某个分类的文件目录
        files = os.listdir(t_dir)  # 列出当前类别的文件目录下的所有文件名
        # print(files)
        for line in files:
            str_line = classes[cnt] + '/' + line + ' ' + str(cnt) + '\n'
            f.write(str_line)  # 文件写入str_line，即标注信息

test_dir = set_path +'test_set/' # 指定测试集文件路径
# 打开指定文件，写入标签信息
with open(set_path +'test.txt','w') as f:
    for cnt in range(len(classes)):
        t_dir = test_dir + classes[cnt]  # 指定测试集某个分类的文件目录
        files = os.listdir(t_dir) # 列出当前类别的文件目录下的所有文件名
        # print(files)
        for line in files:
            str_line = classes[cnt] + '/' + line + ' '+str(cnt) +'\n'
            f.write(str_line)