# 常见问题解答
云文档链接：https://docs.qq.com/sheet/DQ0htU2poQlRSUmxn?tab=BB08J2

Q1：XEdu和OpenInnoLab什么关系？都是人工智能平台吗？

A1：XEdu是上海人工智能实验室智能教育中心为中小学AI教育设计的一套完整的学习工具。OpenInnoLab平台是上海人工智能实验室智能教育中心的开源平台，是个既可以学习AI也可以做AI项目的平台，网址就是**www.openinnolab.org.cn**。OpenInnoLab平台人工智能工坊提供多种主流深度学习框架的预置编程环境，包括XEdu工具的，在OpenInnoLab平台上可在线基于XEdu制作AI项目。



Q2：运行MMEdu相关代码 - 报错：No module named 'mmcv._ext'。

A2：①卸载mmcv：pip uninstall mmcv-full mmcv -y ②安装:pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.8/index.html



Q3：训练时标明device='cuda‘，实际好像用的是CPU？

A3：首先排查一下环境是否安装好了，看看cuda是否可用，如果cuda不可用，训练时标明device='cuda‘，实际用的自然还是CPU。



Q4：XEdu有一直在更新吗？

A4：XEdu一直在更新，迭代记录详见https://xedu.readthedocs.io/zh/master/about/version_update.html#id4，每次迭代后相关模块的教程也会及时配套更新。