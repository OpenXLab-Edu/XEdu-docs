# 大模型服务的API与密钥获取

## API与密钥是什么？

API（Application Programming Interface，应用程序编程接口）是一种允许开发者将特定功能或数据集成到自己应用程序中的工具。借助大模型服务商提供的API服务，即可在自己的代码中远程调用大模型的能力，实现特定的功能。 

API密钥（API key）是一种唯一标识符，是用来访问和调用API服务的凭证。API密钥用于验证请求者的身份，确保只有授权用户才能使用服务。这个密钥由API服务提供商发放，用户在注册并验证身份后，一般通过平台的用户界面获取。通过API，开发者可以发送HTTP请求到指定的URL，并附上必要的参数和API密钥，来获取所需的服务或数据。 

大语言模型API允许开发者将大型语言模型集成到自己的应用程序中，通过发送HTTP请求到指定的URL，并附上必要的参数和API密钥，来获取模型的预测结果。

**注意:** 保护API密钥是非常重要的，因为如果密钥被未授权的第三方获取，可能导致费用增加或者服务被滥用。因此，我们应该确保自己的API密钥安全存储。

## XEduLLM支持的大模型API注册与密钥获取

XEduLLM支持多种大语言模型服务提供商，可以通过support_provider()来查看。

```python
from XEdu.LLM import Client
Client.support_provider() # 默认查看服务商英文名
# 输出：['openrouter', 'moonshot', 'deepseek', 'glm', 'ernie']
Client.support_provider('zh') # 查看服务商中文名
# 输出：['openrouter', '月之暗面-Kimi', '幻方-深度求索', '智谱-智谱清言', '百度-文心一言']

```

support_provider可以设置参数lang，表示语言，支持['en','zh']，默认'en'。

每个服务商需要分别注册和获取相应的密钥，密钥获取的注册网址和key的获取方式如下表所示：
<table class="docutils align-default">
    <thead>
        <tr class="row-odd">
            <th class="head">名称</th>
            <th class="head">注册网址</th>
            <th class="head">如何获取key</th>
            <th class="head">tokens赠送情况</th>
        </tr>
    </thead>
    <tbody>
        <tr class="row-even">
            <td>qwen（通义千问）</td>
            <td><a href="https://dashscope.console.aliyun.com/apiKey">https://dashscope.console.aliyun.com/apiKey</a></td>
            <td>API-KEY管理-创建新的API-KEY Key</td>
            <td>无</td>
        </tr>
    </tbody>
    <tbody>
        <tr class="row-even">
            <td>openrouter</td>
            <td><a href="https://openrouter.ai">https://openrouter.ai</a></td>
            <td>右上角个人头像-Keys-Create Key</td>
            <td>无</td>
        </tr>
    </tbody>
    <tbody>
        <tr class="row-even">
            <td>moonshot(月之暗面)</td>
            <td><a href="https://platform.moonshot.cn/console">https://platform.moonshot.cn/console</a></td>
            <td>左侧API Key管理-新建</td>
            <td>15.00 元（认证后领取）</td>
        </tr>
    </tbody>
    <tbody>
        <tr class="row-even">
            <td>deepseek(深度求索)</td>
            <td><a href="https://platform.deepseek.com/sign_up">https://platform.deepseek.com/sign_up</a></td>
            <td>左侧API keys-创建API key</td>
            <td>500万tokens（注意要去首页认证领取）</td>
        </tr>
    </tbody>
    <tbody>
        <tr class="row-even">
            <td>glm(智谱清言)</td>
            <td><a href="https://open.bigmodel.cn/login">https://open.bigmodel.cn/login</a></td>
            <td>左侧API keys-创建API key</td>
            <td>2500万tokens（有效期1个月）</td>
        </tr>
    </tbody>
    <tbody>
        <tr class="row-even">
            <td>ernie(文心一言)</td>
            <td><a href="https://console.bce.baidu.com/qianfan/ais/console/applicationConsole/application">https://console.bce.baidu.com/qianfan/ais/console/applicationConsole/application</a></td>
            <td>左侧应用接入-创建应用（不能重名）-同时需要API Key与Secret Key</td>
            <td>无（注意要实名认证开启服务，可在个人中心确认）</td>
        </tr>
    </tbody>
</table>


**注意:** 目前阶段，很多大模型服务商都有促销政策，多家服务商提供了免费的token额度，用于基本教学已经足够。
