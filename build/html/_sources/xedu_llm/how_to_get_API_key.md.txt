# API与密钥获取

## API与密钥是什么？

语言模型API是一种应用程序编程接口，它允许开发者将大型语言模型集成到自己的应用程序中。

大语言模型的密钥是一个唯一标识符，通常被称为API密钥（API key），它是用来访问和调用大语言模型API的凭证。API密钥用于验证请求者的身份，确保只有授权用户才能使用服务。这个密钥是由提供大语言模型服务的提供商发放给用户的，用户在注册并验证身份后，一般通过平台的用户界面获取。

通过大语言模型的API，发送HTTP请求到指定的URL，并附上必要的参数和API密钥，来获取模型的预测结果。


**注意:** 保护API密钥是非常重要的，因为如果密钥被未授权的第三方获取，可能导致费用增加或者服务被滥用。因此，我们应该确保自己的API密钥安全存储。

## API注册与密钥获取

XEduLLM支持多种大模型服务提供商，可以通过support_provider()来查看。

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
            <td>openrouter</td>
            <td><a href="https://openrouter.ai">https://openrouter.ai</a></td>
            <td>右上角个人头像-Keys-Create Key</td>
            <td>无限制</td>
        </tr>
    </tbody>
    <tbody>
        <tr class="row-even">
            <td>moonshot(月之暗面)</td>
            <td><a href="https://platform.moonshot.cn/console">https://platform.moonshot.cn/console</a></td>
            <td>左侧API Key管理-新建</td>
            <td>15.00 元</td>
        </tr>
    </tbody>
    <tbody>
        <tr class="row-even">
            <td>deepseek(深度求索)</td>
            <td><a href="https://platform.deepseek.com/sign_up">https://platform.deepseek.com/sign_up</a></td>
            <td>左侧API keys-创建API key</td>
            <td>500万tokens（要去首页认证领取）</td>
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
            <td>左侧应用接入-创建应用-API KeySecret Key</td>
            <td>0</td>
        </tr>
    </tbody>
</table>
