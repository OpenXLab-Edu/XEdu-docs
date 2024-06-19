# XEduLLM功能详解

XEduLLM是一个大语言模型工具库，为开发者提供了一种简便的方式来与大语言模型进行交互和集成。该工具库围绕特定的大语言模型API构建，使得开发者可以在自己的应用程序中轻松地利用这些模型的强大功能。

## 通用接口访问工具Client 

Client是一个通过API（应用程序编程接口）与最先进的大语言模型交流而设计的通用接口访问工具。通过几行代码就可以通过API调用各种优秀的大语言模型，发送请求到远程服务器，服务器运行着大模型，利用服务器算力处理请求，并返回响应。
大语言模型接口通常需要使用API密钥（API key）来进行访问。API密钥是一个唯一标识符，用于跟踪和控制API的使用情况，确保只有授权用户才能访问模型的服务。API密钥的获取请参考<a href="https://xedu.readthedocs.io/zh-cn/master/xedu_llm/how_to_get_API_key.html">API与密钥获取</a>。

### 功能一：通过服务商发送请求

XEduLLM支持多种大语言模型服务提供商，可以通过support_provider()来查看，代码如下：

```python
from XEdu.LLM import Client
Client.support_provider() # 默认查看服务商英文名
# 输出：['openrouter', 'moonshot', 'deepseek', 'glm', 'ernie']
Client.support_provider(lang = 'zh') # 查看服务商中文名
# 输出：['openrouter', '月之暗面-Kimi', '幻方-深度求索', '智谱-智谱清言', '百度-文心一言']
```

support_provider可以设置参数lang，表示语言，支持['en','zh']，默认'en'。

#### 向 openrouter 服务商发送请求示例

完整代码如下：

```python
from XEdu.LLM import Client # 导入库
chatbot = Client(provider='openrouter',
               api_key='sk-or-v1-6d7672a58c3c837f2……c0f30a3b1c3') # 实例化客户端
res = chatbot.inference('你好，用中文介绍一下你自己') # 输入请求，执行推理并得到结果
print(res) # 结果输出
```

输出示例：

![](../images/xedullm/output0.jpg)

接下来对示例代码进行详细说明。

#### 1. 客户端声明

```python
from XEdu.LLM import Client # 导入库
chatbot = Client(provider='openrouter',
               api_key='sk-or-v1-6d7672a58c3c837f2……c0f30a3b1c3') # 实例化客户端
```
客户端声明函数Client()中有六个参数可以设置，本功能中使用的参数是`provider`和`api_key`，部分服务商还需要提供`secret_key`，根据具体要求设置即可。

##### 参数说明

- `base_url`(str):API的服务器地址。
- `provider`(str):指定服务提供商的名称。可以通过Client.support_provider()语句来查看支持哪些服务提供商。声明时，支持多种不同provider书写格式，英文/中文/公司/产品，如'deepseek'，'幻方-深度求索'，'幻方'，'深度求索'。
- `api_key`(str):访问密钥（Access Key），用于验证用户身份并授权访问API服务。
- `secret_key`(str):秘密密钥（Secret Key），与API密钥一起使用，提供更高级别的安全性。在文心一言ernie中，需要同时提供API密钥和秘密密钥来进行身份验证，其他的服务商不需要。
- `model`(str):这个参数用于指定使用的具体模型。在一个API中，可能有多个不同的模型可供选择，可以通过设置这个参数来选择需要的模型。此外可以通过`print(chatbot.support_model())`语句来查看该chatbot支持哪些模型。下文对此参数使用有详细说明。

#### 2. 模型推理

推理方式一：单句对话
```python
res = chatbot.inference("你好，用中文介绍一下你自己")
```
推理方式二：多句对话
```python
talk = [
    {'role':'user'     ,'content':'用一段话介绍数组'},
    {'role':'assistant','content':'数组是一种数据结构，用于存储具有相同数据类型的多个值。它们允许使用一个变量来存储多个元素，并可以通过下标（index）快速访问这些元素。数组可以是固定大小的或动态大小的，而且可以在数组中添加、删除和修改元素。在许多编程语言中，数组是非常重要的数据结构，并广泛用于数据处理和算法实现中。'},
    {'role':'user'     ,'content':'除此之外还有什么'},
]
res = chatbot.inference(talk)
```

##### 参数说明

- `input` (str|list): 对话机器人处理的输入。它可以是单个字符串，也可以是一个列表，列表中的每个元素是一个包含role和content键的字典，分别表示角色的类型（如"user"或"assistant"）和对话内容。
- `temperature` (float): 控制输出的随机性。它的值介于0和1之间。较高的值（如0.7）会使输出更加随机和创造性，而较低的值（如0.2）会使输出更加稳定和确定性。不同模型默认值不同。
- `top_p (float)`: 指定模型考虑的概率质量。它的值介于0和1之间，表示考虑概率质量最高的标记的结果的百分比。例如，0.1意味着只考虑概率质量最高的10%的标记。不同模型默认值不同。
- `stream` (bool): 是否返回一个生成器对象，默认为False。当stream为True时，client.inference函数将返回一个生成器对象。生成器对象是一个迭代器，它会在每次迭代时返回一部分输出结果，而不是一次性返回所有结果。这使得您可以按需获取输出，而不是等待整个结果集生成后再处理。

#### 3. 推理结果输出

推理结果输出方式一：直接输出
```python
res = chatbot.inference("你好,用中文介绍一下你自己",stream=False)
print(res)
```
当推理函数的参数`stream`为False时，返回的结果的是字符串，默认`stream`为False。允许输出各种格式文本，比如表格、代码等格式，参考代码如下。

```python
# 带格式的输出
res1 = chatbot.inference('给出一段代码，实现猜数字小游戏')
print(res1)
# 输出表格
res2 = chatbot.inference('给出一个表格，可以存储学生姓名、学号、选课课程、学分、成绩')
print(res2)
```


推理结果输出方式二：流式输出
```python
res = chatbot.inference("你好,用中文介绍一下你自己",stream=True)
for i in res:
    print(i, flush=True, end='')
```

当参数`stream`为True时，返回一个生成器对象。生成器对象是一个迭代器，它会在每次迭代时返回一部分输出结果，而不是一次性返回所有结果，所以不能直接使用print(res)输出，可以使用for循环来迭代生成器并打印生成器中的每个元素。

#### 举一反三，尝试使用不同的服务商
完成了向openrouter服务商发送请求的代码学习，我们可以举一反三向其他服务商发送请求。

### 功能二：指定使用模型

一个服务商中可以有多个不同的大语言模型，每个模型都有不同性能特点。我们可以指定服务商中包含的模型，来处理请求，并返回相应的结果。

步骤一：查看该服务商支持哪些大语言模型

可以通过`print(chatbot.support_model())`语句来查看该chatbot支持哪些模型，代码如下所示：

```python
from XEdu.LLM import Client # 导入库
chatbot = Client(provider='openrouter',
               api_key='sk-or-v1-6d7672a58c3c837f2……c0f30a3b1c3') # 实例化客户端
print(chatbot.support_model()) # 查看该chatbot支持哪些模型
```

输出示例：
```python
['openrouter/auto', 'nousresearch/nous-capybara-7b:free', 'mistralai/mistral-7b-instruct:free', 'huggingfaceh4/zephyr-7b-beta:free', 'openchat/openchat-7b:free', 'gryphe/mythomist-7b:free', 'undi95/toppy-m-7b:free', 'google/gemma-7b-it:free', 'meta-llama/llama-3-8b-instruct:free', 'microsoft/phi-3-mini-128k-instruct:free', 'microsoft/phi-3-medium-128k-instruct:free', 'koboldai/psyfighter-13b-2', 'intel/neural-chat-7b', 'mancer/weaver', 'pygmalionai/mythalion-13b', 'xwin-lm/xwin-lm-70b', 'alpindale/goliath-120b', 'neversleep/noromaid-20b', 'gryphe/mythomist-7b', 'sophosympatheia/midnight-rose-70b', 'sao10k/fimbulvetr-11b-v2', 'neversleep/llama-3-lumimaid-8b', 'neversleep/llama-3-lumimaid-70b', 'undi95/remm-slerp-l2-13b:extended', 'gryphe/mythomax-l2-13b:extended', 'meta-llama/llama-3-8b-instruct:extended', 'neversleep/llama-3-lumimaid-8b:extended', 'nousresearch/nous-capybara-7b', 'meta-llama/codellama-34b-instruct', 'codellama/codellama-70b-instruct', 'phind/phind-codellama-34b', 'open-orca/mistral-7b-openorca', 'teknium/openhermes-2-mistral-7b', 'undi95/remm-slerp-l2-13b', '01-ai/yi-34b-chat', '01-ai/yi-34b', '01-ai/yi-6b', 'togethercomputer/stripedhyena-nous-7b', 'togethercomputer/stripedhyena-hessian-7b', 'mistralai/mixtral-8x7b', 'nousresearch/nous-hermes-yi-34b', 'nousresearch/nous-hermes-2-mixtral-8x7b-sft', 'nousresearch/nous-hermes-2-mistral-7b-dpo', 'meta-llama/llama-3-8b', 'meta-llama/llama-3-70b', 'databricks/dbrx-instruct', 'allenai/olmo-7b-instruct', 'snowflake/snowflake-arctic-instruct', 'qwen/qwen-110b-chat', 'qwen/qwen-14b-chat', 'qwen/qwen-7b-chat', 'qwen/qwen-4b-chat', 'mistralai/mixtral-8x7b-instruct:nitro', 'openai/gpt-3.5-turbo', 'openai/gpt-3.5-turbo-0125', 'openai/gpt-3.5-turbo-1106', 'openai/gpt-3.5-turbo-0613', 'openai/gpt-3.5-turbo-0301', 'openai/gpt-3.5-turbo-16k', 'openai/gpt-4o', 'openai/gpt-4o-2024-05-13', 'openai/gpt-4-turbo', 'openai/gpt-4-turbo-preview', 'openai/gpt-4-1106-preview', 'openai/gpt-4', 'openai/gpt-4-0314', 'openai/gpt-4-32k', 'openai/gpt-4-32k-0314', 'openai/gpt-4-vision-preview', 'openai/gpt-3.5-turbo-instruct', 'google/palm-2-chat-bison', 'google/palm-2-codechat-bison', 'google/palm-2-chat-bison-32k', 'google/palm-2-codechat-bison-32k', 'google/gemini-pro', 'google/gemini-pro-vision', 'google/gemini-pro-1.5', 'google/gemini-flash-1.5', 'perplexity/llama-3-sonar-small-32k-chat', 'perplexity/llama-3-sonar-small-32k-online', 'perplexity/llama-3-sonar-large-32k-chat', 'perplexity/llama-3-sonar-large-32k-online', 'fireworks/firellava-13b', 'anthropic/claude-3-opus', 'anthropic/claude-3-sonnet', 'anthropic/claude-3-haiku', 'anthropic/claude-2', 'anthropic/claude-2.0', 'anthropic/claude-2.1', 'anthropic/claude-instant-1', 'anthropic/claude-3-opus:beta', 'anthropic/claude-3-sonnet:beta', 'anthropic/claude-3-haiku:beta', 'anthropic/claude-2:beta', 'anthropic/claude-2.0:beta', 'anthropic/claude-2.1:beta', 'anthropic/claude-instant-1:beta', 'meta-llama/llama-2-13b-chat', 'meta-llama/llama-2-70b-chat', 'nousresearch/nous-hermes-llama2-13b', 'nousresearch/nous-capybara-34b', 'jondurbin/airoboros-l2-70b', 'austism/chronos-hermes-13b', 'teknium/openhermes-2.5-mistral-7b', 'gryphe/mythomax-l2-13b', 'undi95/toppy-m-7b', 'lizpreciatior/lzlv-70b-fp16-hf', 'mistralai/mixtral-8x7b-instruct', 'neversleep/noromaid-mixtral-8x7b-instruct', 'nousresearch/nous-hermes-2-mixtral-8x7b-dpo', 'rwkv/rwkv-5-world-3b', 'recursal/rwkv-5-3b-ai-town', 'recursal/eagle-7b', 'google/gemma-7b-it', 'huggingfaceh4/zephyr-7b-beta', 'meta-llama/llama-3-8b-instruct', 'meta-llama/llama-3-70b-instruct', 'meta-llama/llama-guard-2-8b', 'microsoft/wizardlm-2-8x22b', 'microsoft/wizardlm-2-7b', 'microsoft/phi-3-mini-128k-instruct', 'microsoft/phi-3-medium-128k-instruct', 'mistralai/mixtral-8x22b', 'mistralai/mixtral-8x22b-instruct', 'mistralai/mistral-7b-instruct-v0.1', 'mistralai/mistral-7b-instruct-v0.2', 'mistralai/mistral-7b-instruct', 'mistralai/mistral-7b-instruct-v0.3', 'lynn/soliloquy-l3', 'jebcarter/psyfighter-13b', 'jondurbin/bagel-34b', 'cognitivecomputations/dolphin-mixtral-8x7b', 'cognitivecomputations/dolphin-mixtral-8x22b', 'liuhaotian/llava-yi-34b', 'nousresearch/hermes-2-pro-llama-3-8b', 'openchat/openchat-7b', 'openchat/openchat-8b', 'bigcode/starcoder2-15b-instruct', 'qwen/qwen-2-72b-instruct', 'qwen/qwen-72b-chat', 'qwen/qwen-32b-chat', 'deepseek/deepseek-chat', 'deepseek/deepseek-coder', 'anthropic/claude-1', 'anthropic/claude-1.2', 'anthropic/claude-instant-1.0', 'anthropic/claude-instant-1.1', 'meta-llama/llama-2-70b-chat:nitro', 'gryphe/mythomax-l2-13b:nitro', 'mistralai/mistral-7b-instruct:nitro', 'google/gemma-7b-it:nitro', 'undi95/toppy-m-7b:nitro', 'meta-llama/llama-3-8b-instruct:nitro', 'meta-llama/llama-3-70b-instruct:nitro', 'liuhaotian/llava-13b', 'nousresearch/nous-hermes-2-vision-7b', 'mistralai/mistral-tiny', 'mistralai/mistral-small', 'mistralai/mistral-medium', 'mistralai/mistral-large', 'cohere/command', 'cohere/command-r', 'cohere/command-r-plus']
```

步骤二：指定使用的大语言模型

指定chatbot使用接口中查询到的某个模型，例如前面查询到的模型`mistralai/mistral-7b-instruct:free`，示例代码如下所示：
```python
from XEdu.LLM import Client # 导入库
chatbot = Client(provider='openrouter',
               api_key='sk-or-v1-6d7672a58c3c837……f30a3b1c3',
               model='mistralai/mistral-7b-instruct:free') # 实例化客户端
res = chatbot.inference('你好，用中文介绍一下你自己') # 输入请求，执行推理并得到结果
print(res) # 结果输出
```
本功能示例代码中声明函数Client()新增使用`model`(str)，这个参数用于指定使用的具体模型。在一个API中，可能有多个不同的模型可供选择，可以通过设置这个参数来选择需要的模型。此外可以通过`print(chatbot.support_model())`语句来查看该chatbot支持哪些模型。

后续模型推理和推理结果输出与功能一一致，不再重复。

### 功能三：通过大模型API的服务器地址发送请求

除了向服务商发送请求外，还可以向服务器地址发送请求，以向openrouter服务器地址发送请求为例，完整代码如下：

```python
from XEdu.LLM import Client # 导入库
chatbot = Client(base_url='https://openrouter.ai/api/v1',
               api_key='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',
               model="mistralai/mistral-7b-instruct:free") # 实例化客户端
res = chatbot.inference('你好，用中文介绍一下你自己') # 输入请求，执行推理并得到结果
print(res)
```

本功能示例代码中声明函数Client()新增使用的参数是`base_url`(str)，为API的服务器地址。

通过阅读模型的官方文档，可以找到该模型所提供的服务器地址（`base_url`）。下面提供了部分供应商的base_url。

通过API的服务器地址发送请求时需要指定模型，具体可指定的模型可以通过`print(chatbot.support_model())`语句查询（部分域下不支持查看模型列表，如智谱清言），推荐查看各大语言模型API文档。

<table class="docutils align-default">
    <thead>
        <tr class="row-odd">
            <th class="head">名称</th>
            <th class="head">base_url</th>
        </tr>
    </thead>
    <tbody>
        <tr class="row-even">
            <td>openrouter</td>
            <td>https://openrouter.ai/api/v1</td>
        </tr>
    </tbody>
    <tbody>
        <tr class="row-even">
            <td>moonshot（月之暗面）</td>
            <td>https://api.moonshot.cn/v1</td>
        </tr>
    </tbody>
    <tbody>
        <tr class="row-even">
            <td>deepseek（深度求索）</td>
            <td>https://api.deepseek.com</td>
        </tr>
    </tbody>
    <tbody>
        <tr class="row-even">
            <td>glm（智谱清言）</td>
            <td>https://open.bigmodel.cn/api/paas/v4/</td>
        </tr>
    </tbody>
        <tbody>
        <tr class="row-even">
            <td>ernie（文心一言）</td>
            <td>不提供，不建议使用此方法</td>
        </tr>
    </tbody>
</table>
我们可以举一反三向其他大模型API的服务器地址发送请求。

### 功能四：启动基于网页的聊天机器人服务

使用`run` 方法可启动一个基于网页的聊天机器人服务，使用户可以通过网页界面与聊天机器人进行对话。

```python
chatbot.run(host='127.0.0.1')
```

这个方法将会在指定的主机地址（默认是 `127.0.0.1`）上启动一个 Web 服务器，并在浏览器中提供一个用户界面，方便用户进行实时对话。

##### 参数说明

- `host` (str): 指定 Web 服务器的主机地址。默认值为 `'127.0.0.1'`，表示本地回环地址，仅能本机访问。你也可以将其设置为 `'0.0.0.0'`，使其可在局域网中访问。
- `port` (int): 指定 Web 服务器的端口号，默认值为 `7860`。

##### 注意事项

1. **升级 Gradio 库**：确保已升级 Gradio 库以支持该功能。

2. **本地主机访问**：默认配置 `host='127.0.0.1'` 仅允许本机访问。如果希望其他设备也能访问，需要进行以下修改：

   - **修改方式1**：在命令行中输入 `ipconfig`，查找本机的局域网 IP 地址，并将其填入 `host` 参数中。例如：

     ```
     python
     复制代码
     chatbot.run(host='192.168.1.100', port=7860)
     ```

     这样，同一网段内的设备即可通过该 IP 地址进行访问。

   - **修改方式2**：将 `host` 设置为 `'0.0.0.0'`，表示开放所有外部 IP 访问。例如：

     ```
     python
     复制代码
     chatbot.run(host='0.0.0.0', port=7860)
     ```

     这样，同一局域网内的所有设备都可以通过本机的 IP 地址进行访问。需要注意的是，其他设备访问时需要提前知道本机的 IP 地址。

#### 直接启动网页功能的完整代码

```
from XEdu.LLM import Client
chatbot = Client(provider='moonshot',
               api_key="sk-cjCzE5Oo***K53EVAZTnln") # 密钥省略
# 启动基于网页的聊天机器人服务
client.run(host='127.0.0.1', port=7860)
```

此方法默认只能本机访问，不适合需要在局域网或互联网中共享的场景。且并发处理能力有限，依赖于 Gradio 库的性能。

### 功能五：聊天机器人服务的二次分发功能

还有种更加灵活的用法，使用二次分发功能，通过将聊天机器人的服务部署在一个固定的地址（例如局域网 IP 地址），并在不同的终端或应用中通过该地址进行访问和调用。这种方式通常需要先获取服务的 IP 地址，然后在客户端中使用该地址进行推理或其他操作。

使用示例如下：

1.获取 IP 地址并启动服务

```python
from XEdu.LLM import Client

# 创建一个聊天机器人客户端
client = Client(provider='moonshot',
               api_key="sk-cjCzE5Oo***K53EVAZTnln") # 密钥省略

# 启动基于网页的聊天机器人服务，并指定局域网 IP 地址和端口
client.run()

```

输出示例：

```
Running on local URL: http://10.1.48.23:7863
```

2.在其他设备或应用中访问该服务

在局域网内的其他设备上，可以通过上述获取的 IP 地址和端口访问聊天机器人服务：

```
from XEdu.LLM import Client

# 使用固定的 IP 地址和端口号，使用获取的ip
client = Client(base_url='http://10.1.48.23:7863')

# 进行推理或其他操作
res = client.inference("今天天气怎么样？")
# 返回：今天阳光明媚，xxx...
# 启动可视化服务
client.run()
# Running on local URL: http://10.1.48.23:7864
```

二次分发功能将聊天机器人服务在一个固定的 IP 地址和端口上运行，并允许在同一局域网内的其他设备通过该地址和端口进行访问和调用。这样可以方便地在多个设备之间共享聊天机器人服务。