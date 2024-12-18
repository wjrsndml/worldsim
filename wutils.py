import pickle
from openai import AsyncOpenAI,OpenAI
import asyncio
from typing import List, Optional
import aiohttp
from asyncio import Semaphore
import logging
from datetime import datetime
import os
import requests
import uuid
import random
ZHIPU_APIKEY=""
DEEPSEEK_APIKEY=""


class TokenUsageMonitor:
    def __init__(self, logger):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost_deepseek = 0  # 按实际价格配置
        self.total_cost_zhipu = 0     # 按实际价格配置
        self.lock = asyncio.Lock()
        self.logger = logger

    async def add_usage(self, model: str, usage_info: dict):
        async with self.lock:
            prompt_tokens = usage_info.prompt_tokens
            completion_tokens = usage_info.completion_tokens
            
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            
            # 根据不同模型计算成本（价格需要按实际配置）
            if model == "deepseek":
                cost = (prompt_tokens * 0.001 + completion_tokens * 0.002) / 1000  # 示例价格
                self.total_cost_deepseek += cost
            elif model == "zhipuai":
                cost = (prompt_tokens * 0.001 + completion_tokens * 0.001) / 1000  # 示例价格
                self.total_cost_zhipu += cost

            self.logger.info(
                f"Token使用统计:\n"
                f"当前请求: prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}\n"
                f"总计: prompt_tokens={self.total_prompt_tokens}, completion_tokens={self.total_completion_tokens}\n"
                f"总成本: DeepSeek={self.total_cost_deepseek:.4f}元, ZhipuAI={self.total_cost_zhipu:.4f}元"
            )

    def get_summary(self):
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "total_cost_deepseek": self.total_cost_deepseek,
            "total_cost_zhipu": self.total_cost_zhipu,
            "total_cost": self.total_cost_deepseek + self.total_cost_zhipu
        }

def setup_logging(filename=None):
    if filename is None:
        # 如果没有提供文件名，使用时间戳创建
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'data_generation_{timestamp}.log'
    
    # 创建logs目录（如果不存在）
    os.makedirs('logs', exist_ok=True)
    log_path = os.path.join('logs', filename)
    
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    return logging.getLogger()

def get_news(str1):
    api_key = ZHIPU_APIKEY
    msg = [
        {
            "role": "user",
            "content":str1
        }
    ]
    tool = "web-search-pro"
    url = "https://open.bigmodel.cn/api/paas/v4/tools"
    request_id = str(uuid.uuid4())
    data = {
        "request_id": request_id,
        "tool": tool,
        "stream": False,
        "messages": msg
    }

    resp = requests.post(
        url,
        json=data,
        headers={'Authorization': api_key},
        timeout=300
    )
    return resp.content.decode()

def extract_bigbraces_content(s):
    """
    提取从第一个 '[' 到最后一个 ']' 之间的内容，包括这两个大括号。
    :param s: 输入字符串
    :return: 提取的字符串内容
    """
    # 查找第一个 '{' 的索引
    start = s.find('{')
    # 查找最后一个 '}' 的索引
    end = s.rfind('}')
    
    # 检查是否找到大括号
    if start != -1 and end != -1 and start < end:
        return s[start:end + 1]  # 包括两个大括号
    else:
        return None  # 如果未找到合适的大括号对，返回空字符串
def extract_braces_content(s):
    """
    提取从第一个 '[' 到最后一个 ']' 之间的内容，包括这两个大括号。
    :param s: 输入字符串
    :return: 提取的字符串内容
    """
    # 查找第一个 '{' 的索引
    start = s.find('[')
    # 查找最后一个 '}' 的索引
    end = s.rfind(']')
    
    # 检查是否找到大括号
    if start != -1 and end != -1 and start < end:
        return s[start:end + 1]  # 包括两个大括号
    else:
        return "[]"  # 如果未找到合适的大括号对，返回空字符串
def save_object_to_file(obj, file_path):
    """
    将Python对象保存到本地文件。

    参数:
    obj (object): 要保存的Python对象。
    file_path (str): 文件路径，包括文件名。
    """
    with open(file_path, 'wb') as file:
        pickle.dump(obj, file)

def load_object_from_file(file_path):
    """
    从本地文件读取Python对象。

    参数:
    file_path (str): 文件路径，包括文件名。

    返回:
    object: 从文件中读取的Python对象。
    """
    with open(file_path, 'rb') as file:
        return pickle.load(file)
    
class wstructure():
    def __init__(self,name):
        self.name=name
        self.introduce=""
        self.component=[]
        self.data=100
    
    def get_all_node_name(self):
        return [node.name for node in self.component]
    def get_all_node_namestr(self):
        return ",".join(self.get_all_node_name())
        
def generate_one_free(messages):
    client = OpenAI(api_key=ZHIPU_APIKEY, base_url="https://open.bigmodel.cn/api/paas/v4")

    response = client.chat.completions.create(
        model="glm-4-flash",
        messages=messages,
        stream=False
    )
    return response.choices[0].message.content      



async def generatedata(source,depth,filename):
    client = AsyncOpenAI(api_key=DEEPSEEK_APIKEY, base_url="https://api.deepseek.com")
    client2 = AsyncOpenAI(api_key=ZHIPU_APIKEY, base_url="https://open.bigmodel.cn/api/paas/v4")
    sem = Semaphore(180)
    sourceresult=wstructure(source)
    sourceresult = wstructure(source)
    logger = setup_logging()
    token_monitor = TokenUsageMonitor(logger)
    async def make_api_request(api_func, messages, api_name):
        try:
            async with sem:
                response = await api_func(messages)
                if response and hasattr(response, 'usage'):
                        # 记录 token 使用情况
                        await token_monitor.add_usage(
                            api_name.lower(),
                            response.usage
                        )
                return response
        except Exception as e:
            logger.error(f"{api_name} API Error: {str(e)}")
            return None

    async def make_deepseek_request(messages):
        return await make_api_request(
            lambda m: client.chat.completions.create(
                model="deepseek-chat",
                messages=m,
                stream=False
            ),
            messages,
            "DeepSeek"
        )

    async def make_zhipu_request(messages):
        return await make_api_request(
            lambda m: client2.chat.completions.create(
                model="glm-4-air",
                messages=m,
                top_p=0.7,
                temperature=0.95,
                max_tokens=4095,
                stream=False
            ),
            messages,
            "ZhipuAI"
        )

    async def process_item(item, prefix, depth):
        result1 = wstructure(item)
        new_prefix = f"{prefix}的{item}"
        
        try:
            logger.info(f"Processing: {new_prefix}")
            
            # 同时发起两个API请求
            component_messages = [
                {
                    "role": "system",
                    "content": "你是一个乐于解答各种问题的助手，你的任务是为用户提供专业、准确、有见地的建议。"
                },
                {
                    "role": "user",
                    "content": f"{new_prefix}由哪些部分组成，请你用中文列举出来，列举出的内容尽可能不要重叠，不遗漏，不要有空间或者物理上的包含关系。例如，如果要列出人体由哪些部分组成，不要同时列出躯干，和肾脏，因为肾脏包含在躯干中。空间上和物理上不重复的可以同时列出，例如左脚和右脚。将列举出的内容写成一个python列表，如果你不知道怎么回答也不需要胡乱回答，只需要输出一个空列表即可。不需要输出其他任何多余的信息："
                }
            ]
            
            function_messages = [
                {
                    "role": "system",
                    "content": "你是一个乐于解答各种问题的助手，你的任务是为用户提供专业、准确、有见地的建议。"
                },
                {
                    "role": "user",
                    "content": f"{new_prefix}的功能和作用是什么，请你尽可能用中文简洁地说明："
                }
            ]
            
            component_response, function_response = await asyncio.gather(
                make_deepseek_request(component_messages),
                make_zhipu_request(function_messages)
            )
            
            if component_response and function_response:
                a1=extract_braces_content(component_response.choices[0].message.content)
                if a1=="" or a1==None:
                    answer=[]
                else:
                    answer = eval(a1)
                intro1 = function_response.choices[0].message.content
                
                logger.info(f"深度: {depth} | 前缀: {new_prefix} | 组成部分: {answer}")
                logger.info(f"深度: {depth} | 前缀: {new_prefix} | 功能描述: {intro1}")
                
                result1.introduce = intro1
                
                # 并发处理所有子项
                if depth > 0:
                    sub_tasks = [process_item(sub_item, new_prefix, depth - 1) for sub_item in answer]
                    result1.component = await asyncio.gather(*sub_tasks)
                    # 过滤掉None结果
                    result1.component = [r for r in result1.component if r is not None]
                
                return result1
            else:
                logger.warning(f"跳过 {new_prefix} - API调用失败")
                return None
                
        except Exception as e:
            logger.error(f"处理 {new_prefix} 时发生错误: {str(e)}")
            return None

    async def process_initial_request():
        try:
            logger.info(f"开始处理源数据: {sourceresult.name}")
            initial_response = await make_deepseek_request([
                {
                    "role": "system",
                    "content": "你是一个乐于解答各种问题的助手，你的任务是为用户提供专业、准确、有见地的建议。"
                },
                {
                    "role": "user",
                    "content": f"{sourceresult.name}由哪些部分组成，请你用中文列举出来，列举出的内容尽可能不要重叠，不遗漏，不要有空间或者物理上的包含关系。例如，如果要列出人体由哪些部分组成，不要同时列出躯干，和肾脏，因为肾脏包含在躯干中。空间上和物理上不重复的可以同时列出，例如左脚和右脚。将列举出的内容写成一个python列表，不需要输出其他任何多余的信息："
                }
            ])
            
            if initial_response:
                answer = eval(extract_braces_content(initial_response.choices[0].message.content))
                logger.info(f"初始组成部分: {answer}")
                return answer
            else:
                logger.error("初始API调用失败")
                return None
                
        except Exception as e:
            logger.error(f"初始API调用时发生错误: {str(e)}")
            return None

    # 主处理流程
    initial_answer = await process_initial_request()
    if initial_answer:
        depth=depth-1
        # 并发处理所有顶级项目
        tasks = [process_item(item, sourceresult.name, depth) for item in initial_answer]
        results = await asyncio.gather(*tasks)
        # 过滤掉None结果
        sourceresult.component = [r for r in results if r is not None]
        
        save_object_to_file(sourceresult, filename)
        # 输出最终统计信息
        summary = token_monitor.get_summary()
        logger.info(
            f"\n===== 总体统计 =====\n"
            f"总提示词tokens: {summary['total_prompt_tokens']}\n"
            f"总完成词tokens: {summary['total_completion_tokens']}\n"
            f"tokens总量: {summary['total_tokens']}\n"
            f"DeepSeek成本: {summary['total_cost_deepseek']:.4f}元\n"
            f"ZhipuAI成本: {summary['total_cost_zhipu']:.4f}元\n"
            f"总成本: {summary['total_cost']:.4f}元"
        )
        logger.info(f"数据生成完成，已保存到文件: {filename}")

async def asygenerate_one_free(messages):
    client = AsyncOpenAI(api_key=ZHIPU_APIKEY, base_url="https://open.bigmodel.cn/api/paas/v4")

    response = await client.chat.completions.create(
        model="glm-4-flash",
        messages=messages,
        stream=False
    )
    return response.choices[0].message.content     

async def generate_datafree(source,depth,filename):
    client = AsyncOpenAI(api_key=ZHIPU_APIKEY, base_url="https://open.bigmodel.cn/api/paas/v4")
    sem = Semaphore(180)
    sourceresult=wstructure(source)
    sourceresult = wstructure(source)
    logger = setup_logging()
    # monitor = ConcurrencyMonitor(logger)
    token_monitor = TokenUsageMonitor(logger)
    async def make_api_request(api_func, messages, api_name):
        try:
            # await monitor.increment()
            async with sem:
                response = await api_func(messages)
                if response and hasattr(response, 'usage'):
                        # 记录 token 使用情况
                        await token_monitor.add_usage(
                            api_name.lower(),
                            response.usage
                        )
                return response
        except Exception as e:
            logger.error(f"{api_name} API Error: {str(e)}")
            return None
        # finally:
        #     await monitor.decrement()

    async def make_zhipu_request(messages):
        return await make_api_request(
            lambda m: client.chat.completions.create(
                model="glm-4-flash",
                messages=m,
                top_p=0.7,
                temperature=0.95,
                max_tokens=4095,
                stream=False
            ),
            messages,
            "ZhipuAI"
        )

    async def process_item(item, prefix, depth):
        result1 = wstructure(item)
        new_prefix = f"{prefix}的{item}"
        
        try:
            logger.info(f"Processing: {new_prefix}")
            
            # 同时发起两个API请求
            component_messages = [
                {
                    "role": "system",
                    "content": "你是一个乐于解答各种问题的助手，你的任务是为用户提供专业、准确、有见地的建议。"
                },
                {
                    "role": "user",
                    "content": f"{new_prefix}由哪些部分组成，请你用中文列举出来，列举出的内容尽可能不要重叠，不遗漏，不要有空间或者物理上的包含关系。例如，如果要列出人体由哪些部分组成，不要同时列出躯干，和肾脏，因为肾脏包含在躯干中。空间上和物理上不重复的可以同时列出，例如左脚和右脚。将列举出的内容写成一个python列表，如果你不知道怎么回答也不需要胡乱回答，只需要输出一个空列表即可。不需要输出其他任何多余的信息："
                }
            ]
            
            function_messages = [
                {
                    "role": "system",
                    "content": "你是一个乐于解答各种问题的助手，你的任务是为用户提供专业、准确、有见地的建议。"
                },
                {
                    "role": "user",
                    "content": f"{new_prefix}的功能和作用是什么，请你尽可能用中文简洁地说明："
                }
            ]
            
            component_response, function_response = await asyncio.gather(
                make_zhipu_request(component_messages),
                make_zhipu_request(function_messages)
            )
            
            if component_response and function_response:
                a1=extract_braces_content(component_response.choices[0].message.content)
                if a1=="[]" or a1==None:
                    answer=[]
                else:
                    answer = eval(a1)
                intro1 = function_response.choices[0].message.content
                
                logger.info(f"深度: {depth} | 前缀: {new_prefix} | 组成部分: {answer}")
                logger.info(f"深度: {depth} | 前缀: {new_prefix} | 功能描述: {intro1}")
                
                result1.introduce = intro1
                
                # 并发处理所有子项
                if depth > 0:
                    sub_tasks = [process_item(sub_item, new_prefix, depth - 1) for sub_item in answer]
                    result1.component = await asyncio.gather(*sub_tasks)
                    # 过滤掉None结果
                    result1.component = [r for r in result1.component if r is not None]
                
                return result1
            else:
                logger.warning(f"跳过 {new_prefix} - API调用失败")
                return None
                
        except Exception as e:
            logger.error(f"处理 {new_prefix} 时发生错误: {str(e)}")
            return None

    async def process_initial_request():
        try:
            logger.info(f"开始处理源数据: {sourceresult.name}")
            initial_response = await make_zhipu_request([
                {
                    "role": "system",
                    "content": "你是一个乐于解答各种问题的助手，你的任务是为用户提供专业、准确、有见地的建议。"
                },
                {
                    "role": "user",
                    "content": f"{sourceresult.name}由哪些部分组成，请你用中文列举出来，列举出的内容尽可能不要重叠，不遗漏，不要有空间或者物理上的包含关系。例如，如果要列出人体由哪些部分组成，不要同时列出躯干，和肾脏，因为肾脏包含在躯干中。空间上和物理上不重复的可以同时列出，例如左脚和右脚。将列举出的内容写成一个python列表，不需要输出其他任何多余的信息："
                }
            ])
            
            if initial_response:
                answer = eval(extract_braces_content(initial_response.choices[0].message.content))
                logger.info(f"初始组成部分: {answer}")
                return answer
            else:
                logger.error("初始API调用失败")
                return None
                
        except Exception as e:
            logger.error(f"初始API调用时发生错误: {str(e)}")
            return None

    # 主处理流程
    initial_answer = await process_initial_request()
    if initial_answer:
        depth=depth-1
        # 并发处理所有顶级项目
        tasks = [process_item(item, sourceresult.name, depth) for item in initial_answer]
        results = await asyncio.gather(*tasks)
        # 过滤掉None结果
        sourceresult.component = [r for r in results if r is not None]
        
        save_object_to_file(sourceresult, filename)
        # 输出最终统计信息
        summary = token_monitor.get_summary()
        logger.info(
            f"\n===== 总体统计 =====\n"
            f"总提示词tokens: {summary['total_prompt_tokens']}\n"
            f"总完成词tokens: {summary['total_completion_tokens']}\n"
            f"tokens总量: {summary['total_tokens']}\n"
            f"DeepSeek成本: {summary['total_cost_deepseek']:.4f}元\n"
            f"ZhipuAI成本: {summary['total_cost_zhipu']:.4f}元\n"
            f"总成本: {summary['total_cost']:.4f}元"
        )
        logger.info(f"数据生成完成，已保存到文件: {filename}")
        # logger.info(f"最终峰值并发数: {monitor.max_tasks}")
        
def generate_prompt(sys1,str1):
    return [
                {
                    "role": "system",
                    "content": sys1
                },
                {
                    "role": "user",
                    "content": str1
                }
    ]


def recursive_task_processor(syspro,miaoshu,syspro2,ws1):
    longprompt=""
    tasks=[ws1]
    while tasks:
        newtasks=[]
        print(len(tasks))
        for i in tasks:
            # if i.name in result:
            #     rnum=random.randint(1,100)
            #     i.data-=rnum
            #     longprompt+=f"{i.name}的数据减少了{rnum}，剩余{i.data}。\n"
            #     tasks.append(i)
            prompt=generate_prompt(syspro,miaoshu+i.get_all_node_namestr())
            result=generate_one_free(prompt)
            print(result)
            result=eval(extract_braces_content(result))
            for j in i.component:
                if j.name in result:
                    newtasks.append(j)
                    rnum=random.randint(1,100)
                    if not hasattr(j,"data"):
                        j.data=100
                    j.data-=rnum
                    longprompt+=f"{j.name}的数据减少了{rnum}，剩余{j.data}。\n"
        
        tasks=newtasks
    print(longprompt)
    prompt=generate_prompt(syspro2,longprompt)
    result=generate_one_free(prompt)
    return result

async def async_recursive_task_processor(syspro, miaoshu, syspro2, ws1):
    # 创建信号量限制并发数
    sem = Semaphore(180)
    longprompt = ""
    tasks = [ws1]
    
    async def process_single_task(task):
        async with sem:  # 使用信号量控制并发
            prompt = generate_prompt(syspro, miaoshu + task.get_all_node_namestr())
            result = await asygenerate_one_free(prompt)
            print(result)
            return task, eval(extract_braces_content(result))
    
    while tasks:
        # 创建所有任务的协程
        coroutines = [process_single_task(task) for task in tasks]
        
        # 并发执行所有任务
        results = await asyncio.gather(*coroutines)
        
        newtasks = []
        # 处理结果
        for task, result in results:
            for component in task.component:
                if component.name in result:
                    newtasks.append(component)
                    rnum = random.randint(1, 100)
                    if not hasattr(component, "data"):
                        component.data = 100
                    component.data -= rnum
                    longprompt += f"{component.name}的数据减少了{rnum}，剩余{component.data}。\n"
        
        tasks = newtasks
        print(f"当前任务数: {len(tasks)}")
    
    print(longprompt)
    # 最终处理
    prompt = generate_prompt(syspro2, longprompt)
    result = await asygenerate_one_free(prompt)
    return result       
    
  
if __name__ == "__main__":
    # async def main():
    #     await generatedata("人体",4,"human.pkl")
    # asyncio.run(main())
    test1=load_object_from_file("human.pkl")
    miaoshu="李雷打了韩梅梅一拳"
    syspro="""你是一个细节描写生成器。当用户输入一个简单的动作描述时，你需要：

准确描述动作的具体部位（例如：脸部、手臂、腹部等）
描述动作造成的直接物理反应或伤害（例如：淤青、流血、疼痛等）
用简短的一句话完成描述
保持客观，不加入情节发展或其他角色

示例：
用户输入：小明推了小红一下
输出：小明用力推搡小红的后背，导致她踉跄几步摔在了地上，膝盖磕破了皮。"""
    prompt=generate_prompt(syspro,miaoshu)
    result=generate_one_free(prompt)
    print(result)
    sys1="""
    你是一个动作描述词汇分析器。你的任务是：

接收用户输入的动作描述（例如：'打人'）
根据这个动作描述，根据下方给定的词汇列表，请精确地选出0-2个其中符合条件（宁少勿多，匹配一定要精确），会被动作影响到的词汇。并且不要输出其他多余的信息，将结果写成一个python列表。（例如：['结果A', '结果B']）

    """
    sys2="""
    你是一个专业的医疗损伤描述生成器。当用户输入一系列身体部位的数值变化数据时，你需要将这些零散的数据整合成一段连贯的医疗损伤报告。
规则：

数值规则：

范围：0-100（100为完好，0为最严重损伤）
90-100：几乎无伤
70-89：轻度损伤
50-69：中度损伤
30-49：重度损伤
0-29：危重损伤
0-10：致命损伤


描述要素：

将相近部位的损伤组合描述
说明损伤的具体表现（如骨裂、断裂、挫伤等）
描述对身体功能的影响
相互关联的损伤要联系起来描述
按照损伤程度从重到轻排序描述


输出格式：
第一段：概述最严重的损伤部位和整体状况
第二段：详细描述各个损伤部位的具体情况
第三段：描述这些损伤对身体功能的综合影响

示例输入：
髁突头的数据减少了32，剩余68。
肋骨结节的数据减少了93，剩余7。
示例输出：
检查发现患者肋骨区域存在致命性损伤，同时下颌关节也受到明显创伤。最危急的是肋骨结节处出现严重的粉碎性骨折，局部组织几近崩溃，存在刺穿内脏的风险。
髁突头区域呈现中度损伤状态，关节囊受到撕裂，导致张口和咀嚼功能受限，局部软组织肿胀明显。肋骨结节处的创伤异常严重，骨片移位明显，伴有大面积软组织损伤，内出血情况危急。
这些损伤的综合作用导致患者呼吸困难，无法正常进食，需要立即进行手术治疗，否则可能危及生命。
    """
    result=""
    async def main():
        result=""
        result=await async_recursive_task_processor(sys1,miaoshu+result+"\n",sys2,test1)
        print(result)
    asyncio.run(main())