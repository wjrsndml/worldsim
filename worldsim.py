import worldsimprompt
from openai import AsyncOpenAI,OpenAI
ZHIPU_APIKEY=""
DEEPSEEK_APIKEY=""
import asyncio
from tqdm import tqdm
import pickle
import copy
import random
import os
class WorldSim:
    def __init__(self,prompt_data=worldsimprompt.worldframework):
        self.prompt_data=copy.deepcopy(prompt_data)
        self.a_zhipuclient = AsyncOpenAI(api_key=ZHIPU_APIKEY, base_url="https://open.bigmodel.cn/api/paas/v4")
        self.zhipuclient = OpenAI(api_key=ZHIPU_APIKEY, base_url="https://open.bigmodel.cn/api/paas/v4")
        self.a_deepseekclient = AsyncOpenAI(api_key=DEEPSEEK_APIKEY, base_url="https://api.deepseek.com")
        self.deepseekclient = OpenAI(api_key=DEEPSEEK_APIKEY, base_url="https://api.deepseek.com")
        self.people=[]

    async def generate_person(self,it1,name):
        person1=copy.deepcopy(worldsimprompt.personframework)
        person1["基础信息"]["姓名信息"]["本名"]=name
        keys1=person1.keys()
        try:
            for i in keys1:
                person1= await self._generate_people(person1,i)
                print(f"{it1} {i}")
            # self.people.append(person1)
            return person1
        except Exception as e:
            print(e)
            return it1

    async def generate_people(self,num):
        names=self.generate_name(num)
        print(names)
        tasks = [self.generate_person(i,name=names[i]) for i in range(num)]
        results = await asyncio.gather(*tasks)
        self.people=results


    def generate_name(self,num):
        results=set([])
        guide=worldsimprompt.nameguide
        geshi="{ 'names':['name1','name2','name3','name4'] }"
        msg1=f"世界观:{str(self.prompt_data)} \n\n 以上是世界观内容，\n\n{guide} \n\n以上是名字创作指南\n\n请你根据以上信息，创作一些符合世界观的具体人物的中文名字，至少创作20个名字，并输出格式如下:\n {geshi}"
        response =self.zhipuclient.chat.completions.create(
            model="glm-4-air",
            messages=[
                {"role": "user", "content": msg1}
            ],
            max_tokens=4095,
            response_format = {
        'type': 'json_object'
    },
           stream=False
        )
        names=eval(response.choices[0].message.content)
        results.update(names["names"])
        while len(results)<num:
            msg1=f"世界观:{str(self.prompt_data)} \n\n 请你根据以上世界观内容，\n\n{guide} \n\n以上是名字创作指南\n\n请你根据以上信息，创作一些符合世界观的具体人物的中文名字，并且输出的名字不要与以下名字重复 {str(results)}，\n\n\n至少创作20个名字，并输出格式如下:\n {geshi}"
            response =self.zhipuclient.chat.completions.create(
                model="glm-4-air",
                messages=[
                    {"role": "user", "content": msg1}
                ],
                max_tokens=4095,
                response_format = {
            'type': 'json_object'
        },
            stream=False
            )
            names=eval(response.choices[0].message.content)
            results.update(names["names"])
            print(len(results))
        if len(results) > num:
            results = set(random.sample(results, num))
    
        return list(results)

    def generate_world(self):
        try:
            self._generate_overview()
            self._generate_rule()
            self._generate_social()
            self._generate_civilization()
            self._generate_history()
            self._generate_geography()
            self._generate_culture()
            return True
        except Exception as e:
            print(e)
            return False

    def _generate_overview(self):
        p1=worldsimprompt.worldprompt1
        p1=p1.replace("keyword",self.prompt_data["世界观关键词"])
        response = self.zhipuclient.chat.completions.create(
            model="glm-4-plus",
            messages=[
                {"role": "user", "content": p1
                 }
            ],
            max_tokens=4095,
            response_format = {
        'type': 'json_object'
    },
           stream=False
        )
        result=eval(response.choices[0].message.content)
        self.prompt_data["世界观名称"]=result["世界观名称"]
        self.prompt_data["世界概述"]=result["世界概述"]
        print(result["世界观名称"]+" overview")
    
    def _generate_rule(self):
        response = self.zhipuclient.chat.completions.create(
            model="glm-4-plus",
            messages=[
                {"role": "user", "content": f"世界观名称:{self.prompt_data['世界概述']} \n {self.prompt_data['世界概述']} \n 请你根据以上内容,补充世界规则,输出格式如下:\n{str(self.prompt_data['世界规则'])}"
                 }
            ],
            max_tokens=4095,
            response_format = {
        'type': 'json_object'
    },
           stream=False
        )
        result=eval(response.choices[0].message.content)
        self.prompt_data["世界规则"]=result
        print(self.prompt_data["世界观名称"]+" rule")

    def _generate_social(self):
        response = self.zhipuclient.chat.completions.create(
            model="glm-4-plus",
            messages=[
                {"role": "user", "content": f"世界观名称:{self.prompt_data['世界概述']} \n {self.prompt_data['世界概述']} \n 世界规则: {str(self.prompt_data['世界规则'])} \n 请你根据以上内容,补充这个世界观的社会结构,输出格式如下:\n{str(self.prompt_data['社会结构'])}"
                 }
            ],
            max_tokens=4095,
            response_format = {
        'type': 'json_object'
    },
           stream=False
        )
        result=eval(response.choices[0].message.content)
        self.prompt_data["社会结构"]=result
        print(self.prompt_data["世界观名称"]+" social")

    def _generate_civilization(self):
        response = self.zhipuclient.chat.completions.create(
            model="glm-4-plus",
            messages=[
                {"role": "user", "content": f"世界观名称:{self.prompt_data['世界概述']} \n {self.prompt_data['世界概述']} \n 世界规则: {str(self.prompt_data['世界规则'])} \n 社会结构: {str(self.prompt_data['社会结构'])}\n 请你根据以上内容,补充这个世界观的文明发展,输出格式如下:\n{str(self.prompt_data['文明发展'])}"
                 }
            ],
            max_tokens=4095,
            response_format = {
        'type': 'json_object'
    },
           stream=False
        )
        result=eval(response.choices[0].message.content)
        self.prompt_data["文明发展"]=result
        print(self.prompt_data["世界观名称"]+" civilization")

    def _generate_history(self):
        response = self.zhipuclient.chat.completions.create(
            model="glm-4-plus",
            messages=[
                {"role": "user", "content": f"世界观名称:{self.prompt_data['世界概述']} \n {self.prompt_data['世界概述']} \n 世界规则: {str(self.prompt_data['世界规则'])} \n 社会结构: {str(self.prompt_data['社会结构'])} \n 文明发展: {str(self.prompt_data['文明发展'])}\n 请你根据以上内容,补充这个世界观的历史背景,输出格式如下:\n{str(self.prompt_data['历史背景'])}"
                 }
            ],
            max_tokens=4095,
            response_format = {
        'type': 'json_object'
    },
           stream=False
        )
        result=eval(response.choices[0].message.content)
        self.prompt_data["历史背景"]=result
        print(self.prompt_data["世界观名称"]+" history")

    def _generate_geography(self):
        response = self.zhipuclient.chat.completions.create(
            model="glm-4-plus",
            messages=[
                {"role": "user", "content": f"世界观名称:{self.prompt_data['世界概述']} \n {self.prompt_data['世界概述']} \n 世界规则: {str(self.prompt_data['世界规则'])} \n 社会结构: {str(self.prompt_data['社会结构'])} \n 文明发展: {str(self.prompt_data['文明发展'])}\n   历史背景: {str(self.prompt_data['历史背景'])} \n 请你根据以上内容,补充这个世界观的地理环境,输出格式如下:\n{str(self.prompt_data['地理环境'])}"
                 }
            ],
            max_tokens=4095,
            response_format = {
        'type': 'json_object'
    },
           stream=False
        )
        result=eval(response.choices[0].message.content)
        self.prompt_data["地理环境"]=result
        print(self.prompt_data["世界观名称"]+" geography")

    def _generate_culture(self):
        response = self.zhipuclient.chat.completions.create(
            model="glm-4-plus",
            messages=[
                {"role": "user", "content": f"世界观名称:{self.prompt_data['世界概述']} \n {self.prompt_data['世界概述']} \n 世界规则: {str(self.prompt_data['世界规则'])} \n 社会结构: {str(self.prompt_data['社会结构'])} \n 文明发展: {str(self.prompt_data['文明发展'])}\n   历史背景: {str(self.prompt_data['历史背景'])} \n   地理环境: {str(self.prompt_data['地理环境'])} \n 请你根据以上内容,补充这个世界观的文化体系,输出格式如下:\n{str(self.prompt_data['文化体系'])}"
                 }
            ],
            max_tokens=4095,
            response_format = {
        'type': 'json_object'
    },
           stream=False
        )
        result=eval(response.choices[0].message.content)
        self.prompt_data["文化体系"]=result
        print(self.prompt_data["世界观名称"]+" culture")

    async def _generate_people(self,person,keyword):
        msg1=f"世界观:{str(self.prompt_data)} \n\n人物信息:{str(person)} \n\n 请你根据以上世界观内容和人物信息，创作一个人物的{keyword}，并输出格式如下:\n{str(person[keyword])}"
        response =await self.a_zhipuclient.chat.completions.create(
            model="glm-4-flash",
            messages=[
                {"role": "user", "content": msg1}
            ],
            max_tokens=4095,
            response_format = {
        'type': 'json_object'
    },
           stream=False
        )
        person[keyword]=eval(response.choices[0].message.content)
        return person


async def generate_scp(i, worldsim1):
    response = await worldsim1.a_zhipuclient.chat.completions.create(
        model="glm-4-flash",
        messages=[
            {"role": "user", "content": f"请你仿照scp基金会的写法，创作一篇scp项目，一定要足够奇怪，怪诞，恐怖，创新，新颖，项目编号为{i+1}"}
        ],
        max_tokens=4095,
        stream=False
    )
    return response.choices[0].message.content

async def piliang1():
    sem = asyncio.Semaphore(200)
    worldsim1 = WorldSim()
    resultsall=[]
    with open("scp1.txt", "w", encoding="utf-8") as f:
        for j in tqdm(range(50)):
            tasks = [generate_scp(j*200+i+1, worldsim1) for i in range(200)]
            results = await asyncio.gather(*tasks)
            f.write("\n\n\n\n-------------------------------------------------\n\n\n\n\n".join(results))
            f.write("\n\n\n\n-------------------------------------------------\n\n\n\n\n")



if __name__ == "__main__":
    # asyncio.run(piliang1())
    # w1=WorldSim()
    # w1.generate_world()
    # with open(f"world/修仙1.pkl", 'wb') as file:
    #     pickle.dump(w1.prompt_data, file)
    
    # with open("world/赛博修仙.pkl", 'rb') as file:
    #     w1.prompt_data= pickle.load(file)
    # asyncio.run(w1.generate_people(10))
    # print(str(w1.people[0]))
    # with open("peoples/赛博修仙.pkl", 'wb') as file:
    #     pickle.dump(w1.people, file)

    # import json
    # with open("world/元宇宙 + 位面穿越.pkl", 'rb') as file:
    #    a=pickle.load(file)
    # json.dump(a,open("world/元宇宙 + 位面穿越.json",'w',encoding='utf-8'), ensure_ascii=False)

    # with open("peoples/元宇宙 + 位面穿越.pkl", 'rb') as file:
    #    a=pickle.load(file)
    # json.dump(a,open("peoples/元宇宙 + 位面穿越.json",'w',encoding='utf-8'), ensure_ascii=False)
    # b=1

    for i in worldsimprompt.world_combinations:
        w1=WorldSim()
        w1.prompt_data["世界观关键词"]=i
        a1=True
        if not os.path.exists(f"world/{i}.pkl"):
            a1=w1.generate_world()
            if a1==False:
                continue
        else:
            print(f"{i}已存在,跳过")
        with open(f"world/{i}.pkl", 'wb') as file:
            pickle.dump(w1.prompt_data, file)
        asyncio.run(w1.generate_people(200))
        print(str(w1.people[0]))
        with open(f"peoples/{i}.pkl", 'wb') as file:
            pickle.dump(w1.people, file)

    # worldsim1=WorldSim()
    # worldsim1.prompt_data["世界概述"]="在这个名为\"扭曲镜界\"的世界里，所有生命都以\"镜像对\"的形式存在 —— 每个生物都有一个镜中的自己，而当现实中的生物闭上眼睛时，镜中的自己就会睁开眼睛并获得行动能力，反之亦然。更奇特的是，这个世界的魔法能量来源于\"念力结晶\" —— 当镜像对双方做出完全相反的动作时（比如一个向左跑，一个向右跑），就会在两者之间产生一种特殊的能量结晶，而收集和使用这些结晶的能力，成为了这个世界的主要力量体系，也催生出了一个专门研究如何与镜中自己产生\"和谐对立\"的特殊职业 —— \"镜舞师\"。但这个体系也带来了独特的困境：过度追求力量的人可能会刻意与自己的镜像对立，导致人格分裂，最终任何一方都无法真正获得安宁。"
    # for i in range(10):
    #     response = worldsim1.zhipuclient.chat.completions.create(
    #     model="glm-4-flash",
    #     messages=[
    #         {"role": "user", "content": f"你是一个世界观创作助手，请你为后文的世界观json补充更多内容，{str(worldsim1.prompt_data)}"
             
    #          }
    #     ],
    #     response_format = {
    #     'type': 'json_object'
    # },
    # max_tokens=4095,
    #     stream=False
    # )
    #     worldsim1.prompt_data=eval(response.choices[0].message.content)
    #     print(worldsim1.prompt_data)
    #     print(len(response.choices[0].message.content))


