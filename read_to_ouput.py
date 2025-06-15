import json
import random
from collections import defaultdict
from MahjongGB import MahjongFanCalculator
from feature_test import FeatureAgent  # 这个是自己经过修改后的文件，原文件保留
from agent import MahjongGBAgent


# 该函数的作用是从单个比赛文件中提取每一个json文件
def extract_json_objects(text):
    decoder = json.JSONDecoder()
    idx = 0
    results = []
    while idx < len(text):
        try:
            obj, idx = decoder.raw_decode(text, idx)
            results.append(obj)
        except json.JSONDecodeError:
            idx += 1
    return results






# 该函数的作用是读取目标文件，并且输出为一个包含json文件的列表，其中每一个json文件都是一场比赛
def read_to_out(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    json_objects = extract_json_objects(content)
    print(f"共找到 {len(json_objects)} 个 JSON 对象：")
    return json_objects






# 将一局比赛的其中一个player的request（也就是需要的数据提取出来）
def a_player_a_match(the_player, the_match):
    epsion_data = []
    n = len(the_match['log'])
    for i in range(n):
        if i%2==0 :
           t = the_match['log'][i]['output']['content'][str(the_player)]
           epsion_data.append(t)
    return epsion_data





# 对得到一个player的一场比赛的数据进行处理，得到可以使用的数据
def a_player_true_data(epsion_data, my_agent_num):
    frist = epsion_data[0]
    frist.split()
    frist = str(frist[0]) +' '+ str(frist[2])

    epsion_data = epsion_data[1:]
    two = epsion_data[0].split()
    two = two[5:]

    k = str(1)
    for tile in two:
        if tile.startswith('H'):  # 忽略花牌
           continue
        if tile not in FeatureAgent.OFFSET_TILE:  # 检查是否是合法牌
           print(f"Invalid tile ignored: {tile}")
           continue
        k +=' ' + tile
    
    epsion_data= [frist]+[k] + epsion_data[1:]
    epsion_data_copy=[]
    my_n = len(epsion_data)
    
    for i in range(my_n-1):  # 这一步的处理是处理暗杠，杠，吃，碰的逻辑行为（使其与样例输入相同）
      my_split = epsion_data[i].split()
    
      if len(my_split) >=3 and my_split[2]=='CHI':
          my_str1 = my_split[0]+' '+ my_split[1] + ' '+ my_split[2]+' '+my_split[3]
          my_str2 = my_split[0]
          my_str2 += ' ' +  my_split[1] + ' ' + 'PLAY' +' ' + my_split[4]
          epsion_data_copy.append(my_str1)
          epsion_data_copy.append(my_str2)
          continue
      if len(my_split) >=3 and my_split[2]=='PENG':
          my_str1 = my_split[0]+' '+ my_split[1] + ' '+ my_split[2]
          my_str2 = my_split[0]
          my_str2 += ' ' +  my_split[1] + ' ' + 'PLAY' +' ' + my_split[3]
          epsion_data_copy.append(my_str1)
          epsion_data_copy.append(my_str2)
          continue

      if len(my_split) >=3 and my_split[1] != my_agent_num and i+1 < my_n-2 and my_split[2] == 'DRAW':
          my_split2 = epsion_data[i+1].split()
          if len(my_split2)>=3 and my_split[1]==my_split2[1] and my_split2[2] =='GANG':
              epsion_data_copy.append(epsion_data[i])
              i+=1
              my_str = my_split2[0] + ' ' + my_split2[1] + 'ANGANG'
              epsion_data_copy.append(my_str)
              continue

      if my_split[0] == '2' and i+1 < my_n-2:
          my_split2 = epsion_data[i+1].split()
          if len(my_split2)>=3 and my_split2[1]==str(my_agent_num) and my_split2[2] =='GANG':
              epsion_data_copy.append(epsion_data[i])
              i+=1
              the_str = my_split2[0] + ' ' + my_split2[1] + 'ANGANG' + ' ' + my_split[1]
              epsion_data_copy.append(the_str)
              continue
      epsion_data_copy.append(epsion_data[i])
    return epsion_data_copy




def the_player(epsion_data, the_player):
    my_agent = FeatureAgent(the_player)
    n = len(epsion_data)
    for i in range(n-1):
        my_agent.request2obs(epsion_data[i])   #到此为止就可以进行不断输出该智能体的状态和动作掩码了
        print(my_agent.fanshu - my_agent.last_fanshu) # 调用这个可以输出每个动作后番数的增减(有些错误，我明天看一下)
        print(my_agent.obs) # 输出现有的手牌和动作掩码，与样例中一致
        print(my_agent.fanshu) # 输出现有的番数









# 以下为使用样例

file_path = r"C:\Users\64809\Desktop\人工智能\6.多智能体\大作业\GBMJ\data\1-100.matches"

json_objects = read_to_out(file_path)



the_iter = iter(json_objects)

one_match = next(the_iter)

player = 1

epsion_data = a_player_a_match(player, one_match)

print(epsion_data)

the_player_epsion_data = a_player_true_data(epsion_data, player)

print(the_player_epsion_data)

the_player(the_player_epsion_data,player)




