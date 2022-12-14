# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 12:43:35 2020

@author: pc
"""

import json
#############################################################
class process(object):
#########################################        
    # json格式文件——获取数据
    def processData(path):
        obj = open(path,'r',encoding='utf-8')
        inputData = []
        outputData = []
        for line in obj.readlines():
            dic = json.loads(line)
            t = dic['input'],dic['output']
            inputData.append(t[0])
            outputData.append(t[1])
        obj.close()
        return inputData, outputData
##########################################   
    # json格式文件——获取参数
    def processParam(path):
        obj = open(path,'r',encoding='utf-8')
        param = []
        for line in obj.readlines():
            dic = json.loads(line)
            t = dic['iterationTime'],dic['initialIndividual'],dic['hiddenLayer'],dic['cross'],dic['mutant']
            for i in range(len(t)):
                param.append(t[i])
        obj.close()
        return param
##########################################
    # json格式文件——获取知识
    def processKnow_pre(path):
        obj = open(path,'r',encoding='utf-8')
        inputData = []
        outputData = []
        status = []
        for line in obj.readlines():
            dic = json.loads(line)
            t = dic['inputs'],dic['outputs'],dic['status']
            inputData.append(t[0])
            outputData.append(t[1])
            status.append(t[2])
        obj.close()
        return inputData, outputData, status
#################################################
    # 知识字典获取方法
    def processKnow_data(inputKnow, inputKnow_P,inputKnow_b):
        for i in range(len(inputKnow)):
            know_dict = inputKnow[i]
            inputKnow_P.append([])
            inputKnow_b.append([])
            for key in know_dict:
                inputKnow_P[i].append(key)
                inputKnow_b[i].append(know_dict[key])
        return inputKnow_P,inputKnow_b
#########################################
    # 知识分门别类获取好
    def processKnow(path):
        inputKnow, outputKnow, status = process.processKnow_pre(path)
        inputKnow_P = []
        inputKnow_b = []
        outputKnow_P = []
        outputKnow_b = []
        in_P,in_B = process.processKnow_data(inputKnow, inputKnow_P,inputKnow_b)
        out_P, out_B = process.processKnow_data(outputKnow, outputKnow_P,outputKnow_b)
        return in_P, in_B, out_P, out_B, status
#############################################################
 # 获取路径
path_data = ".\TestData.json"
path_param = "C:\data\TestParam.json"
path_know = "C:\data\TestKnow.json"
##########################################
# #获取内容
#inputData, outputData = process.processData(path_data)
param = process.processParam(path_param)
in_P, in_B, out_P, out_B, status = process.processKnow(path_know)

print(in_P, in_B, out_P, out_B, status)


    
    
    

