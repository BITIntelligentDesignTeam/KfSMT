

def checkDataKnow(dataSet,knowlist):
    """
    检查知识和数据是否含有相同的输入和输出变量
    :param dataSet: 数据集
    :param knowlist: 知识dict组成的列表
    :return: 通过检测的知识列表和数据列表
    """
    inputTitle = dataSet["title"][0]
    outputTitle = dataSet["title"][1]

    #如果知识的输入或者输出不在数据的输入之中，将该条知识从列表中删除
    for i in knowlist:
        for j in i["input_type"]:
            if j not in inputTitle:
                knowlist.remove(i)
                break

        for j in i["output_type"]:
            if j not in inputTitle:
                knowlist.remove(i)
                break

    return dataSet,knowlist




a=[[1,2],[1,6]]
b=[1,2,3,4]
for i in a:
    for j in i :
        if j not in b:
            a.remove(i)
            break

print(a)


