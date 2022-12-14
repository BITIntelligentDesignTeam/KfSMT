from xml_read import getSpaceKnowledge
import numpy as np


# 用于交换列表中的顺序
def swapPositions(list_i, pos1, pos2):
    list_i[pos1], list_i[pos2] = list_i[pos2], list_i[pos1]
    return list_i


def sortSpaceKnowledge(*path):


    #对于一条知识的情况
    if len(path) == 1:

        space = getSpaceKnowledge(path[0])
        print(space)

        Space_list = space['Relation']
        Space_num = len(Space_list)

        # sort_list是最后的输出列表，包含pram_num, pram_name, pram_range, weight
        sort_list = []

        for i in range(Space_num):
            Input_i = Space_list[i]

            pram_name = []
            pram_range = []

            level = Input_i['area_level']
            if level == '较复杂':
                weight = 1.5
            elif level == '很复杂':
                weight = 2.5

            Input_list = Input_i['InputType']
            Rang_list = Input_i['InputRang']

            pram_num = len(Input_list)

            for j in range(len(Input_list)):
                pram_name.append(Input_list[j])
                pram_range.append(Rang_list[j])

            for k in range(len(pram_name)):
                if pram_name[k] == '攻角':
                    swapPositions(pram_name, 0, k)
                    swapPositions(pram_range, 0, k)
                    if pram_name[k] == '马赫数':
                        swapPositions(pram_name, 1, k)
                        swapPositions(pram_range, 1, k)

            sort_list.append([pram_num, pram_name, pram_range, weight])

        return sort_list

    elif len(path) == 2:
        obj = 2
        for i in range(obj):
            excel_path = path[i]
            space = getSpaceKnowledge(excel_path)
            Output = space['OutputType']

            if Output == '法向力':
                Space_list = space['Relation']
                Space_num = len(Space_list)

                # sort_list是最后的输出列表，包含pram_num, pram_name, pram_range, weight
                sort_list1 = []

                for i in range(Space_num):
                    Input_i = Space_list[i]

                    pram_name = []
                    pram_range = []

                    level = Input_i['area_level']

                    #不同重要区域的权重、-
                    if level == '较复杂':
                        weight = 1.5
                    elif level == '很复杂':
                        weight = 2.5

                    Input_list = Input_i['InputType']
                    Rang_list = Input_i['InputRang']

                    pram_num = len(Input_list)

                    for j in range(len(Input_list)):
                        pram_name.append(Input_list[j])
                        pram_range.append(Rang_list[j])

                    for k in range(len(pram_name)):
                        if pram_name[k] == '攻角':
                            swapPositions(pram_name, 0, k)
                            swapPositions(pram_range, 0, k)
                            if pram_name[k] == '马赫数':
                                swapPositions(pram_name, 1, k)
                                swapPositions(pram_range, 1, k)

                    sort_list1.append([pram_num, pram_name, pram_range, weight])

            elif Output == '压心系数':
                Space_list = space['Relation']
                Space_num = len(Space_list)

                # sort_list是最后的输出列表，包含pram_num, pram_name, pram_range, weight
                sort_list2 = []

                for i in range(Space_num):
                    Input_i = Space_list[i]

                    pram_name = []
                    pram_range = []

                    level = Input_i['area_level']
                    if level == '较复杂':
                        weight = 1.5
                    elif level == '很复杂':
                        weight = 2.5

                    Input_list = Input_i['InputType']
                    Rang_list = Input_i['InputRang']

                    pram_num = len(Input_list)

                    for j in range(len(Input_list)):
                        pram_name.append(Input_list[j])
                        pram_range.append(Rang_list[j])

                    for k in range(len(pram_name)):
                        if pram_name[k] == '攻角':
                            swapPositions(pram_name, 0, k)
                            swapPositions(pram_range, 0, k)
                            if pram_name[k] == '马赫数':
                                swapPositions(pram_name, 1, k)
                                swapPositions(pram_range, 1, k)

                    sort_list2.append([pram_num, pram_name, pram_range, weight])

        return sort_list1, sort_list2


def knowledge(x1, x2):

    knowledge_list = sortSpaceKnowledge(r'C:\Users\石磊\Desktop\馥琳姐代码\Source\空间型知识1.txt')
    weight = 1

    for i in range(len(knowledge_list)):

        space = knowledge_list[i]

        if space[0] == 2:
            if space[1] == ['攻角', '马赫数']:

                if (space[2][0][0] <= x1 < space[2][0][1]) and (space[2][1][0] <= x2 < space[2][1][1]):
                    if weight == 1:
                        weight = space[3]
                else:
                    if weight != 1:
                        continue
                    else:
                        weight = 1

    return weight


if __name__ == '__main__':
    a = np.array([0.0])
    ma = np.array([2.0])

    a1 = np.array([2.0])
    ma1 = np.array([3.5])

    knowledge_list = sortSpaceKnowledge(r'C:\Users\石磊\Desktop\馥琳姐代码\Source\空间型知识1.txt')
    print(knowledge_list)

    # print(knowledge(x1=a, x2=ma))
    # print(knowledge(x1=a1, x2=ma1))
