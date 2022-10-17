import xml.dom.minidom


def getSpaceKnowledge(know_path):
    know = {}  # 创建知识字典用于储存知识

    # 将使用"gb2312"的xml文件转化为'utf-8'以便后续操作
    file_object = open(know_path, 'r+', encoding='utf-8')

    ori_xml = file_object.read()
    file_object.close()
    pro_xml = ori_xml.replace("utf-8", "gb2312")
    JJ_priority = 0

    # 读取xml文件
    dom = xml.dom.minidom.parseString(pro_xml)
    root = dom.documentElement

    # 知识类型
    know_type = root.getAttribute("infoType")
    know["KnowType"] = know_type

    # 空间型知识的输出类型
    output_type = root.getAttribute("outputType")
    know["OutputType"] = output_type

    # 空间型知识的不同区域
    relation = []

    area = root.getElementsByTagName('area')
    for i in range(len(area)):

        inputType = []
        inputRange = []
        area_i = area[i]

        _area = {}
        _area["ID"] = area_i.getAttribute("id")
        area_type = area_i.getAttribute("areaType")
        area_level = area_i.getAttribute('areaLevel')
        _area["area_type"] = area_type
        _area["area_level"] = area_level


        param = area_i.getElementsByTagName('param')
        for j in range(len(param)):
            param_j = param[j]
            input_type = param_j.getAttribute("input")
            inputType.append(input_type)
            input_min = param_j.getAttribute("minValue")
            input_max = param_j.getAttribute("maxValue")
            input_range = [float(input_min), float(input_max)]
            inputRange.append(input_range)

        _area["InputType"] = inputType
        _area["InputRang"] = inputRange

        relation.append(_area)

    know["Relation"] = relation

    return know


if __name__ == "__main__":
    space = getSpaceKnowledge(r'C:\Users\DELL\Desktop\pythonProject\空间型知识1.txt')
    print(space)
