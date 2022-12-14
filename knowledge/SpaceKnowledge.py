from knowledge.Experience import ExperienceBase

import xml.dom.minidom


class SpaceKnowledge(ExperienceBase):

    def __init__(self, path):

        super(SpaceKnowledge, self).__init__(path)
        self.type = '空间型'
        self.space_relation = []

    def readKnowledge(self):

        super(SpaceKnowledge, self).readKnowledge()
        self.input_type = []
        self.space_relation = []
        self.output_type = []

        # 空间型知识的输出类型
        output_type = self.root.getAttribute("outputType")
        self.knowledge["output_type"] = output_type

        self.output_type = [output_type]

        # 空间型知识的输入类型
        input_num = self.root.getAttribute("inputNum")
        for i in range(int(input_num)):
            self.input_type.append(self.root.getAttribute("inputType" + str(i + 1)))

        self.knowledge["input_type"] = self.input_type

        area = self.root.getElementsByTagName('area')
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

            _area["input_type"] = inputType
            _area["input_range"] = inputRange

            self.space_relation.append(_area)

        self.knowledge["space_relation"] = self.space_relation

        return self.knowledge

    def visualKnowledge(self):
        """
        查看知识
        :return:
        """
        print("_" * 75)
        print('知识名称:' + self.path)
        print('知识类型:' + self.type)
        print('变量:' + self.input_type[0])
        print('性能:' + self.output_type[0])

        for i in range(len(self.space_relation)):
            print("空间" + str(i + 1) + ":")
            print("空间" + str(i + 1) + "类型:" + self.space_relation[i]['area_type'])
            print("空间" + str(i + 1) + "程度:" + self.space_relation[i]['area_level'])
            for j in range(len(self.space_relation[i]['input_type'])):
                print("空间" + str(i + 1) + "变量" + str(j + 1) + ":" + str(self.space_relation[i]['input_type'][j]))
                print("空间" + str(i + 1) + "范围" + str(j + 1) + ":" + str(self.space_relation[i]['input_range'][j]))

    def writeKnowledge(self,
                       input_type: [],
                       output_type: [],
                       space_relation: []
                       ):
        """

        :param input_type: 空间型知识的输入参数名称
        :param output_type: 空间型知识的输出参数名称
        :param space_relation: 空间型知识的空间关系，示例：[{'ID': '1', 'area_type': '复杂区域', 'area_level': '较复杂', 'input_type': ['攻角', '马赫数'], 'input_range': [[0.0, 10.0], [3.0, 4.0]]},
                                                         {'ID': '2', 'area_type': '复杂区域', 'area_level': '很复杂', 'input_type': ['攻角', '马赫数'], 'input_range': [[0.0, 0.0], [3.0, 3.0]]}]
        :return:
        """

        self.input_type = input_type
        self.output_type = output_type
        self.space_relation = space_relation

        self.doc = xml.dom.minidom.Document()
        # 创建一个根节点对象
        self.root = self.doc.createElement('info')
        # 设置根节点的属性
        self.root.setAttribute('infoType', self.type)

        inputNum = len(self.input_type)
        self.root.setAttribute('inputNum', str(inputNum))

        for i in range(inputNum):
            self.root.setAttribute('inputType' + str(i + 1), self.input_type[i])

        self.root.setAttribute('outputType', self.output_type[0])

        self.doc.appendChild(self.root)

        # 设置area节点属性
        for i in range(len(self.space_relation)):

            nodeArea = self.doc.createElement('area')

            nodeArea.setAttribute('id', str(i + 1))
            nodeArea.setAttribute('areaLevel', self.space_relation[i]["area_level"])
            nodeArea.setAttribute('areaType', self.space_relation[i]["area_type"])

            # 设置param节点属性
            for j in range(len(self.space_relation[i]['input_type'])):
                nodeParam = self.doc.createElement('param')

                nodeParam.setAttribute('id', str(j + 1))
                nodeParam.setAttribute('input', self.space_relation[i]["input_type"][j])
                nodeParam.setAttribute('minValue', str(self.space_relation[i]["input_range"][j][0]))
                nodeParam.setAttribute('maxValue', str(self.space_relation[i]["input_range"][j][1]))

                nodeArea.appendChild(nodeParam)

            self.root.appendChild(nodeArea)

        try:
            fp = open(self.path, 'x')
            self.doc.writexml(fp, indent='\t', addindent='\t', newl='\n', encoding="utf-8")
        except FileExistsError:
            print('该知识路径已被创建过,请勿重复创建！')
        else:
            print(str(self.path) + '创建成功！')


if __name__ == "__main__":
    s = SpaceKnowledge(r'C:\data\新空间型知识2.txt')
    space = s.readKnowledge()
    s.visualKnowledge()
    print(space)

    s2 = SpaceKnowledge(r'C:\data\新空间型知识2.txt')
    s2.writeKnowledge(input_type=['马赫数', '攻角'],
                      output_type=['压心系数'],
                      space_relation=[{'ID': '1', 'area_type': '复杂区域', 'area_level': '较复杂', 'input_type': ['攻角', '马赫数'],
                                       'input_range': [[0.0, 10.0], [3.0, 4.0]]},
                                      {'ID': '2', 'area_type': '复杂区域', 'area_level': '很复杂', 'input_type': ['攻角', '马赫数'],
                                       'input_range': [[0.0, 0.0], [3.0, 3.0]]}]
                      )
    # s2.visualKnowledge()
    # k = s2.readKnowledge()
    # print(k)
