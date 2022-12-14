from knowledge.MappingBase import MappingBase


class MonotonicityKnowledge(MappingBase):

    # 单调型知识

    def __init__(self, path):

        super(MonotonicityKnowledge, self).__init__(path)
        self.type = '单调型'
        self.input_range = []
        self.output_range = None

    def readKnowledge(self):

        self.input_range = []

        super(MonotonicityKnowledge, self).readKnowledge()

        section = self.root.getElementsByTagName('section')
        section_ = section[0]
        input_min = section_.getAttribute("minValue")
        input_max = section_.getAttribute("maxValue")
        mono_type = section_.getAttribute("singleTypeName")
        input_range = [float(input_min), float(input_max)]

        self.input_range.append(input_range)
        self.knowledge['input_range'] = self.input_range

        self.mapping_relation.append(mono_type)

        self.knowledge['mapping_relation'] = self.mapping_relation

        return self.knowledge

    def writeKnowledge(self,
                       input_type: [],
                       output_type: [],
                       input_range: [],
                       mapping_relation: [],
                       convar=[]):

        self.input_type = input_type
        self.output_type = output_type
        self.input_range = input_range
        self.mapping_relation = mapping_relation
        self.convar = convar

        super(MonotonicityKnowledge, self).writeKnowledge()

        nodeSection = self.doc.createElement('section')

        nodeSection.setAttribute('singleTypeName', self.mapping_relation[0])
        nodeSection.setAttribute('minValue', str(self.input_range[0][0]))
        nodeSection.setAttribute('maxValue', str(self.input_range[0][1]))

        self.root.appendChild(nodeSection)

        try:
            fp = open(self.path, 'x')
            self.doc.writexml(fp, indent='\t', addindent='\t', newl='\n', encoding="utf-8")
        except FileExistsError:
            print('该知识路径已被创建过,请勿重复创建！')
        else:
            print(str(self.path) + '创建成功！')

    def visualKnowledge(self):

        print("_" * 75)
        print('知识名称:' + self.path)
        print('知识类型:' + self.type)
        print('变量:' + self.input_type[0])
        print('变量范围:' + str(self.input_range[0]))
        print('性能:' + self.output_type[0])
        print('单调性:' + self.mapping_relation[0])

        super(MonotonicityKnowledge, self).visualKnowledge()


if __name__ == "__main__":
    # know1 = MonotonicityKnowledge("C:\data\CMGMonoKnowledge3.xml")
    #
    # know1.writeKnowledge(input_type=['转台转速'],
    #                      output_type=['径向间隙减少量'],
    #                      input_range=[[0, 5.0]],
    #                      mapping_relation=['单调递增'],
    #                      convar=[{'convar_type': '温度', 'convar_RangeOrValue': 'range', 'convar_range': [-20, 60]},
    #                              {'convar_type': '框架转速', 'convar_RangeOrValue': 'range', 'convar_range': [5, 57.3]}])
    #
    #
    # know1.visualKnowledge()
    # a = know1.readKnowledge()
    # print(a)

    know2 = MonotonicityKnowledge( "C:\data\单调型知识1.txt")
    a = know2.readKnowledge()
    know2.visualKnowledge()

    print(a)
