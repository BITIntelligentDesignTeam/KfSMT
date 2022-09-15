from knowledge.MappingBase import MappingBase


class AttributeKnowledge(MappingBase):
    # 属性型知识

    def __init__(self, path):

        super(AttributeKnowledge, self).__init__(path)

        self.type = '属性型'
        self.input_range = []
        self.output_range = []

    def readKnowledge(self):

        self.input_range = []
        self.output_range = []

        super(AttributeKnowledge, self).readKnowledge()

        input = self.root.getElementsByTagName('input')
        input_ = input[0]
        input_min = input_.getAttribute("minValue")
        input_max = input_.getAttribute("maxValue")
        input_range = [float(input_min), float(input_max)]
        self.input_range.append(input_range)
        self.knowledge['input_range'] = self.input_range

        output = self.root.getElementsByTagName('output')
        output_ = output[0]
        output_min = output_.getAttribute("minValue")
        output_max = output_.getAttribute("maxValue")
        output_range = [float(output_min), float(output_max)]
        self.output_range.append(output_range)
        self.knowledge['output_range'] = self.output_range

        self.mapping_relation = None  # 目前是这样，之后会修改

        return self.knowledge

    def writeKnowledge(self,
                       input_type: [],
                       output_type: [],
                       input_range: [],
                       output_range: [],
                       mapping_relation: [],
                       convar=[]):

        self.input_type = input_type
        self.output_range = output_range
        self.output_type = output_type
        self.input_range = input_range
        self.mapping_relation = mapping_relation
        self.convar = convar

        super(AttributeKnowledge, self).writeKnowledge()

        nodeInput = self.doc.createElement('input')
        nodeInput.setAttribute('minValue', str(self.input_range[0][0]))
        nodeInput.setAttribute('maxValue', str(self.input_range[0][1]))
        self.root.appendChild(nodeInput)

        nodeOutput = self.doc.createElement('output')
        nodeOutput.setAttribute('minValue', str(self.output_range[0][0]))
        nodeOutput.setAttribute('maxValue', str(self.output_range[0][1]))
        self.root.appendChild(nodeOutput)

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
        print('性能范围:' + str(self.output_range[0]))


if __name__ == "__main__":
    know1 = AttributeKnowledge("C:\data\属性型知识5.xml")
    know1.writeKnowledge(input_type=['攻角'],
                         output_type=['法向力'],
                         input_range=[[1.0, 2.5]],
                         output_range=[[2.5, 6.0]],
                         mapping_relation=None)
    know1.visualKnowledge()

    know2 = AttributeKnowledge("C:\data\属性型知识1.txt")
    a = know2.readKnowledge()
    print(a)
    know2.visualKnowledge()
