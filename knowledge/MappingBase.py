from knowledge.ExperienceBase import ExperienceBase


class MappingBase(ExperienceBase):

    def __init__(self, path):

        super(MappingBase, self).__init__(path)

        self.mapping_relation = []

        self.convar = []

    def readKnowledge(self):

        super(MappingBase, self).readKnowledge()
        self.mapping_relation = []

        self.convar = []

        # 读取知识的变量类型
        input_type = self.root.getAttribute("inputType")
        output_type = self.root.getAttribute("outputType")

        self.input_type.append(input_type)
        self.output_type.append(output_type)

        self.knowledge['input_type'] = self.input_type
        self.knowledge['output_type'] = self.output_type

        # 读取知识的协变量信息
        param = self.root.getElementsByTagName('param')

        for i in range(len(param)):

            convar_i = {}
            param_i = param[i]
            convar_type = param_i.getAttribute('relation')
            convar_i['convar_type'] = convar_type
            convar_RangeOrValue = param_i.getAttribute('RangeOrValue')
            convar_i['convar_RangeOrValue'] = convar_RangeOrValue
            if convar_RangeOrValue == "value":
                convar_value = param_i.getAttribute('minValue')
                convar_i['convar_value'] = float(convar_value)
            else:
                convar_min = param_i.getAttribute('minValue')
                convar_max = param_i.getAttribute('maxValue')
                convar_range = [float(convar_min), float(convar_max)]
                convar_i['convar_range'] = convar_range
            self.convar.append(convar_i)

        if self.convar == []:
            self.convar = None

        self.knowledge['convar'] = self.convar

        return self.knowledge

    def writeKnowledge(self):

        # 新建知识的协变量信息

        super(MappingBase, self).writeKnowledge()

        if self.convar != []:
            nodeParams = self.doc.createElement('params')

            ID = 0
            for i in self.convar:
                nodeParam = self.doc.createElement('param')
                nodeParam.setAttribute('id', str(ID))
                nodeParam.setAttribute('relation', i['convar_type'])
                nodeParam.setAttribute('RangeOrValue', i['convar_RangeOrValue'])

                if i['convar_RangeOrValue'] == 'value':
                    nodeParam.setAttribute('minValue', str(i['convar_value']))
                else:
                    nodeParam.setAttribute('minValue', str(i['convar_range'][0]))
                    nodeParam.setAttribute('maxValue', str(i['convar_range'][1]))

                nodeParams.appendChild(nodeParam)
                ID += 1

            self.root.appendChild(nodeParams)

    def visualKnowledge(self):

        num = 1
        if self.convar != None:
            for i in self.convar:

                if i['convar_RangeOrValue'] == 'value':
                    print('协变量' + str(num) + ':  ' + '协变量参数:' + i['convar_type'] + '  取值类型:值' + '  协变量范围:' + str(
                        i['convar_value']))
                else:
                    print('协变量' + str(num) + ':  ' + '协变量参数:' + i['convar_type'] + '  取值类型:范围' + '  协变量范围:' + str(
                        i['convar_range']))
                num += 1
