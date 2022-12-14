from knowledge.KnowledgeBase import KnowledgeBase
import xml.dom.minidom
from abc import ABCMeta, abstractmethod
import os
import matplotlib.pyplot as plt


class ExperienceBase(KnowledgeBase):

    def __init__(self, path):
        super(ExperienceBase, self).__init__(path)

        self.input_type = []
        self.output_type = []

    def readKnowledge(self):

        #不能删
        self.input_type = []
        self.output_type = []

        try:
            file_object = open(self.path, 'r+', encoding="gb2312")
            xmlfile = file_object.read()
            file_object.close()
        except UnicodeDecodeError:
            file_object = open(self.path, 'r+', encoding="utf-8")

            # 将"gb2312"格式转化为"utf-8"格式,针对旋成体系统
            xmlfile = file_object.read()
            file_object.close()
            xmlfile = xmlfile.replace("utf-8", "gb2312")

        # 读取xml文件
        dom = xml.dom.minidom.parseString(xmlfile)
        self.root = dom.documentElement
        self.know_type = self.root.getAttribute("infoType")
        self.knowledge['type'] = self.know_type

        return self.knowledge

    def writeKnowledge(self):
        # 在内存中创建一个空的文档
        self.doc = xml.dom.minidom.Document()
        # 创建一个根节点对象
        self.root = self.doc.createElement('info')
        # 设置根节点的属性
        self.root.setAttribute('infoType', self.type)
        self.root.setAttribute('inputType', self.input_type[0])
        self.root.setAttribute('outputType', self.output_type[0])

        self.doc.appendChild(self.root)

    def visualKnowledge(self):
        pass



