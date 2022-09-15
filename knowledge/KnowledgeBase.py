from abc import ABCMeta, abstractmethod



class KnowledgeBase(object,metaclass = ABCMeta):


    def __init__(self,path):
        self.type = ''
        self.path = path
        self.knowledge = {}

    @abstractmethod
    def readKnowledge(self):
        pass

    @abstractmethod
    def writeKnowledge(self):
        pass

    @abstractmethod
    def visualKnowledge(self):
        pass




