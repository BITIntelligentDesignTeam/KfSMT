# KfSMT
融合知识的代理模型工具包(Knowledge-fused Surrogate Modeling Toolbox)

# 工具包的特性
在工程问题中，使用代理模型代替昂贵的计算与计算机模拟已被广泛地应用。然而，在设计复杂产品时，由于通过仿真试验或者实物试验获得数据的成本较高，往往只能获得有限的数据。仅利用这些数据构建代理模型，其精度往往达不到设计人员需要的精度。可以从两个思路上去解决这一问题：

添加知识约束：工程设计领域内的代理模型从本质来说是设计问题中变量之间的映射规律。而小数据问题中，过于少的数据量致使了变量之间一些映射关系信息缺失，从而导致构建出的代理模型精度达不到要求。考虑到在工程领域，设计人员在多年的设计过程中积累了一定的知识，即对于设计问题本身的映射规律有着一定的认识，这些知识可以补充因小数据缺失的映射关系信息，从而提高代理模型的精度。

提高数据质量：因为数据获取的成本问题，能够获取数据的数量有限。为了提高代理模型的精度，就要求每一个数据点的质量尽可能的高。可以定义数据的效用函数，利用序贯式采样的思想，最大化每次采点的效用函数，从而获得质量更好的数据点，提高代理模型的精度。 

基于以上的两种思路，构建了融合知识的代理模型工具包。


# KfSMT的说明文档
[Documentation of Knowledge-fused Surrogate Modeling Toolbox](https://kfsmt.readthedocs.io/zh_CN/latest/).

