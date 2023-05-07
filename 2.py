import matplotlib.pyplot as plt
from math import log
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import tree
import xlrd
import copy


decisionNodeStyle = dict(boxstyle="sawtooth", fc="0.8")
leafNodeStyle = {"boxstyle": "round4", "fc": "0.8"}
arrowArgs = {"arrowstyle": "<-"}


# 画节点
def plotNode(nodeText, centerPt, parentPt, nodeStyle):
    createPlot.ax1.annotate(nodeText, xy=parentPt, xycoords="axes fraction", xytext=centerPt
                            , textcoords="axes fraction", va="center", ha="center", bbox=nodeStyle,
                            arrowprops=arrowArgs)


# 添加箭头上的标注文字
def plotMidText(centerPt, parentPt, lineText):
    xMid = (centerPt[0] + parentPt[0]) / 2.0
    yMid = (centerPt[1] + parentPt[1]) / 2.0
    createPlot.ax1.text(xMid, yMid, lineText)


# 画树
def plotTree(decisionTree, parentPt, parentValue):
    # 计算宽与高
    leafNum, treeDepth = 2, 2       #getTreeSize(decisionTree)
    # 在 1 * 1 的范围内画图，因此分母为 1
    # 每个叶节点之间的偏移量
    plotTree.xOff = plotTree.figSize / (plotTree.totalLeaf - 1)
    # 每一层的高度偏移量
    plotTree.yOff = plotTree.figSize / plotTree.totalDepth
    # 节点名称
    nodeName = list(decisionTree.keys())[0]
    # 根节点的起止点相同，可避免画线；如果是中间节点，则从当前叶节点的位置开始，
    #      然后加上本次子树的宽度的一半，则为决策节点的横向位置
    centerPt = (plotTree.x + (leafNum - 1) * plotTree.xOff / 2.0, plotTree.y)
    # 画出该决策节点
    plotNode(nodeName, centerPt, parentPt, decisionNodeStyle)
    # 标记本节点对应父节点的属性值
    plotMidText(centerPt, parentPt, parentValue)
    # 取本节点的属性值
    treeValue = decisionTree[nodeName]
    # 下一层各节点的高度
    plotTree.y = plotTree.y - plotTree.yOff
    # 绘制下一层
    for val in treeValue.keys():
        # 如果属性值对应的是字典，说明是子树，进行递归调用； 否则则为叶子节点
        if type(treeValue[val]) == dict:
            plotTree(treeValue[val], centerPt, str(val))
        else:
            plotNode(treeValue[val], (plotTree.x, plotTree.y), centerPt, leafNodeStyle)
            plotMidText((plotTree.x, plotTree.y), centerPt, str(val))
            # 移到下一个叶子节点
            plotTree.x = plotTree.x + plotTree.xOff
    # 递归完成后返回上一层
    plotTree.y = plotTree.y + plotTree.yOff


# 画出决策树
def createPlot(decisionTree):
    fig = plt.figure(1, facecolor="white")
    fig.clf()
    axprops = {"xticks": [], "yticks": []}
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    # 定义画图的图形尺寸
    plotTree.figSize = 1.5
    # 初始化树的总大小
    plotTree.totalLeaf, plotTree.totalDepth = 2, 2          #getTreeSize(decisionTree)
    # 叶子节点的初始位置x 和 根节点的初始层高度y
    plotTree.x = 0
    plotTree.y = plotTree.figSize
    plotTree(decisionTree, (plotTree.figSize / 2.0, plotTree.y), "")
    plt.show()


# 无用
# --------------------------------------------------------------------------#
# x = [i / 100 for i in range(1, 101)]
# y = [-log(i, 2) for i in x]
# y = [round(i, 1) for i in y]
# plt.xlabel("p(x)")
# plt.ylabel("l(x)")
# plt.plot(x, y)
# plt.show()
# 生成数据
# data = [
#     [1, 0, 0, '打球'],
#     [1, 0, 1, '打球'],
#     [0, 1, 1, '打球'],
#     [0, 0, 1, '不打球'],
#     [1, 1, 1, '打球']
# ]
# # 将list结构数据转变为DataFrame结构数据
# df1 = pd.DataFrame(data)
# print(df1)
# # 提取目标变量
# target = df.iloc[:, -1]
# # 计算目标变量各个类型的数量,并将数据类型转变为“字典”
# label_counts = target.value_counts()
# label_dict = label_counts.to_dict()
# # 初始化信息熵的值，并计算信息熵
# data_length = len(data)
# entropy = 0
# for key in label_dict:
#     prob = float(label_dict[key]) / data_length
#     entropy -= prob * np.log2(prob)
# print(entropy)

# 1.创建筛选条件并查看结果
# selector = df1.loc[:, 0] == 1
# # 2.筛选数据并查看结果
# df_split = df1[selector]
# # 3.使用groups工具实现数据分割
# data_group = {}
# groups = df1.groupby(by=0)  # 定义分组依据第几列特征进行分割，by=0为第一列
# for key in groups.groups.keys():
#     data_group[key] = df1.loc[groups.groups[key], :]
# # 对所选择特征中存在的所有可能子特征做遍历，每次取对应的切片矩阵
# # 4.查看分组结果：
# print(data_group[0])


# --------------------------------------------------------------------------------#
# 之后的所有思考设计到种子的都要这个读取文件
file_name = 'F:/大下二/机器学习实验/data.xls'
df_seed = pd.read_excel(file_name)

# 思考1
# x = [i / 100 for i in range(1, 101)]
# z1 = [-i * log(i, 2) for i in x]
# z = []
# for i in range(1, 101):
#     z.append(sum(z1[:i]))
# plt.xlabel("p(x)")
# plt.ylabel("l(x)")
# plt.plot(x, z, color="blue")  # 这改线条颜色
# plt.show()


# 思考2
# data_1 = [
#     [1, 0, 0, '打球'],
#     [1, 0, 1, '打球'],
#     [0, 1, 1, '打球'],
#     [0, 0, 1, '不打球'],
#     [1, 1, 1, '打球']
# ]
#
#
# def ent(data):
#     """
#
#     :param data:list
#     :return: entropy
#     """
#     df = pd.DataFrame(data)
#     entropy = 0
#     target = df.iloc[:, -1]
#     label_counts = target.value_counts()
#     label_dict = label_counts.to_dict()
#     data_length = len(data)
#     entropy = 0
#     for key in label_dict:
#         prob = float(label_dict[key]) / data_length
#         entropy -= prob * np.log2(prob)
#     return entropy


# print(ent(data_1))

# 对种子的信息熵求解
# def ent_seed(df):
#     """
#     :作用：对种子的信息熵求解
#     :param data:pandas数据库
#     :return: entropy
#     """
#     entropy = 0
#     target = df.iloc[:, -1]
#     label_counts = target.value_counts()
#     label_dict = label_counts.to_dict()
#     data_length = len(df.iloc[:, 0])
#     entropy = 0
#     for key in label_dict:
#         prob = float(label_dict[key]) / data_length
#         entropy -= prob * np.log2(prob)
#     return entropy
#
#
# print(ent_seed(df_seed))


# 思考3
# 按照data_1进行筛选
# data_1 = [
#     [1, 0, 0, '打球'],
#     [1, 0, 1, '打球'],
#     [0, 1, 1, '打球'],
#     [0, 0, 1, '不打球'],
#     [1, 1, 1, '打球']
# ]
#
#
# def split(data, i):
#     df = pd.DataFrame(data)
#     data_group = {}
#     groups = df.groupby(by=i-1)
#     for key in groups.groups.keys():
#         data_group[key] = df.loc[groups.groups[key], :]
#     return data_group
#
#
# print(split(data_1, 1))

# 思考4
# data = [
#     [1, 0, 0, '打球'],
#     [1, 0, 1, '打球'],
#     [0, 1, 1, '打球'],
#     [0, 0, 1, '不打球'],
#     [1, 1, 1, '打球']
# ]
#
#
# def ent(data):
#     """
#     :param data:list
#     :return: entropy
#     """
#     df = pd.DataFrame(data)
#     entropy = 0
#     target = df.iloc[:, -1]
#     label_counts = target.value_counts()
#     label_dict = label_counts.to_dict()
#     data_length = len(data)
#     entropy = 0
#     for key in label_dict:
#         prob = float(label_dict[key]) / data_length
#         entropy -= prob * np.log2(prob)
#     return entropy
#
#
# def split(data, i):
#     df = pd.DataFrame(data)
#     data_group = {}
#     groups = df.groupby(by=i - 1)
#     for key in groups.groups.keys():
#         data_group[key] = df.loc[groups.groups[key], :]
#     return data_group
#
#
# def ent_gain(data, feature_rank):
#     """
#        :param data:list, str or int
#        :return: info_gain
#     """
#
#     init_ent = ent(data)
#     data_group = split(data, feature_rank)
#     new_ent = 0
#     for key in data_group:
#         prob = len(data_group[key]) / len(data)
#         new_ent += prob * ent(data_group[key])
#     info_gain = init_ent - new_ent
#
#     return info_gain
#
#
# feature_rank = 1
# print(ent_gain(data, feature_rank))


# 思考5
# data = [
#     [1, 0, 0, '打球'],
#     [1, 0, 1, '打球'],
#     [0, 1, 1, '打球'],
#     [0, 0, 1, '不打球'],
#     [1, 1, 1, '打球']
# ]
#
#
# def ent(data):
#     """
#     :param data:list
#     :return: entropy
#     """
#     df = pd.DataFrame(data)
#     entropy = 0
#     target = df.iloc[:, -1]
#     label_counts = target.value_counts()
#     label_dict = label_counts.to_dict()
#     data_length = len(data)
#     entropy = 0
#     for key in label_dict:
#         prob = float(label_dict[key]) / data_length
#         entropy -= prob * np.log2(prob)
#     return entropy
#
#
# def split(data, i):
#     df = pd.DataFrame(data)
#     data_group = {}
#     groups = df.groupby(by=i - 1)
#     for key in groups.groups.keys():
#         data_group[key] = df.loc[groups.groups[key], :]
#     return data_group
#
#
# def ent_gain(data, feature_rank):
#     """
#        :param data:list, str or int
#        :return: info_gain
#     """
#
#     init_ent = ent(data)
#     data_group = split(data, feature_rank)
#     new_ent = 0
#     for key in data_group:
#         prob = len(data_group[key]) / len(data)
#         new_ent += prob * ent(data_group[key])
#     info_gain = init_ent - new_ent
#
#     return info_gain
#
#
# def split_index(data):
#     [data_len, data_wit] = np.array(data).shape
#     gain_list= []
#     for i in range(1, data_wit):
#         gain_list.append(ent_gain(data, i))
#     max_index = gain_list.index(max(gain_list))
#     return max_index
#
# print(split_index(data))

# 选做题
data = [
    [1, 0, 0, '打球'],
    [1, 0, 1, '打球'],
    [0, 1, 1, '打球'],
    [0, 0, 1, '不打球'],
    [1, 1, 1, '打球']
]


def ent(data):
    """
    :param data:list
    :return: entropy
    """
    df = pd.DataFrame(data)
    entropy = 0
    target = df.iloc[:, -1]
    label_counts = target.value_counts()
    label_dict = label_counts.to_dict()
    data_length = len(data)
    entropy = 0
    for key in label_dict:
        prob = float(label_dict[key]) / data_length
        entropy -= prob * np.log2(prob)
    return entropy


def split(data, i):
    data_group = {}
    a = data
    groups = data.groupby(by=i)
    for key in groups.groups.keys():
        data_group[key] = data.loc[groups.groups[key], :]
    return data_group


def ent_gain(data, feature_rank):
    """
       :param data:list, str or int
       :return: info_gain
    """

    init_ent = ent(data)
    data_group = split(data, feature_rank)
    new_ent = 0
    for key in data_group:
        prob = len(data_group[key]) / len(data)
        new_ent += prob * ent(data_group[key])
    info_gain = init_ent - new_ent

    return info_gain


def split_index(data):
    [data_len, data_wit] = np.array(data).shape
    gain_list = []
    for i in range(0, data_wit - 1):
        if i in data.columns:
            gain_list.append(ent_gain(data, i))
        else:
            gain_list.append(0)
    max_index = gain_list.index(max(gain_list))
    return max_index

# def PostPruning(tree):
#     """
#     后剪枝：先构造一颗完整的决策树，然后自底向上对非叶结点进行考察，若该结点对应的子树换为叶结点能够带来泛化性能的提升则把子树替换为叶结点
#     """

# ID3决策树的具体循环代码
def ID3(data, jianzhi = False):
    """
       实现ID3决策树
       :param data:list
       :return: 决策树
    """
    data = pd.DataFrame(data)
    Data = copy.copy(data)
    ID3_Tree = {}  # 使用字典存储决策树的信息
    max_index_list = []  # 按顺序存储树的分支特征
    flag = 0  # 默认表示进行了划分
    father_tree = ""  #表示子树的父树是什么
    father_tree_list = []
    while True:
        if len(Data.iloc[0, :]) == 1 or flag == 1:  # 结束条件如果只存在一个或者不需要进行划分
            return ID3_Tree  # 返回ID3决策数
        max_index = split_index(Data)
        max_index_list.append(max_index)
        feature = split(Data, max_index)  # 完整数据
        Tree_key_list = []
        for key in feature:
            df = pd.DataFrame(feature[key])
            if father_tree == "":   #表示不是树墩
                Tree_key = str(key) + "_" + str(max_index)  # 用于作为ID3存储的key
            else:
                Tree_key = father_tree + str(key)+ "_" + str(max_index)
            Tree_key_list.append(Tree_key)     # 当前子树的分叉项
            ID3_Tree[Tree_key] = df
        num, Num = 0, 0  # 表示进行了几次划分
        for i, v in ID3_Tree.items():  # 选择子树
            if i in Tree_key_list:
                Num += 1
                if v.iloc[:, -1].nunique() == 1:  # 无需分类，则跳转下一个子树
                    num += 1
                    continue
                else:
                    father_tree = i + "_"
                    father_tree_list.append(father_tree)
                    Data = v.drop(max_index, axis=1)  # 删除了最大指针的数据

        if num == Num :
            # if jianzhi:
            #     PostPruning(ID3_Tree)
            flag = 1


TREE = ID3(data)
print(TREE)
# createPlot(TREE)