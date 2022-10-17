# coding:utf-8
import xlrd
import numpy as np


# 用于交换列表中的顺序
def swapPositions(list_i, pos1, pos2):
    list_i[pos1], list_i[pos2] = list_i[pos2], list_i[pos1]
    return list_i


def data_process(excel_path):
    # excel路径
    # excle_path = r'C:\Users\DELL\Desktop\pythonProject\全弹法向力系数1_3d.xls'
    # 打开excel读取文件
    data = xlrd.open_workbook(excel_path)
    # 根据sheet下标选择读取内容
    sheet = data.sheet_by_index(0)
    # 根据第一行读取滚转角PHI值
    label1 = sheet.row_values(0)

    # 获取PHI标签，获取D_MACH、D_ALPHA标签
    for i in range(len(label1)):

        if label1[i] == 'PHI':
            label_phi = i

        elif label1[i] == 'D_MACH':
            label_mach = i

        elif label1[i] == 'D_ALPHA':
            label_alpha = i

    # 去掉第一行，获取列值
    col_phi = sheet.col_values(colx=label_phi, start_rowx=1)

    # 根据滚转角0，45，90分别拆分数据
    data_0 = []
    data_45 = []
    data_90 = []

    for i in range(len(col_phi)):
        if float(col_phi[i]) == 0:
            list0 = sheet.row_values(rowx=i + 1)
            a0 = [float(j) for j in list0]

            # 交换攻角在前，马赫数在后
            a0 = swapPositions(a0, label_mach, label_alpha)
            data_0.append(a0)

        elif float(col_phi[i]) == 45:
            list45 = sheet.row_values(rowx=i + 1)
            a45 = [float(j) for j in list45]

            # 交换攻角在前，马赫数在后
            a45 = swapPositions(a45, label_mach, label_alpha)
            data_45.append(a45)

        elif float(col_phi[i]) == 90:
            list90 = sheet.row_values(rowx=i + 1)
            a90 = [float(j) for j in list90]

            # 交换攻角在前，马赫数在后
            a90 = swapPositions(a90, label_mach, label_alpha)
            data_90.append(a90)

    # 转换成np数组
    data_0 = np.array(data_0)
    data_45 = np.array(data_45)
    data_90 = np.array(data_90)

    # 去除离散值
    for k in range(data_0.shape[1]):
        row_0 = data_0[:, k]

        if (np.squeeze(row_0) == float(0.0)).all():
            data_0 = np.delete(data_0, k, axis=1)
            break

    for q in range(data_45.shape[1]):
        row_45 = data_45[:, q]

        if (np.squeeze(row_45) == float(45.0)).all():
            data_45 = np.delete(data_45, q, axis=1)
            break

    for w in range(data_90.shape[1]):
        row_90 = data_90[:, w]

        if (np.squeeze(row_90) == float(90.0)).all():
            data_90 = np.delete(data_90, w, axis=1)
            break

    return data_0, data_45, data_90


def object_label(excel_path):
    # 打开excel读取文件
    data = xlrd.open_workbook(excel_path)
    # 根据sheet下标选择读取内容
    sheet = data.sheet_by_index(0)
    # 根据第一行值
    label1 = sheet.row_values(0)
    object = label1[-1]
    return object


if __name__ == '__main__':
    d = data_process(r'C:\Users\石磊\Desktop\馥琳姐代码\Source\全弹法向力系数1_3d.xls')
    print(d)
