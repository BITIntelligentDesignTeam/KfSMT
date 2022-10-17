import csv
import numpy as np

def csv_read(path):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        result = list(reader)
    title_put = result[0]
    num_input = title_put.count("输入")
    title_par = result[2]
    dic = {'input':title_par[0:num_input],'output':title_par[num_input:]}
    data = result[3:]
    for i in range(len(data)):
        for j in range(len(data[0])):
            data[i][j] = float(data[i][j])
    data = np.array(data)
    inputData = data[:, 0:num_input]
    outputData = data[:, num_input:]
    return inputData, outputData, dic

if __name__ == "__main__":
    a,b,c=csv_read('c:\data\学科w.csv')
    print(a)
    print(b)
    print(c)



