# This is a sample Python script.

# Token:
# cd5c0f2b8bb97f0cd4e46dd6f0e5647922a163d3

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import sys
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# numm=-math.inf
# if numm<10000000:print (math.atan(numm))
# exit()
import xlsxwriter

DEBUG1 = True
DEBUG2 = False
DEBUG3 = False
DEBUG4 = False
DEBUG5 = False

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# ['Num','Date','MinTemp','RayFrom','RayTo','dx','dy','K','B','FLiine']
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')


input_file_name='daily-min-temperatures-03.csv'
df = pd.read_csv(input_file_name,
                 names=['Date', 'MinTemp'])

#TODO сделать одномерный массив data_list заполнить для замены дат
data_list = df.to_numpy()
max_dim=len(data_list)
new_ind=np.arange(0,max_dim)
data_list[:,0]=new_ind # Замена даты на порядковый номер (index)
#TODO построить
# tips_df = sns.load_dataset('tips')
# tips_df.head()
# plt.show()


#
# TODO замена дат порядковыми номероми
# for ind in range(0,len(df)):
#     new_ind.append(ind)
#     print("new_ind:", ind, new_ind[ind])
#data_list[:,0]=new_ind
# if DEBUG:
#     for ind in range(0,len(df)): print("DEBUG_3:", ind, df)

def line(x0, x1, x):
    y0 = data_list[x0][1]
    y1 = data_list[x1][1]

    k = (y1 - y0) / (x1 - x0)
    B = (x1 * y0 - x0 * y1) / (x1 - x0)
    # eps = 1.e-3
    #return [k, B, k * x + B, (k * x + B - data_list[x]) <= 0]
    return k

#def cluster_fill(x0):


def is_visible(x0, x1, x):
    y0 = data_list[x0]
    y1 = data_list[x1]

    k = (y1 - y0) / (x1 - x0)
    B = (x1 * y0 - x0 * y1) / (x1 - x0)
    # eps = 1.e-3
    return (k * x + B - data_list[x] <= 0)


# max_dim = sum(1 for my_line in open(input_file_name,'r'))

def print_graph_array(graph_array):
    for ind in range(0,len(graph_array)):print(ind,":",graph_array[ind])

def build_graph_with_clusters(start_point, max_dim, DEBUG = False):
    x0 = start_point    # начало кластера
    cluster_start = start_point    # начало кластера
    x1 = x0 + 1         # рабочая вершина
    x2 = x1+1           # контрольная вершина
    cluster_size = 1
    graph_array = np.eye(max_dim)
    graph_array.fill(1)
    if max_dim == 2: # если всего 2 вершины, матрица 2x2
        vis_k = line(x0, x1, x1)
        try:
            vis_k_next = line(x0, x2, x1)
        except IndexError as e:
            # print(e)
            # print(sys.exc_info())
            vis_k_next = vis_k
        ind = np.argmax([vis_k, vis_k_next])
        if ind==0:
            graph_array[x0][x0] = 2
        else:
            graph_array[x1][x1] = 2
        # graph_array[x0][x0] = 1
        # graph_array[x1][x0] = 1
        return graph_array
    if max_dim == 3: # если матрица 3x3
        vis_k = line(x0, x1, x1)
        try:
            vis_k_next = line(x0, x2, x1)
        except IndexError as e:
            # print(e)
            # print(sys.exc_info())
            vis_k_next = vis_k
        ind=np.argmax([vis_k,vis_k_next])

        if ind==0:
            graph_array[x0][x0]=2


        #graph_array.fill(1)
        #graph_array[x1][x1]=2
        return graph_array
    array_k = []
    array_cluster_start=[]
    array_cluster_sizes=[]
# -----------------------------------------------
# Start main algorhithm
# -----------------------------------------------

    while True:
        x1 = x0+1
        x2 = x1+1
        vis_k = line(x0, x1, x1)
        #vis_k_next = line(x0, x2, x1)
        try:
            vis_k_next = line(x0, x2, x1)
        except IndexError as e:
            # print(e)
            # print(sys.exc_info())
            vis_k_next = vis_k
        ind=np.argmax([vis_k,vis_k_next])

        #ind=np.argmax([vis_k,vis_k_next])

        # if max_dim == 2:
        #     graph_array = [[2, 1], [1, 1]]
        #     graph_array[x0][x1] = 1
        #     graph_array[x1][x0] = 1
        # if x2 >= max_dim:
        #     break
        vis_k = line(x0, x1, x1)
        #x2 = x1+1
        # if x2 < max_dim:
        #     vis_k_next = line(x0, x2, x2)

        cluster_len=1
        while x1 <= max_dim - 1:
            vis_k = line(x0, x1, x1)
            x2 = x1+1
            try:
                vis_k_next = line(x0, x2, x1)
            except IndexError as e:
                # print(e)
                # print(sys.exc_info())
                vis_k_next = vis_k

            ind = np.argmax([vis_k, vis_k_next])
            if ind==0: # cluster end x0
                cluster_len=1
                array_cluster_start.append(cluster_start)
                x0=x1 # новое начало кластера x0=x1
                x1=x0+1
                x2=x1+1
                print ("**0****",x0,x1)
                if DEBUG1:print("DEBUG_1:From= ", x0, "To= ", x1, "k= ", vis_k, "Next= ", x2, "next_k=", vis_k_next,"cluster_size=", cluster_size,"ind=",ind,"cluster_len=",cluster_len)
            if ind==1: # continue cluster x0
                cluster_len+=1
                if len(array_cluster_sizes)==0:
                    array_cluster_sizes.append(cluster_len)
                array_cluster_sizes[cluster_start]=cluster_len+1
                print ("**1****",x0,x1)
                if DEBUG1:print("DEBUG_1:From= ", x0, "To= ", x1, "k= ", vis_k, "Next= ", x2, "next_k=", vis_k_next,"cluster_size=", cluster_size,"ind=",ind,"cluster_len=",cluster_len)
            array_k.append([x0, x1, vis_k, vis_k_next, cluster_size])
            x1+=1
            x2 = x1+1
        if x0 > (max_dim-3):
            break

            is_growing = vis_k <= vis_k_next
            is_decrease = not is_growing
            if is_growing:
                # Вершина x2 вхоит в кластер? -да
                graph_array[x0][x1] = 1
                #graph_array[x0][x2] = 1
                graph_array[x1][x0] = 1
                #graph_array[x2][x0] = 1
                if DEBUG4: print("DEBUG_4:From= ", x0, "To= ", x1, "k= ", vis_k, "Next= ", x2, "next_k=",
                                vis_k_next,"cluster_size=",
                                cluster_size, is_growing)

                #x0 = x1
                x1 = x1 + 1
                x2 = x1+1
                # TODO  graph_array[x0][x0] += 1
                graph_array[x0][x0] += 1
                # val=graph_array[x0][x0]
                # cluster_range=range(x0,val)
                # for i in range(x0,x0+cluster_range-1):
                #     for j in range(x0, x0+cluster_range-1):
                #             graph_array[i][j]=1
                # cluster_size = cluster_size + 1
                # graph_array[x1][x0] = 1
            else:
                # Вершина x1 вхоит в x0 кластер? -нет
                graph_array[x0][x0] = cluster_size
                cluster_size = 1
                # начало нового кластера в x1
                x0 = x1
                x1 = x0 + 1
                x2 = x1+1
                #TODO Проверить логику
                #graph_array[x0][x1] = 1
                #graph_array[x1][x0] = 1
                x1 = x0 + 1
                x2 = x1+1
                if DEBUG2:print("DEBUG_2:",x0, x1, x2, vis_k, vis_k_next, cluster_size, is_decrease)
        # if x0 == max_dim - 3:
        #     #graph_array[x0][x0] = cluster_size+1
        #     #graph_array[x0][x1] = 1
        #     #graph_array[x0][x2] = 1
        #     #graph_array[x1][x0] = 1
        #     #graph_array[x2][x0] = 1
        #     #graph_array[x2][x0] = 1
        #     break
        # if x1 == max_dim - 1:
        #     break
    if DEBUG:
        lenk=len(array_k)
        print("****  Array_k  ****")
        for ind in range(0, lenk):
            print("DEBUG_5: ind=", ind, array_k[ind])
        print("******************")

    return graph_array

graph_array = build_graph_with_clusters(0,max_dim, True)
# print(graph_array)
# fig = plt.figure()
# ax = fig.add_axes([0,0,1,1])
# x = ['C', 'C++', 'Java', 'Python', 'PHP']
# y = [4,2,3,7,3]
# ax.bar(x,y)
plt.matshow(graph_array)

plt.show()


# eOutput = pd.DataFrame(graph_array)
# writer = pd.ExcelWriter('ArrayFromPycharm.xlsx', engine='xlsxwriter')
#
# # Convert the dataframe to an XlsxWriter Excel object.
# eOutput.to_excel(writer, sheet_name='Sheet1', index=False)
#
# # Close the Pandas Excel writer and output the Excel file.
# writer.save()

#---------------------------------  Version 1.0 ------------------------------
