import numpy as np
import pickle
# path='D:\\video\\mv\\mv5'
#flag=False


# I_frame=[1]
# refer_list0=[]
# refer_list1=[]
# for i in range(1,2998):
#     path = 'D:\\video\\v_t\\mv\\mv'+str(i)
#     print(i)
#     mv_list = np.zeros((80, 45, 2, 3))
#     with open(path) as f:
#         is_I=True
#         refer_list0.append([])
#         refer_list1.append([])
#         x,y=0,0
#         mv0x,mv0y=[],[]
#         mv1x, mv1y =[],[]
#         POC0,POC1=0,0
#         for idx, line in enumerate(f.readlines()):
#             #将上一个宏块mv数据保存，然后要处理一个新的宏块，初始化
#             if '---------'in line and idx!=0:
#                 flag=False
#                 if len(mv0x)!=0:
#                     mvx=sum(mv0x)/len(mv0x)
#                     mvy = sum(mv0y) / len(mv0y)
#                     #print(mvx, mvy, I_frame[-1] + POC1 / 2)
#                     mv_list[x][y][0][0],mv_list[x][y][0][1],mv_list[x][y][0][2]=mvx,mvy,I_frame[-1]+POC0/2
#                 if len(mv1x)!=0:
#                     mvx=sum(mv1x)/len(mv1x)
#                     mvy = sum(mv1y) / len(mv1y)
#
#                     mv_list[x][y][0][0],mv_list[x][y][0][1],mv_list[x][y][0][2]=mvx,mvy,I_frame[-1]+POC1/2
#                 mv0x, mv0y = [], []
#                 mv1x, mv1y = [], []
#                 POC0, POC1 = 0, 0
#             #保存MB位置信息
#             elif 'MB:'in line :
#                 MB=line.replace('MB: ','').split(', ')
#                 x=int(MB[0])
#                 y=int(MB[1])
#             #前向参考帧进行处理
#             elif 'MV0'in line :
#                 is_I=False
#                 MV=line.replace('\n','').replace(',',' ').replace('        MV','').split()
#                 #print(MV)
#                 mv0x.append(int(MV[1]))
#                 mv0y.append(int(MV[2]))
#                 POC0=int(MV[3].replace('poc:',''))
#                 #print(POC0)
#                 r_l=I_frame[-1]+POC0/2
#                 if r_l not in refer_list0[-1]:
#                     refer_list0[-1].append(int(r_l))
#                     #print(i,'ckz0',r_l)
#             #后向参考帧进行处理
#             elif 'MV1' in line:
#                 is_I = False
#                 MV = line.replace('\n', '').replace(',',' ').replace('        MV', '').split()
#                 # print(MV)
#                 mv1x.append(int(MV[1]))
#                 mv1y.append(int(MV[2]))
#                 POC1 = int(MV[3].replace('poc:', ''))
#                 r_l = I_frame[-1] + POC1 / 2
#                 if r_l not in refer_list1[-1]:
#                     refer_list1[-1].append(int(r_l))
#                     #print(i,'ckz1', r_l)
#         if is_I:
#             I_frame.append(i)
#         if len(mv0x) != 0:
#             mvx = sum(mv0x) / len(mv0x)
#             mvy = sum(mv0y) / len(mv0y)
#             mv_list[x][y][0] = [mvx, mvy, I_frame[-1] + POC0 / 2]
#         if len(mv1x) != 0:
#             mvx = sum(mv1x) / len(mv1x)
#             mvy = sum(mv1y) / len(mv1y)
#             mv_list[x][y][1] = [mvx, mvy, I_frame[-1] + POC1 / 2]
#
#         with open("D:\\video\\v_t\\pro_mv\\mv"+str(i)+".txt", "wb") as myprofile:
#             pickle.dump(mv_list, myprofile)
# with open("D:\\video\\v_t\\pro_mv\\I_frame.txt", "wb") as myprofile:
#     pickle.dump(I_frame, myprofile)
# with open("D:\\video\\v_t\\pro_mv\\refer_list0.txt", "wb") as myprofile:
#     pickle.dump(refer_list0, myprofile)
# with open("D:\\video\\v_t\\pro_mv\\refer_list1.txt", "wb") as myprofile:
#     pickle.dump(refer_list1, myprofile)

# with open("D:\\video\\pro_mv\\mv"+str(2)+".txt", "rb") as get_myprofile:
#     times=pickle.load(get_myprofile)
# print(times)
width=1280
height=720

with open("D:\\VASRL\\mv\\refer_list0.txt", "rb") as get_myprofile:
    r0l = pickle.load(get_myprofile)
    print(r0l)
with open("D:\\VASRL\\mv\\refer_list1.txt", "rb") as get_myprofile:
    r1l=pickle.load(get_myprofile)
    print(r1l)
P=[]
for idx,i in enumerate(r0l):
    if len(i)==0 and len(r1l[idx])==0:
        P.append(idx)
print(P)
with open("D:\\VASRL\\mv\\all_mv.txt", "rb") as get_myprofile:
    all_mv=pickle.load(get_myprofile)
def getmblist(x,y,w,h):
    x0 = int(x * width)
    y0 = int(y * height)
    x1 = int(w * width+x0)
    y1 = int((h * height) + y0)
    return min(79,int(x0/16)),min(44,int(y0/16)),min(79,int(x1/16)+1),min(44,int(y1/16)+1)

def find_refer(frame_id):
    # with open("D:\\video\\v_t\\pro_mv\\refer_list0.txt", "rb") as get_myprofile:
    #     r0l=pickle.load(get_myprofile)
    # with open("D:\\video\\v_t\\pro_mv\\mv"+str(frame_id)+".txt", "rb") as get_myprofile:
    #     MV=pickle.load(get_myprofile)
    MV=all_mv[frame_id-1]
    x0,y0,x1,y1=getmblist(0,0,1,1)
    if x0==x1:
        x1+=1
    if y0==y1:
        y1+=1
    rl=np.zeros(len(r0l[frame_id-1]))
    rl=rl.tolist()
    for i in range(x0,x1,1):
        for j in range(y0,y1,1):
            #print(MV[i][j])
            if MV[i][j][0][2] in r0l[frame_id-1]:
                rl[r0l[frame_id-1].index(MV[i][j][0][2])]+=1
    #print(frame_id,rl)
    if len(rl)==0:
        return -1
    return r0l[frame_id-1][rl.index(max(rl))]

def find_re_refer(frame_id):

    # with open("D:\\video\\v_t\\pro_mv\\mv"+str(frame_id)+".txt", "rb") as get_myprofile:
    #     MV=pickle.load(get_myprofile)
    MV = all_mv[frame_id - 1]
    x0,y0,x1,y1=getmblist(0,0,1,1)
    if x0==x1:
        x1+=1
    if y0==y1:
        y1+=1
    rl=np.zeros(len(r1l[frame_id-1]))
    rl=rl.tolist()
    for i in range(x0,x1,1):
        for j in range(y0,y1,1):
            #print(MV[i][j])
            if MV[i][j][1][2] in r1l[frame_id-1]:
                rl[r1l[frame_id-1].index(MV[i][j][1][2])]+=1
    return r1l[frame_id-1][rl.index(max(rl))]

def clean(data_list):
    data_array = np.asarray(data_list)

    if len(data_list)>5:
        data_array=data_array[data_array != 0]
        if data_array.shape[0]==0:
            return data_array
        mean = np.mean(data_array, axis=0)
        std = np.std(data_array, axis=0)

        preprocessed_data_array = [x for x in data_array if (x > mean - 0.8 * std)]
        preprocessed_data_array = [x for x in preprocessed_data_array if (x < mean + 0.8 * std)]
        return preprocessed_data_array
    return data_array

def move_bbox(frame_id,x,y,w,h,refer):
    MV = all_mv[frame_id - 1]
    x0, y0, x1, y1 = getmblist(x, y, w, h)
    x=[]
    y=[]
    if x0==x1:
        x1+=1
    if y0==y1:
        y1+=1
    for i in range(x0, x1, 1):
        for j in range(y0, y1, 1):
            if MV[i][j][0][2]==refer:
                x.append(MV[i][j][0][0])
                y.append(MV[i][j][0][1])
    x=clean(x)
    y=clean(y)
    if len(x)==0 or len(y)==0:
        return 0,0
    # elif len(x)!=0:
    #     return sum(x)/len(x),0
    # elif len(y)!=0:
    #     return 0,sum(y)/len(y)
    return sum(x)/len(x),sum(y)/len(y)

# print(move_bbox(1037,0.04632002760450812,0.6901557743949392,0.12054962582058376,0.12123177846272781,1034))
def move_re_bbox(frame_id,x,y,w,h,refer):
    MV = all_mv[frame_id - 1]
    x0, y0, x1, y1 = getmblist(x, y, w, h)
    x=[]
    y=[]
    if x0==x1:
        x1+=1
    if y0==y1:
        y1+=1
    for i in range(x0, x1, 1):
        for j in range(y0, y1, 1):
            if MV[i][j][1][2]==frame_id:
                x.append(MV[i][j][1][0])
                y.append(MV[i][j][1][1])

    x=clean(x)
    y=clean(y)
    if len(x)==0 or len(y)==0:
        return 0,0

    return sum(x)/len(x),sum(y)/len(y)
# ob=[-61.0, -28.5, -38.142857142857146, -65.0, -52.125, -60.625, -51.0, -50.75, -39.8, -64.0, -54.57142857142857, -56.0, -60.0, -44.0, -52.0, -57.285714285714285, -58.857142857142854, -53.75, -24.5]
# from pykalman import KalmanFilter
# ob=[0.0, 0.0, -61.0, -80.0, -78.66666666666667, -81.0, 0.0, 0.0, -77.0, -80.0, 0.0, -70.5, -73.66666666666667, -28.5, -66.0, -73.25, -77.0, 0.0, -38.142857142857146, -65.0, -71.4, -70.0, -72.0, -74.5, -52.125, -60.625, -67.0, -67.25, 0.0, -51.0, -50.75, -39.8, -64.0, -68.0, 0.0, -54.57142857142857, -56.0, -60.0, -44.0, 0.0, -52.0, -57.285714285714285, -58.857142857142854, -53.75, 0.0, -4.0, -24.5, -14.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# ob=clean(ob)
# print(ob)
# import networkx as nx
#
# import matplotlib.pyplot as plt
#
#
# with open("D:\\video\\pro_mv\\refer_list0.txt", "rb") as get_myprofile:
#     r0l= pickle.load(get_myprofile)
# with open("D:\\video\\pro_mv\\refer_list1.txt", "rb") as get_myprofile:
#     r1l= pickle.load(get_myprofile)
# list=[]
# e_list=[]
# print(r1l[4])
# # for idx,r_f in enumerate(r0l):
# #     if idx==0:
# #         list.append(idx)
# #     elif idx<40:
# #         list.append(idx)
# #         #for i in r_f:
# #         i=find_refer(idx+1)-1
# #         e_list.append((idx,i))
# for idx,r_f in enumerate(r1l):
#     if idx==0:
#         list.append(idx)
#         continue
#     elif idx<30:
#         #list.append(idx)
#         #for i in r_f:
#         i=find_refer(idx+1)-1
#         e_list.append((idx,i))
# G = nx.DiGraph()
#
# #G.add_node('z')     # 添加节点z
#
# G.add_nodes_from(list)   # 添加节点 1 2 3
#
# #G.add_edge('x', 'y')          # 添加边  起点为x  终点为y
#
# G.add_edges_from(e_list)   # 添加多条边
#
# x=1.1
# #x=float(x)
# np.clip(x,0,1)
# print(x)

# 网络图绘制与显示
# pos = nx.circular_layout(G)
# nx.draw(G,pos=pos, with_labels=True,font_size=10,node_size =100)
#
# plt.show()


# xb=[-120.42857142857143, -113.66666666666667, -123.75, -120.0, -127.0, -124.0, -122.875, -125.83333333333333, -124.0, -124.0, -131.4, -121.0, -132.875, -128.0, -128.0, -128.0, -131.2, -127.57142857142857, -119.0, -126.0, -131.5, -127.5, -129.8, -131.42857142857142, -132.42857142857142, -110.8, -136.0, -64.0, -100.0, -40.0, -84.0, -129.33333333333334, 49.625, -144.0, -132.0, -109.0, -108.0, -108.0, -96.0, -99.0, -97.0, -136.0, -127.0, -136.5, -126.0, -95.71428571428571]
# xq=[-28.714285714285715, -27.166666666666668, -31.75, -24.0, -27.0, -37.5, -41.5, -44.833333333333336, -26.0, -28.0, -32.6, -36.0, -47.125, -28.0, -29.0, -32.0, -33.6, -35.142857142857146, -25.625, -14.0, -30.5, -30.5, -33.8, -31.0, -32.0, -23.8, -28.0, -19.0, -20.0, -28.0, -24.0, -37.833333333333336, 14.5, -30.0, -28.0, -24.0, -24.0, -24.0, -28.0, -28.0, -28.0, -27.875, -26.875, -34.0, -31.5, -27.142857142857142]
# xb=[-144.0, 0.0, -4.0, -7.8, -86.5, -243.0, -179.375, -192.0, -168.0, -16.0, -44.2, -152.0, -158.71428571428572, -168.0, -159.0, -158.0, -157.42857142857142, -153.42857142857142, -159.0, -160.66666666666666, -151.5, -160.0, -161.0, -158.28571428571428, -153.0, -150.0, -164.25]
# xq=[-44.0, 0.0, -1.0, -1.6, -24.0, -64.0, -49.375, -48.0, -32.0, 0.0, -10.2, -40.0, -42.142857142857146, -47.0, -40.0, -42.5, -39.142857142857146, -38.285714285714285, -45.0, -41.333333333333336, -45.0, -49.0, -48.5, -47.0, -43.0, -48.5, -48.0]
# print(sum(clean(xb))/len(clean(xb)))
# print(sum(xb)/len(xb))
# xb=[-1.8, 0.0, -27.625, -7.4, -20.2, -26.125, -25.0, -25.5, -25.0, -21.25, -27.0, -27.0]
# import numpy as np
# data_list =xb
# data_array = np.asarray(data_list)
# # print(sum(clean(xb))/len(clean(xb)))
# print(sum(xb)/len(xb))
# mean = np.mean(data_array , axis=0)
# std = np.std(data_array , axis=0)
#
# preprocessed_data_array = [x for x in data_array if (x > mean - 1*std)]
# preprocessed_data_array = [x for x in preprocessed_data_array if (x < mean + 1*std)]
# print(sum(preprocessed_data_array)/len(preprocessed_data_array) ,preprocessed_data_array,len(xb))
