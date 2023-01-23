import os
#import psutil
import numpy as np
import pandas as pd
import cv2
blank=[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
blank1=np.zeros((8,8))
blank1=blank1.tolist()
from numpy import *
# def yuv2rgb422(y, u, v):
#     """
#     :param y: y分量
#     :param u: u分量
#     :param v: v分量
#     :return: rgb格式数据以及r,g,b分量
#     """
#
#     rows, cols = y.shape[:2]
#     #print(y.shape[:2])
#     # 创建r,g,b分量
#     r = np.zeros((rows, cols), np.uint8)
#     g = np.zeros((rows, cols), np.uint8)
#     b = np.zeros((rows, cols), np.uint8)
#
#     for i in range(rows):
#         for j in range(int(cols / 2)):
#             #print(j)
#             if j < cols/2:
#                 r[i, 2 * j] = max(0, min(255, y[i, 2 * j] + 1.402 * (v[i, j] - 128)))
#                 g[i, 2 * j] = max(0, min(255, y[i, 2 * j] - 0.34414 * (u[i, j] - 128) - 0.71414 * (v[i, j] - 128)))
#                 b[i, 2 * j] = max(0, min(255, y[i, 2 * j] + 1.772 * (u[i, j] - 128)))
#             if (2 * j + 1)<cols:
#                 r[i, 2 * j + 1] = max(0, min(255, y[i, 2 * j + 1] + 1.402 * (v[i, j] - 128)))
#                 g[i, 2 * j + 1] = max(0, min(255, y[i, 2 * j + 1] - 0.34414 * (u[i, j] - 128) - 0.71414 * (v[i, j] - 128)))
#                 b[i, 2 * j + 1] = max(0, min(255, y[i, 2 * j + 1] + 1.772 * (u[i, j] - 128)))
#
#     rgb = cv2.merge([b, g, r])
#
#     return rgb, r, g, b
def yuv2rgb(Y,U,V,width,height):
    U=repeat(U,2,0)
    U=repeat(U,2,1)
    V=repeat(V,2,0)
    V=repeat(V,2,1)
    rf=zeros((height,width),float,'C')
    gf=zeros((height,width),float,'C')
    bf=zeros((height,width),float,'C')

    rf=Y+1.14*(V-128.0)
    gf=Y-0.395*(U-128.0)-0.581*(V-128.0)
    bf=Y+2.032*(U-128.0)

    for m in range(height):
        for n in range(width):
            if(rf[m,n]>255):
                rf[m,n]=255
            if(gf[m,n]>255):
                gf[m,n]=255
            if(bf[m,n]>255):
                bf[m,n]=255

    r=rf.astype(uint8)
    g=gf.astype(uint8)
    b=bf.astype(uint8)
    rgb = cv2.merge([b, g, r])
    return rgb

def fin_res(res,_MB):
    #print(_MB)
    while _MB!=22:
        res=res+blank
        _MB+=1
        #print(len(res))
    return res
def fin_res1(res,_MB):
    #print(_MB)
    while _MB!=22:
        res=res+blank1
        _MB+=1
        #print(len(res))
    return res
from PIL import Image
def a2i(indata):
    mg = Image.new('L',indata.transpose().shape)
    mn = indata.min()
    a = indata-mn
    mx = a.max()
    a = a*256./mx
    mg.putdata(a.ravel())
    return mg
def get_res_pic_Y(path):
    all_residual = []
    residual = []
    index = 0
    _MB = -1
    with open(path) as f:
        for idx,line in enumerate(f.readlines()):
            #print(idx)
            #print(line)
            if idx%38==1:
                MB=line.split('MB: ')[1].replace("\n","").split(', ')
                MB=[int(i) for i in MB]
                if MB[0]-index>1:
                    if len(residual)==0:
                        residual = fin_res(residual, -1)
                        all_residual = residual
                        #print('l',len(all_residual))
                        residual = []
                    elif index==0:
                        residual = fin_res(residual, _MB)
                        all_residual = residual
                        residual = []
                    else:
                        residual = fin_res(residual, _MB)
                        for i in range(0, len(residual)):
                            # print(i)
                            # print(residual[i])
                            #print('0',len(all_residual[0]))
                            all_residual[i] = all_residual[i] + residual[i]
                        residual = []
                        _MB = -1
                    for i in range(MB[0]-index-1):
                        residual = fin_res(residual, -1)
                        #print(index)
                        for j in range(0, len(residual)):
                            # print(i)
                            # print(residual[i])
                            #print('1',len(all_residual[0]))
                            all_residual[j] = all_residual[j] + residual[j]

                        residual = []
                        _MB = -1

                elif MB[0]==1 and index==0:
                    #print(len(residual))
                    residual=fin_res(residual,_MB)
                    all_residual = residual
                    #print(len(residual), len(residual[0]))
                    residual = []

                    _MB = -1
                else:

                    if MB[0]!=index:
                        residual = fin_res(residual, _MB)
                        for i in range(0,len(residual)):
                            #print(i)
                            #print(residual[i])
                            #print('2',len(all_residual[0]))
                            all_residual[i]=all_residual[i]+residual[i]

                        residual=[]
                        _MB = -1



                while _MB!=MB[1]-1:
                    #print(0)
                    residual=residual+blank

                    _MB+=1

                _MB+=1
                index=MB[0]
            if idx%38 in range(3,19) :
                _line=line.split(',')[:16]
                _line=[int(i) for i in _line]
                residual.append(_line)
                # print(len(_line))
                #print(len(residual))
    residual = fin_res(residual, _MB)
    for i in range(len(residual)):
        all_residual[i] += residual[i]
    #print(len(all_residual), len(all_residual[0]))
    #print(index)
    if index!=39:
        residual=[]
        for i in range(39-index):
            residual = fin_res(residual, -1)
            for i in range(0, len(residual)):
                # print(i)
                # print(residual[i])
                all_residual[i] = all_residual[i] + residual[i]

            residual = []
            _MB = -1
            index+=1
    # for i in all_residual:
    #     print(i)
    #print(len(all_residual),len(all_residual[0]))
    res=np.array(all_residual)
    return res
    #res=np.matrix(res)
    #print(res.shape)
def get_res_pic_U(path):
    all_residual = []
    residual = []
    index = 0
    _MB = -1
    with open(path) as f:
        for idx,line in enumerate(f.readlines()):
            #print(idx)
            # print(line)
            if idx%38==1:
                MB=line.split('MB: ')[1].replace("\n","").split(', ')
                MB=[int(i) for i in MB]
                if MB[0]-index>1:
                    if len(residual)==0:
                        residual = fin_res1(residual, -1)
                        all_residual = residual
                        #print('l',len(all_residual))
                        residual = []
                    elif index==0:
                        residual = fin_res1(residual, _MB)
                        all_residual = residual
                        residual = []
                    else:
                        residual = fin_res1(residual, _MB)
                        for i in range(0, len(residual)):
                            # print(i)
                            # print(residual[i])
                            #print('0',len(all_residual[0]))
                            all_residual[i] = all_residual[i] + residual[i]
                        residual = []
                        _MB = -1
                    for i in range(MB[0]-index-1):
                        residual = fin_res1(residual, -1)
                        #print(index)
                        for j in range(0, len(residual)):
                            # print(i)
                            # print(residual[i])
                            #print('1',len(all_residual[0]))
                            all_residual[j] = all_residual[j] + residual[j]

                        residual = []
                        _MB = -1

                elif MB[0]==1 and index==0:
                    #print(len(residual))
                    residual=fin_res1(residual,_MB)
                    all_residual = residual
                    #print(len(residual), len(residual[0]))
                    residual = []

                    _MB = -1
                else:

                    if MB[0]!=index:
                        residual = fin_res1(residual, _MB)
                        for i in range(0,len(residual)):
                            #print(i)
                            #print(residual[i])
                            #print('2',len(all_residual[0]))
                            all_residual[i]=all_residual[i]+residual[i]

                        residual=[]
                        _MB = -1



                while _MB!=MB[1]-1:
                    #print(0)
                    residual=residual+blank1

                    _MB+=1

                _MB+=1
                index=MB[0]
            if idx%38 in range(20,28) :
                _line=line.split(',')[:8]
                _line=[int(i) for i in _line]
                residual.append(_line)
                # print(len(_line))
                #print(len(residual))
    residual = fin_res1(residual, _MB)
    for i in range(len(residual)):
        all_residual[i] += residual[i]
    #print(len(all_residual), len(all_residual[0]))
    #print(index)
    if index!=39:
        residual=[]
        for i in range(39-index):
            residual = fin_res1(residual, -1)
            for i in range(0, len(residual)):
                # print(i)
                # print(residual[i])
                all_residual[i] = all_residual[i] + residual[i]

            residual = []
            _MB = -1
            index+=1
    # for i in all_residual:
    #     print(i)
    #print(len(all_residual),len(all_residual[0]))
    res=np.array(all_residual)
    return res


def get_res_pic_V(path):
    all_residual = []
    residual = []
    index = 0
    _MB = -1
    with open(path) as f:
        for idx,line in enumerate(f.readlines()):
            #print(idx)
            # print(line)
            if idx%38==1:
                MB=line.split('MB: ')[1].replace("\n","").split(', ')
                MB=[int(i) for i in MB]
                if MB[0]-index>1:
                    if len(residual)==0:
                        residual = fin_res1(residual, -1)
                        all_residual = residual
                        #print('l',len(all_residual))
                        residual = []
                    elif index==0:
                        residual = fin_res1(residual, _MB)
                        all_residual = residual
                        residual = []
                    else:
                        residual = fin_res1(residual, _MB)
                        for i in range(0, len(residual)):
                            # print(i)
                            # print(residual[i])
                            #print('0',len(all_residual[0]))
                            all_residual[i] = all_residual[i] + residual[i]
                        residual = []
                        _MB = -1
                    for i in range(MB[0]-index-1):
                        residual = fin_res1(residual, -1)
                        #print(index)
                        for j in range(0, len(residual)):
                            # print(i)
                            # print(residual[i])
                            #print('1',len(all_residual[0]))
                            all_residual[j] = all_residual[j] + residual[j]

                        residual = []
                        _MB = -1

                elif MB[0]==1 and index==0:
                    #print(len(residual))
                    residual=fin_res1(residual,_MB)
                    all_residual = residual
                    #print(len(residual), len(residual[0]))
                    residual = []

                    _MB = -1
                else:

                    if MB[0]!=index:
                        residual = fin_res1(residual, _MB)
                        for i in range(0,len(residual)):
                            #print(i)
                            #print(residual[i])
                            #print('2',len(all_residual[0]))
                            all_residual[i]=all_residual[i]+residual[i]

                        residual=[]
                        _MB = -1



                while _MB!=MB[1]-1:
                    #print(0)
                    residual=residual+blank1

                    _MB+=1

                _MB+=1
                index=MB[0]
            if idx%38 in range(29,37) :
                _line=line.split(',')[:8]
                _line=[int(i) for i in _line]
                residual.append(_line)
                # print(len(_line))
                #print(len(residual))
    residual = fin_res1(residual, _MB)
    for i in range(len(residual)):
        all_residual[i] += residual[i]
    #print(len(all_residual), len(all_residual[0]))
    #print(index)
    if index!=39:
        residual=[]
        for i in range(49-index):
            residual = fin_res1(residual, -1)
            for i in range(0, len(residual)):
                # print(i)
                # print(residual[i])
                all_residual[i] = all_residual[i] + residual[i]

            residual = []
            _MB = -1
            index+=1
    # for i in all_residual:
    #     print(i)
    #print(len(all_residual),len(all_residual[0]))
    res=np.array(all_residual)
    return res


# image=Image.fromarray(res)
# image.show()
# image.save('D:\\video\\traffic1_res\\res'+name+'.PNG')
    #image=Image.open('D:\\video\\traffic1_res\\res.PNG')
    # image.show()
# print(len(all_residual),len(all_residual[0]))
edges=[]
for i in range(16,18):
    Y=get_res_pic_Y('D:\\VASRL\\result\\result\\res\\res'+str(i))
    U = get_res_pic_U('D:\\VASRL\\result\\result\\res\\res' + str(i))
    V = get_res_pic_V('D:\\VASRL\\result\\result\\res\\res' + str(i))
    print('1',Y.shape)


    print('2',U.shape)
    print(i)
    print('3',V.shape)
    rgb=yuv2rgb(Y,U,V,640,360)
    blur = cv2.GaussianBlur(rgb, (5, 5),
                            0)
    # 边缘检测
    # gray_lap = cv2.Laplacian(rgb, cv2.CV_16S, ksize=3)
    # dst = cv2.convertScaleAbs(gray_lap)
    edge = cv2.Canny(blur, 101, 255)
    print(edge)
    edges.append(edge)
    import matplotlib.pylab as plt
    #cv2.imwrite('D:\\video\\v_t\\p_res\\res'+str(i)+'.jpg',rgb)
    #s = cv2.imread('rgb1.jpg')
    #gray = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('img2',edge)
    # cv2.waitKey(0)
frame_diff = cv2.absdiff(edges[0], edges[1])
frame_diff = cv2.threshold(frame_diff, 101, 255,
                           cv2.THRESH_BINARY)[1]
cv2.imshow('img2', frame_diff)
cv2.waitKey(0)
def get_frame_feature(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5),0)
    #边缘检测
    gray_lap = cv2.Laplacian(blur, cv2.CV_16S, ksize=3)
    dst = cv2.convertScaleAbs(gray_lap)
    edge = cv2.Canny(dst, 101,  255)
    # cv2.imshow('img2',dst)
    # cv2.waitKey(0)
    return edge
#对边缘进行差值计算
def cal_frame_diff(edge, prev_edge):
    total_pixels = edge.shape[0] * edge.shape[1]
    frame_diff = cv2.absdiff(edge, prev_edge)
    frame_diff = cv2.threshold(frame_diff, 21, 255,
                               cv2.THRESH_BINARY)[1]
    changed_pixels = cv2.countNonZero(frame_diff)
    fraction_changed = changed_pixels / total_pixels
    return fraction_changed
import diff_processor
# dp=diff_processor.DiffProcessor.str2class('edge')(0)
# states=dp.get_diff_vector('D:\\video\\video_test.mp4')
# print(states)
#frame=cv2.imread('D:\\shiyan\\server\\server\\my_dds_sr_619\\dataset\\video_test\\src\\0000000004.png')
# frame1=cv2.imread('D:\\shiyan\\server\\server\\my_dds_sr_619\\dataset\\video_test\\src\\0000000005.png')
#
# frame=cv2.imread('D:\\video\\v_t\\p_res\\res4.jpg')
# frame1=cv2.imread('D:\\video\\v_t\\p_res\\res5.jpg')
# # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# # ret, thresh = cv2.threshold(gray,126, 255,cv2.THRESH_BINARY_INV) # 这里的第二个参数要调，是阈值！！
# import time
#
# fea=get_frame_feature(frame)
# start=time.time()
# for i in range(1000):
#
#     fea1=get_frame_feature(frame1)
#     a=cal_frame_diff(fea,fea1)
# end=time.time()
# print(end-start)
#
# a=2.18/1000
# b=3.68/1000
# print(a/b)
# # encoding:utf-8
# 用python3生成纯色图像

# import cv2
# import numpy
# #全黑的灰度图
# gray0=numpy.zeros((720,1280),dtype=numpy.uint8)
# #cv2.imshow('0',gray0)
# #全白的灰度图
# gray0[:,:]=255
# gray255=gray0[:,:]
# #cv2.imshow('255',gray255)
# #将灰度图转换成彩色图
# Img_rgb=cv2.cvtColor(gray255,cv2.COLOR_GRAY2RGB)
# # #将RGB通道全部置成0
# # Img_rgb[:,:,0:3]=0
# # cv2.imshow('(0,0,0)',Img_rgb)
# #将RGB通道全部置成255
# Img_rgb[:,:,0:3]=0
#
#
# # cv2.imshow('(255,255,255)',Img_rgb)
# # cv2.waitKey(0)
# cv2.imwrite("black.png",Img_rgb)



# import os
# import uuid
# from ffmpy import FFmpeg
#
#
# #调整视频大小
# def change_size(video_path: str, output_dir: str, width: int, height: int, bit_rate=2000):
#     ext = os.path.basename(video_path).strip().split('.')[-1]
#     if ext not in ['mp4']:
#         raise Exception('format error')
#     _result_path = os.path.join(
#         output_dir, '{}.{}'.format(
#             uuid.uuid1().hex, ext))
#     ff = FFmpeg(inputs={'{}'.format(video_path): None}, outputs={
#         _result_path: '-s {}*{} -b {}k'.format(width, height, bit_rate)})
#     print(ff.cmd)
#     ff.run()
#     return _result_path
#
# change_size("D:\\shiyan\\server\\server\\my_dds_sr_619\\dataset\\video_test\\src\\video_test.mp4","D:\\shiyan\\server\\server\\my_dds_sr_619\\dataset\\video_test"
#                 ,960,540)
# merge_rpn1=[]
# for gt_region in gt[fid]:
#     b = (max(0, gt_region.x - en / width), max(0, gt_region.y - en / height), gt_region.w + 2 * en / width,
#          gt_region.h + 2 * en / height)
#     for index, region in enumerate(merge_rpn1):
#
#         if overlap(b, region):
#             x1 = min(b[0], region[0])
#             y1 = min(b[1], region[1])
#             x2 = max(b[0] + b[2], region[0] + region[2])
#             y2 = max(b[1] + b[3], region[1] + region[3])
#             b = (x1, y1, x2 - x1, y2 - y1)
#             # merge_rpns[index]=b_new
#             merge_rpn1.remove(region)
#     merge_rpn1.append(b)
# merge_rpn=[]
# for id, (x, y, w, h) in enumerate(merge_rpn1):
#     if w * h > 0.5:
#         continue
#     b = (x,y,w,h)
#
#     x0 = max(int(b[0] * width), 0)
#     x1 = min(int((b[0] + b[2]) * width) + 2, width)
#     y0 = max(int(b[1] * height), 0)
#     y1 = min(int((b[1] + b[3]) * height) + 2, height)
#     w_.append(x1 - x0)
#     h_.append(y1 - y0)
#     merge_rpn.append(b)