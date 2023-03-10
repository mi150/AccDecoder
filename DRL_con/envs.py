from DRL_con.pro_mv import find_refer,find_re_refer,move_bbox,move_re_bbox
import copy
from sklearn.decomposition import PCA
from PIL import Image
import time as T
from DRL_con import diff_processor
#from backend.object_detector import Detector
import pickle
import torchvision.transforms as transforms
import numpy as np
from gym import spaces
from gym.utils import seeding
from dds_utils import (Results, read_results_dict, filter_bbox_group,evaluate, cleanup, Region,
                       compute_regions_size, merge_boxes_in_results, extract_images_from_video,calc_iou)
import time as T
import math
"""
thres=range(50,800,50)
thres=[i/1000 for i in thres]
# print(len(thres))
#r_thres=range(0,2200,75)
r_thres=range(0,3000,600)
"""
###
finish_idx=0
change_idx=[]
ground_path='video_test_gt'
#ground_path='video_test_gt'
raw_images_path='dataset\\video_test\\src\\'
thres=range(0,800,10)
#thres=range(0,800,40)
thres=[i/1000 for i in thres]
# print(len(thres))
#r_thres=range(0,2200,75)
r_thres=range(0,3000,100)
r_thres=[i/1000 for i in r_thres]


class Envs3:

    def __init__(self, height, width, length, states, diff_gop, times, result,h_result,res, features):
        # self.args=create_args()
        with open('I_frame.txt', "rb") as get_myprofile:
            self.I_frame = pickle.load(get_myprofile)
            print(self.I_frame)
        #print(self.I_frame)
        self.environment_title='video_V0'
        #self.action_space=spaces.Discrete(75)
        self.action_space = spaces.Box(
            low=np.array([-1]),
            high=np.array([1]),
            dtype=np.float32
        )
        self.f1list=[]
        high=np.zeros(128+60,dtype=np.float32)
        high=np.array([np.finfo(np.float32).max for _ in high])
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.seed()
        self.queue = 0
        self.height = height
        self.width = width
        self.length = length
        self.states = states
        self.diff_gop = diff_gop
        self.idx = 0
        self.goplen = 30
        self.a1 = 0.5
        self.a2 = 0.5
        self.state = []
        self.last_frame = None
        self.last_sr_frame=None
        self.diff_last = None
        self.l_result = None
        self._max_episode_steps=100
        #self.model = Detector()
        self.ground_truth_dict = read_results_dict(ground_path)
        self.times = times
        self.result = result
        self.show_results = Results()
        self.h_result=h_result
        self.res=res
        self.d_pro = diff_processor.DiffProcessor.str2class('edge')(0)
        self.features = features
        pca = PCA(n_components=128)  # ?????????
        self.pca = pca.fit(np.array(features))  # ????????????
        self.srl=[]
        self.dtl=[]
        self.s_a=0
        self.d_a=0
        # self.server=Server(self.args)
        # self.client=Client(self.args.hname, self.args, self.server)
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def reset(self):
        self.idx = 30
        self.queue = 0
        state_ = copy.deepcopy(self.states[0])
        res_ = copy.deepcopy(self.res[1:30])
        self.last_sr_frame=0
        #
        self.last_frame = 0
        self.diff_last = 0
        state_.insert(0, 0)
        res_.insert(0,0)
        #print(len(res_),self.idx)
        #
        self.state = np.array(state_)
        state_ += self.pca.transform([self.features[0]])[0].tolist()
        state_+=res_
        # state_.append(self.queue)
        # print(len(self.states[0]))
        # print(len(state_))
        return np.array(state_)
    # def delete(self):
    #     for r in self.l_result:
    #         label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)

    def get_SRlist(self,s_frames,thre):
        srlist=[]
        for fra in s_frames:
            if sum(self.res[self.last_sr_frame:fra+1])>thre or fra in change_idx:
                self.last_sr_frame=fra
                srlist.append(fra)
        return srlist
    def move(self,frame_idx):
        if frame_idx + 1 in self.I_frame:
            for r in self.l_result:
                label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                r = Region(frame_idx, x, y, w, h, conf, label,
                           0, origin="mpeg")
                
                #final_results.append(r)
        else:
            _d_result = Results()
            refer = find_refer(frame_idx + 1) - 1
            if refer + 1 == frame_idx:
                for r in self.l_result:
                    label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                    _x, _y = move_bbox(frame_idx + 1, x, y, w, h, refer + 1)
                    _x, _y = _x / 4, _y / 4
                    # _x, _y = 0, 0

                    r = Region(frame_idx, (x - _x / 960), (y - _y / 540), w, h, conf,
                               label,
                               0, origin="mpeg")
                    _d_result.append(r)

            elif refer == (find_refer(frame_idx) - 1) and find_refer(frame_idx) != -1:
                for r in self.l_result:
                    label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                    _x, _y = move_bbox(frame_idx + 1, x, y, w, h, refer + 1)
                    __x, __y = move_bbox(frame_idx, x, y, w, h, refer + 1)
                    if _x != 0 and _y != 0:
                        _x = _x - __x
                        _y = _y - __y
                    _x, _y = _x / 4, _y / 4

                    # _x, _y = 0, 0

                    r = Region(frame_idx, (x - _x / 960), (y - _y / 540), w, h, conf,
                               label,
                               0, origin="mpeg")
                    _d_result.append(r)

            else:

                for r in self.l_result:
                    label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                    _x, _y = move_re_bbox(frame_idx, x, y, w, h, frame_idx)
                    _x, _y = _x / 4, _y / 4

                    # _x, _y =0,0

                    r = Region(frame_idx, (x - _x / 960), (y - _y / 540), w, h, conf, label,
                               0, origin="mpeg")
                    _d_result.append(r)
                    # ??????????????????client??????origin??????????????????low-res??? ??????get_first_phase_results
                    #final_results.append(r)
            self.l_result = _d_result.regions_dict[frame_idx]

    def resize(self,r):
        if (r.x+r.w)>1:
            r.w=1-r.x
        if (r.y + r.h) > 1:
            r.w = 1 - r.y
        r.x=max(0,r.x)
        r.y=max(0,r.y)
        r.h=max(0,r.h)
        r.w=max(0,r.w)
        r.x = min(1, r.x)
        r.y = min(1, r.y)
        return r

    def isexist(self,region,result):
        for r in result:
            if calc_iou(r,region)>0.1:
                return True
        return False
    def isdelete(self,r):

        if r.w==0 or r.h==0:
            return False
        return True

    def clear_sr(self,frame_idx):
        res=[]
        for region in self.l_result:
            flag=False
            for h_region in self.result[frame_idx]:
                if filter_bbox_group(region, h_region, 0.1):
                    flag=True
                    break
            if not flag:
                res.append(region)
        self.l_result=res+self.result[frame_idx]

    def step(self, action):
        s_frames = self.d_pro.batch_diff_noobj_last(thres[int(action%80)], self.state, (self.idx - self.goplen),
                                                    self.goplen)
        #self.select.append(s_frames)r_thres[int(action/15)]
        SR_list=self.get_SRlist(s_frames,r_thres[int(action/80)])
        # s_frames=SR_list
        # s_frames = self.d_pro.batch_diff_noobj_last((action%0.1)*10, self.state, (self.idx - self.goplen),self.goplen)
        # #SR_list=self.get_SRlist(s_frames,r_thres[int(action[1])])
        # SR_list = self.get_SRlist(s_frames,(action-action%0.1)*3)
        #print(action,(action%0.1)*10,(action-action%0.1)*3)
        self.srl+=SR_list
        self.dtl+=s_frames

        # SR_list =[]
        # s=range(180)
        # s_frames=np.array(s).tolist()
        #s_frames=[0,30,60,90,120,150]
        #SR_list=s_frames
        # print("s_a",int(action/15))
        # print('SR',SR_list)
        # print('s_frames',s_frames)
        final_results = Results()
        # ???s_frames??????????????????f1score
        if s_frames:
            self.last_frame = s_frames[-1]
        time = 0
        #all_time = 0
        # print(self.idx-self.goplen)

        for frame_idx in range(self.idx - self.goplen, self.idx):
            if frame_idx in s_frames:
                if frame_idx in SR_list:
                    # if frame_idx==1065:
                    #     print(1065)
                    self.l_result = self.h_result[frame_idx]
                    #self.l_result = self.result[frame_idx]
                    for r in self.l_result:
                        label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                        # print(y)
                        r = Region(frame_idx, x, y, w, h, conf, label,
                                   0, origin="mpeg")
                        # ??????????????????client??????origin??????????????????low-res??? ??????get_first_phase_results
                        #if self.isdelete(r):
                        final_results.append(r)

                    time += 0.21+self.times[frame_idx]+0.02
                else:
                    #for r in self.self.result[frame_idx]:
                    _d_result = Results()
                    #self.move(frame_idx)
                    #self.clear_sr(frame_idx)
                    self.l_result = self.result[frame_idx]

                    time += self.times[frame_idx]+0.02
                    # for r in self.l_result:
                    #     label, conf, (x, y, w, h)=r.label,r.conf,(r.x,r.y,r.w,r.h)
                    #     r = Region(frame_idx, x, y, w, h, conf, label,
                    #                0, origin="mpeg")
                    #     r = self.resize(r)
                    #     if self.isdelete(r):
                    #         _d_result.append(r)
                    # self.l_result = merge_boxes_in_results(_d_result.regions_dict, 0.3, 0.3).regions_dict[frame_idx]

                    for r in self.l_result:
                        label, conf, (x, y, w, h)=r.label,r.conf,(r.x,r.y,r.w,r.h)
                        #print(y)
                        r = Region(frame_idx, x, y, w, h, conf, label,
                                   0, origin="mpeg")
                        r = self.resize(r)
                        # ??????????????????client??????origin??????????????????low-res??? ??????get_first_phase_results
                        if self.isdelete(r):
                            final_results.append(r)
                #all_time += self.times[frame_idx]
                continue
            # else:
            #     for r in self.l_result:
            #         label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
            #         # print(y)
            #         r = Region(frame_idx, x, y, w, h, conf, label,
            #                    0, origin="mpeg")
            #         r = self.resize(r)
            #         # ??????????????????client??????origin??????????????????low-res??? ??????get_first_phase_results
            #         if self.isdelete(r):
            #             final_results.append(r)
            #     continue
            #     for r in self.l_result:
            #         label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
            #         r = Region(frame_idx, x, y, w, h, conf, label,
            #                    0, origin="mpeg")
            #         # ??????????????????client??????origin??????????????????low-res??? ??????get_first_phase_results
            #         final_results.append(r)
            #     #all_time += self.times[frame_idx]
            #     continue
            ti6me1 = T.time()
            if frame_idx + 1 in self.I_frame:
                for r in self.l_result:
                    label, conf, (x, y, w, h)=r.label,r.conf,(r.x,r.y,r.w,r.h)
                    r = Region(frame_idx, x, y, w, h, conf, label,
                               0, origin="mpeg")
                    r = self.resize(r)
                    # ??????????????????client??????origin??????????????????low-res??? ??????get_first_phase_results
                    if self.isdelete(r):
                        final_results.append(r)
#0.7868061224489794
            else:

                _d_result = Results()
                refer = find_refer(frame_idx + 1) - 1
                if refer + 1 == frame_idx:
                    for r in self.l_result:
                        label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                        _x, _y = move_bbox(frame_idx + 1, x, y, w, h, refer + 1)
                        _x, _y = _x / 4, _y / 4
                        # _x, _y = 0, 0

                        r = Region(frame_idx, (x - _x / 960), (y - _y / 540), w, h, conf,
                                   label,
                                   0, origin="mpeg")
                        r = self.resize(r)
                        _d_result.append(r)
                        # ??????????????????client??????origin??????????????????low-res??? ??????get_first_phase_results
                        if self.isdelete(r):
                            final_results.append(r)

                elif refer == (find_refer(frame_idx) - 1) and find_refer(frame_idx)!=-1:
                    for r in self.l_result:
                        label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                        _x, _y = move_bbox(frame_idx + 1, x, y, w, h, refer + 1)
                        __x, __y = move_bbox(frame_idx, x, y, w, h, refer + 1)
                        if _x != 0 and _y != 0:
                            _x = _x - __x
                            _y = _y - __y
                        _x, _y = _x / 4, _y / 4
                        r = Region(frame_idx, (x - _x / 960), (y - _y / 540), w, h, conf,
                                   label,
                                   0, origin="mpeg")
                        r = self.resize(r)
                        _d_result.append(r)
                        
                        if self.isdelete(r):
                            final_results.append(r)

                else:

                    for r in self.l_result:
                        label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                        _x, _y = move_re_bbox(frame_idx, x, y, w, h, frame_idx)
                        _x, _y = _x / 4, _y / 4
                        r = Region(frame_idx, (x - _x / 960), (y - _y / 540), w, h, conf, label,
                                   0, origin="mpeg")
                        r = self.resize(r)
                        _d_result.append(r)
                        
                        if self.isdelete(r):
                            final_results.append(r)

                self.l_result = _d_result.regions_dict[frame_idx]
                # print('f_r',final_results.regions_dict)
            #all_time += self.times[frame_idx]
            # print('result',results)

        final_results = merge_boxes_in_results(final_results.regions_dict, 0.3, 0.3)

        #self.show_results.combine_results(final_results)
        tp, fp, fn, _, _, _, f1,f1_list = evaluate(
            self.idx - 1, final_results.regions_dict, self.ground_truth_dict,
            0.5, 0.5, 0.4, 0.4)
        self.f1list+=f1_list
        # final_results = Results()
        # # ???s_frames??????????????????f1score
        # #print('f1:',f1)
        # # print(self.idx-self.goplen)
        # for frame_idx in range(self.idx - self.goplen, self.idx):
        #
        #     results = self.result[frame_idx]
        #
        #     # all_time+=self.times[frame_idx]
        #     for r in results:
        #         label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
        #         r = Region(frame_idx, x, y, w, h, conf, label,
        #                    0, origin="mpeg")
        #         # ??????????????????client??????origin??????????????????low-res??? ??????get_first_phase_results
        #         final_results.append(r)
        #         # print('f_r',final_results.regions_dict)
        # tp, fp, fn, _, _, _, s_f1 = evaluate(
        #     self.idx - 1, final_results.regions_dict, self.ground_truth_dict,
        #     0.3, 0.5, 0.4, 0.4)

        # print(name,'f1',f1,"all_f1",s_f1,'Select',s_frames,len(s_frames),'time',time)

        reward = self.a1 * (f1)  - self.a2 * (1 if time>1-0.2 else 0)

        # ????????????
        # self.queue =max(0,(self.queue+time-1))
        ###
        # ???????????????gop??????feature??????
        state_ = copy.deepcopy(self.states[int(self.idx / self.goplen)])
        res_ = copy.deepcopy(self.res[self.idx-self.goplen+1:self.idx])
        if self.idx not in change_idx:
            if s_frames:
                # print('?????????',self.state,self.state[(s_frames[-1]%30)+1:])
                self.diff_last = np.sum(self.state[(s_frames[-1] % 30) + 1:]) + self.diff_gop[int(self.idx / 30) - 1]
                state_.insert(0, self.diff_last)
            else:
                self.diff_last += np.sum(self.state) + self.diff_gop[int(self.idx / 30) - 1]
                state_.insert(0, self.diff_last)
            res_.insert(0,sum(self.res[self.last_sr_frame+1:self.idx-self.goplen+1]))
        else:
            res_.insert(0,0)
            state_.insert(0, 0)
        ###
        self.state = np.array(state_)
        self.idx += self.goplen
        # return state_, reward, (self.idx==self.length)
        if self.idx == finish_idx:
            done = True
        else:
            done = False
        #print(len(res_), self.idx)
        state_ += self.pca.transform([self.features[self.idx - self.goplen]])[0].tolist()
        state_+=res_
        # state_.append(self.queue)
        # print(self.idx)
        # print('time',time)
        # print('f1',f1)

        return np.array(state_), reward, done, None,(f1,time)


#print((r_thres[14]))
#r_thres=[0,0,0,0]
# class Envs3:
#
#     def __init__(self, height,width,length,states,diff_gop,times,result,features):
#         #self.args=create_args()
#         self.queue=0
#         self.height=height
#         self.width=width
#         self.length=length
#         self.states=states
#         self.diff_gop=diff_gop
#         self.idx=0
#         self.goplen=30
#         self.a1=0.5
#         self.a2=0.5
#         self.state=[]
#         self.last_frame=None
#         self.diff_last=None
#         self.l_result=None
#         self.model=Detector()
#         self.ground_truth_dict = read_results_dict(ground_path)
#         self.times=times
#         self.result=result
#         self.d_pro=diff_processor.DiffProcessor.str2class('edge')(0)
#         self.features = features
#         pca = PCA(n_components=128)  # ?????????
#         self.pca = pca.fit(np.array(features))  # ????????????
#         self.trans=transforms.ToTensor()
#         #self.server=Server(self.args)
#         #self.client=Client(self.args.hname, self.args, self.server)
#
#     def reset(self):
#         self.idx=30
#         self.queue = 0
#         state_ =copy.deepcopy(self.states[0])
#         #
#         self.last_frame =0
#         self.diff_last =0
#         state_.insert(0,0)
#         #
#         self.state =np.array(state_)
#         image = Image.open(raw_images_path + f"{str(int(0)).zfill(10)}.png")
#         tensor = self.trans(image)
#         #state_ += self.features[0]
#         #state_.append(self.queue)
#         #print(len(self.states[0]))
#         return np.array(state_),tensor.unsqueeze(0)
#
#     def step(self,action):
#
#
#         #print(action,self.state)
#         #print(self.state)
#         s_frames=self.d_pro.batch_diff_noobj_last(np.array(thres)[action],self.state,(self.idx-self.goplen),self.goplen)
#         #
#         # print('state',self.state)
#         # print('s_frames',s_frames)
#         #s_frames = [0, 4, 7, 11, 13, 15, 17, 19, 21, 23, 24, 25, 27, 29, 30, 31, 32, 33, 35, 38, 42, 46, 50, 53, 56, 59, 60, 63, 66, 69, 74, 80, 87, 90, 102, 111, 116, 120, 131, 140, 147, 150, 180, 210, 240, 270, 300, 309, 316, 321, 325, 329, 330, 341, 348, 353, 358, 360, 367, 390, 405, 412, 416, 419, 420, 436, 443, 448, 450, 456, 459, 462, 465, 471, 479, 480, 486, 494, 500, 504, 507, 510, 516, 521, 529, 540, 546, 552, 559, 565, 570, 582, 589, 600, 607, 615, 621, 625, 629, 630, 637, 646, 655, 660, 667, 675, 687, 690, 697, 703, 710, 716, 720, 735, 747, 750, 759, 766, 775, 780, 792, 799, 804, 808, 810, 814, 818, 821, 825, 829, 833, 836, 839, 840, 842, 843, 845, 847, 849, 851, 854, 859, 867, 870, 880, 888, 894, 899, 900, 916, 927, 930, 940, 946, 952, 960, 970, 975, 984, 990, 1001, 1009, 1020, 1036, 1047, 1050, 1052, 1053, 1055, 1058, 1059, 1062, 1064, 1069, 1073, 1077, 1080, 1085, 1089, 1093, 1097, 1100, 1105, 1110, 1114, 1119, 1125, 1132, 1138, 1140, 1149, 1155, 1162, 1168, 1170, 1182, 1189, 1194, 1200, 1207, 1213, 1219, 1225, 1230, 1237, 1243, 1249, 1257, 1260, 1265, 1272, 1278, 1284, 1290, 1298, 1305, 1313, 1319, 1320, 1325, 1329, 1334, 1339, 1344, 1349, 1350, 1355, 1361, 1369, 1380, 1396, 1402, 1409, 1410, 1428, 1437, 1440, 1444, 1447, 1450, 1454, 1459, 1462, 1466, 1469, 1470, 1476, 1482, 1489, 1496, 1500, 1507, 1515, 1524, 1530, 1537, 1545, 1556, 1560, 1568, 1576, 1584, 1590, 1597, 1604, 1609, 1613, 1617, 1620, 1635, 1645, 1650, 1656, 1661, 1670, 1679, 1680, 1693, 1710, 1724, 1735, 1740, 1755, 1765, 1770, 1777, 1784, 1790, 1795, 1800, 1808, 1815, 1824, 1830, 1835, 1840, 1845, 1851, 1855, 1859, 1860, 1869, 1876, 1885, 1890, 1897, 1905, 1916, 1920, 1928, 1933, 1937, 1942, 1948, 1950, 1958, 1965, 1971, 1977, 1980, 1986, 1991, 1996, 2003, 2010, 2019, 2026, 2033, 2040, 2049, 2061, 2070, 2083, 2090, 2096, 2100, 2107, 2116, 2123, 2130, 2138, 2148, 2158, 2160, 2165, 2170, 2174, 2180, 2184, 2186, 2189, 2190, 2197, 2203, 2208, 2214, 2220, 2228, 2235, 2242, 2249, 2250, 2258, 2266, 2273, 2280, 2292, 2299, 2306, 2310, 2325, 2333, 2340, 2342, 2344, 2346, 2347, 2349, 2352, 2355, 2360, 2365, 2370, 2380, 2386, 2394, 2400, 2410, 2417, 2424, 2429, 2430, 2434, 2437, 2440, 2445, 2457, 2460, 2474, 2483, 2490, 2500, 2505, 2516, 2520, 2535, 2544, 2550, 2565, 2574, 2580, 2589, 2596, 2603, 2609, 2610, 2625, 2636, 2640, 2651, 2660, 2667, 2670, 2685, 2694, 2700, 2730, 2743, 2747, 2751, 2755, 2759, 2760, 2765, 2769, 2775, 2781, 2787, 2790, 2795, 2800, 2805, 2811, 2818, 2820, 2850, 2880, 2910, 2940, 2961]
#
#         final_results=Results()
#         #???s_frames??????????????????f1score
#         f1=0
#         s_f1=1
#         results=self.result[self.last_frame]
#         #print(self.last_frame)
#         if s_frames:
#             self.last_frame=s_frames[-1]
#         time=0
#         all_time=0
#         #print(self.idx-self.goplen)
#
#         for frame_idx in range(self.idx-self.goplen,self.idx):
#             if frame_idx in s_frames:
#                 #print(raw_images_path,f"{str(frame_idx).zfill(10)}.png")
#                 # frame=cv.imread(raw_images_path+f"{str(int(frame_idx)).zfill(10)}.png")
#                 # starttime = T.time()
#                 # results=self.model.infer(frame)
#                 # endtime = T.time()
#                 # time += float(endtime - starttime)
#                 # #print(results)
#                 results=self.result[frame_idx]
#                 time+=self.times[frame_idx]
#             all_time+=self.times[frame_idx]
#             #print('result',results)
#             for label, conf, (x, y, w, h) in results:
#                 r = Region(frame_idx, x, y, w, h, conf, label,
#                            0, origin="mpeg")
#                 # ??????????????????client??????origin??????????????????low-res??? ??????get_first_phase_results
#                 final_results.append(r)
#                 #print('f_r',final_results.regions_dict)
#         tp, fp, fn, _, _, _, f1 = evaluate(
#             self.idx-1, final_results.regions_dict, self.ground_truth_dict,
#             0.3, 0.5, 0.4, 0.4)
#         final_results=Results()
#         #???s_frames??????????????????f1score
#
#         #print(self.idx-self.goplen)
#         for frame_idx in range(self.idx-self.goplen,self.idx):
#
#             results=self.result[frame_idx]
#
#             #all_time+=self.times[frame_idx]
#             for label, conf, (x, y, w, h) in results:
#                 r = Region(frame_idx, x, y, w, h, conf, label,
#                            0, origin="mpeg")
#                 # ??????????????????client??????origin??????????????????low-res??? ??????get_first_phase_results
#                 final_results.append(r)
#                 #print('f_r',final_results.regions_dict)
#         tp, fp, fn, _, _, _, s_f1 = evaluate(
#             self.idx-1, final_results.regions_dict, self.ground_truth_dict,
#             0.3, 0.5, 0.4, 0.4)
#
#         #print(name,'f1',f1,"all_f1",s_f1,'Select',s_frames,len(s_frames),'time',time)
#         reward=self.a1*(f1)/s_f1-self.a2*(time+self.queue)/(all_time)
#
#         # ????????????
#         #self.queue =max(0,(self.queue+time-1))
#         ###
#         #???????????????gop??????feature??????
#         state_ = copy.deepcopy(self.states[int(self.idx/self.goplen)])
#         if self.idx not in [150,300,450,600,750,900,1050,1350,1680,2280,2700]:
#             if s_frames:
#                 #print('?????????',self.state,self.state[(s_frames[-1]%30)+1:])
#                 self.diff_last=np.sum(self.state[(s_frames[-1]%30)+1:])+self.diff_gop[int(self.idx/30)-1]
#                 state_.insert(0,self.diff_last)
#             else:
#                 self.diff_last+=np.sum(self.state)+self.diff_gop[int(self.idx/30)-1]
#                 state_.insert(0,self.diff_last)
#         else:
#             state_.insert(0, 0)
#         ###
#         self.state = np.array(state_)
#         self.idx += self.goplen
#         #return state_, reward, (self.idx==self.length)
#         if self.idx==2970:
#             done=True
#         else:
#             done=False
#         image = Image.open(raw_images_path + f"{str(int(self.idx-self.goplen)).zfill(10)}.png")
#         tensor=self.trans(image)
#         #state_ += self.features[self.idx - self.goplen]
#         #state_.append(self.queue)
#         return np.array(state_),tensor.unsqueeze(0),reward,done,None

# class Envs3:
#
#     def __init__(self, height, width, length, states, diff_gop, times, result,h_result,res, features):
#         # self.args=create_args()
#         with open('D:\\VASRL\\mv\\I_frame.txt', "rb") as get_myprofile:
#             self.I_frame = pickle.load(get_myprofile)
#         #print(self.I_frame)
#         self.environment_title='video_V0'
#         self.action_space=spaces.Discrete(75)
#         high=np.zeros(128+60,dtype=np.float32)
#         high=np.array([np.finfo(np.float32).max for _ in high])
#         self.observation_space = spaces.Box(-high, high, dtype=np.float32)
#         self.seed()
#         self.queue = 0
#         self.height = height
#         self.width = width
#         self.length = length
#         self.states = states[0]
#         self.statess=states
#         self.diff_gop = diff_gop
#         self.idx = 0
#         self.goplen = 30
#         self.a1 = 0.5
#         self.a2 = 0.5
#         self.speed=[]
#         self.density=[]
#         self.state = []
#         self.last_frame = None
#         self.last_sr_frame=None
#         self.diff_last = None
#         self.l_result = []
#         self._max_episode_steps=100
#         self.turn=1
#         #self.model = Detector()
#         self.ground_truth_dict = read_results_dict(ground_path)
#         self.times = times
#         self.result = result[0]
#         self.results = result
#         #self.show_results = Results()
#         self.select=[]
#         self.select_sr=[]
#         self.h_result=h_result[0]
#         self.h_results = h_result
#         self.res=res
#         self.d_pro = diff_processor.DiffProcessor.str2class('edge')(0)
#         self.features = features
#         pca = PCA(n_components=128)  # ?????????
#         self.pca = pca.fit(np.array(features))  # ????????????
#         # self.server=Server(self.args)
#         # self.client=Client(self.args.hname, self.args, self.server)
#     def seed(self, seed=None):
#         self.np_random, seed = seeding.np_random(seed)
#         return [seed]
#
#     def changeP(self):
#         self.state=self.states[self.turn]
#         self.h_result = self.h_results[self.turn]
#         self.result = self.results[self.turn]
#         self.idx = 30
#         self.queue = 0
#         state_ = copy.deepcopy(self.states[0])
#         res_ = copy.deepcopy(self.res[1:30])
#         self.last_sr_frame = 0
#         #
#         self.last_frame = 0
#         self.diff_last = 0
#         state_.insert(0, 0)
#         res_.insert(0, 0)
#         # print(len(res_),self.idx)
#         #
#         self.state = np.array(state_)
#         state_ += self.pca.transform([self.features[0]])[0].tolist()
#         state_ += res_
#         state_.append(self.turn)
#         # print(len(self.states[0]))
#         # print(len(state_))
#         return np.array(state_)
#
#     def reset(self):
#         self.turn=0
#         return self.changeP()
#     # def delete(self):
#     #     for r in self.l_result:
#     #         label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
#
#     def get_SRlist(self,s_frames,thre):
#         srlist=[]
#         for fra in s_frames:
#             if sum(self.res[self.last_sr_frame:fra+1])>thre or fra in [0,150, 300, 450, 600, 750, 900, 1050, 1350, 1680, 2280, 2700]:
#                 self.last_sr_frame=fra
#                 srlist.append(fra)
#         return srlist
#     def move(self,frame_idx):
#         if frame_idx + 1 in self.I_frame:
#             for r in self.l_result:
#                 label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
#                 r = Region(frame_idx, x, y, w, h, conf, label,
#                            0, origin="mpeg")
#                 # ??????????????????client??????origin??????????????????low-res??? ??????get_first_phase_results
#                 #final_results.append(r)
#         else:
#             _d_result = Results()
#             refer = find_refer(frame_idx + 1) - 1
#             if refer + 1 == frame_idx:
#                 for r in self.l_result:
#                     label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
#                     _x, _y = move_bbox(frame_idx + 1, x, y, w, h, refer + 1)
#                     _x, _y = _x / 4, _y / 4
#                     # _x, _y = 0, 0
#
#                     r = Region(frame_idx, (x - _x / 1280), (y - _y / 720), w, h, conf,
#                                label,
#                                0, origin="mpeg")
#                     _d_result.append(r)
#                     # ??????????????????client??????origin??????????????????low-res??? ??????get_first_phase_results
#                     #final_results.append(r)
#             elif refer == (find_refer(frame_idx) - 1) and find_refer(frame_idx) != -1:
#                 for r in self.l_result:
#                     label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
#                     _x, _y = move_bbox(frame_idx + 1, x, y, w, h, refer + 1)
#                     __x, __y = move_bbox(frame_idx, x, y, w, h, refer + 1)
#                     if _x != 0 and _y != 0:
#                         _x = _x - __x
#                         _y = _y - __y
#                     _x, _y = _x / 4, _y / 4
#
#                     # _x, _y = 0, 0
#
#                     r = Region(frame_idx, (x - _x / 1280), (y - _y / 720), w, h, conf,
#                                label,
#                                0, origin="mpeg")
#                     _d_result.append(r)
#                     # ??????????????????client??????origin??????????????????low-res??? ??????get_first_phase_results
#                     #final_results.append(r)
#             else:
#
#                 for r in self.l_result:
#                     label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
#                     _x, _y = move_re_bbox(frame_idx, x, y, w, h, frame_idx)
#                     _x, _y = _x / 4, _y / 4
#
#                     # _x, _y =0,0
#
#                     r = Region(frame_idx, (x - _x / 1280), (y - _y / 720), w, h, conf, label,
#                                0, origin="mpeg")
#                     _d_result.append(r)
#                     # ??????????????????client??????origin??????????????????low-res??? ??????get_first_phase_results
#                     #final_results.append(r)
#             self.l_result = _d_result.regions_dict[frame_idx]
#
#     def resize(self,r):
#         if (r.x+r.w)>1:
#             r.w=1-r.x
#         if (r.y + r.h) > 1:
#             r.w = 1 - r.y
#         r.x=max(0,r.x)
#         r.y=max(0,r.y)
#         r.h=max(0,r.h)
#         r.w=max(0,r.w)
#         r.x = min(1, r.x)
#         r.y = min(1, r.y)
#         return r
#
#     def isexist(self,region,result):
#         for r in result:
#             if calc_iou(r,region)>0.1:
#                 return True
#         return False
#     def isdelete(self,r):
#
#         if r.w==0 or r.h==0:
#             return False
#         return True
#     def clear_sr(self,frame_idx):
#         res=[]
#         for region in self.l_result:
#             flag=False
#             for h_region in self.h_result[frame_idx]:
#                 if filter_bbox_group(region, h_region, 0.005):
#                     flag=True
#                     break
#             if not flag:
#                 res.append(region)
#         self.l_result=res+self.h_result[frame_idx]
#     def step(self, action):
#         #print(action)
#         # print(action,self.state)
#         # print(self.state)np.array(r_thres)
#         #print(action)
#         #action=0
#         s_frames = self.d_pro.batch_diff_noobj_last(thres[np.array(action%15)], self.state, (self.idx - self.goplen),
#                                                     self.goplen)
#         #self.select.append(s_frames)
#         SR_list=self.get_SRlist(s_frames,r_thres[int(action/15)])
#         #SR_list=[]
#         #SR_list=s_frames
#         self.select+=s_frames
#         #SR_list=s_frames
#         self.select_sr+=SR_list
#         #SR_list=s_frames
#         # s_frames = self.d_pro.batch_diff_noobj_last(np.array(thres[10]), self.state, (self.idx - self.goplen),
#         #                                             self.goplen)
#         # SR_list=self.get_SRlist(s_frames,r_thres[int(4)])
#         # s_frames = self.d_pro.batch_diff_noobj_last(0.15, self.state, (self.idx - self.goplen),
#         #                                             self.goplen)
#         #print('thres:',0.5)
#         # print('state',self.state)
#         # print('s_frames',s_frames)
#         # print('SR_list',SR_list)
#         # s_frames = [0, 4, 7, 11, 13, 15, 17, 19, 21, 23, 24, 25, 27, 29, 30, 31, 32, 33, 35, 38, 42, 46, 50, 53, 56, 59, 60, 63, 66, 69, 74, 80, 87, 90, 102, 111, 116, 120, 131, 140, 147, 150, 180, 210, 240, 270, 300, 309, 316, 321, 325, 329, 330, 341, 348, 353, 358, 360, 367, 390, 405, 412, 416, 419, 420, 436, 443, 448, 450, 456, 459, 462, 465, 471, 479, 480, 486, 494, 500, 504, 507, 510, 516, 521, 529, 540, 546, 552, 559, 565, 570, 582, 589, 600, 607, 615, 621, 625, 629, 630, 637, 646, 655, 660, 667, 675, 687, 690, 697, 703, 710, 716, 720, 735, 747, 750, 759, 766, 775, 780, 792, 799, 804, 808, 810, 814, 818, 821, 825, 829, 833, 836, 839, 840, 842, 843, 845, 847, 849, 851, 854, 859, 867, 870, 880, 888, 894, 899, 900, 916, 927, 930, 940, 946, 952, 960, 970, 975, 984, 990, 1001, 1009, 1020, 1036, 1047, 1050, 1052, 1053, 1055, 1058, 1059, 1062, 1064, 1069, 1073, 1077, 1080, 1085, 1089, 1093, 1097, 1100, 1105, 1110, 1114, 1119, 1125, 1132, 1138, 1140, 1149, 1155, 1162, 1168, 1170, 1182, 1189, 1194, 1200, 1207, 1213, 1219, 1225, 1230, 1237, 1243, 1249, 1257, 1260, 1265, 1272, 1278, 1284, 1290, 1298, 1305, 1313, 1319, 1320, 1325, 1329, 1334, 1339, 1344, 1349, 1350, 1355, 1361, 1369, 1380, 1396, 1402, 1409, 1410, 1428, 1437, 1440, 1444, 1447, 1450, 1454, 1459, 1462, 1466, 1469, 1470, 1476, 1482, 1489, 1496, 1500, 1507, 1515, 1524, 1530, 1537, 1545, 1556, 1560, 1568, 1576, 1584, 1590, 1597, 1604, 1609, 1613, 1617, 1620, 1635, 1645, 1650, 1656, 1661, 1670, 1679, 1680, 1693, 1710, 1724, 1735, 1740, 1755, 1765, 1770, 1777, 1784, 1790, 1795, 1800, 1808, 1815, 1824, 1830, 1835, 1840, 1845, 1851, 1855, 1859, 1860, 1869, 1876, 1885, 1890, 1897, 1905, 1916, 1920, 1928, 1933, 1937, 1942, 1948, 1950, 1958, 1965, 1971, 1977, 1980, 1986, 1991, 1996, 2003, 2010, 2019, 2026, 2033, 2040, 2049, 2061, 2070, 2083, 2090, 2096, 2100, 2107, 2116, 2123, 2130, 2138, 2148, 2158, 2160, 2165, 2170, 2174, 2180, 2184, 2186, 2189, 2190, 2197, 2203, 2208, 2214, 2220, 2228, 2235, 2242, 2249, 2250, 2258, 2266, 2273, 2280, 2292, 2299, 2306, 2310, 2325, 2333, 2340, 2342, 2344, 2346, 2347, 2349, 2352, 2355, 2360, 2365, 2370, 2380, 2386, 2394, 2400, 2410, 2417, 2424, 2429, 2430, 2434, 2437, 2440, 2445, 2457, 2460, 2474, 2483, 2490, 2500, 2505, 2516, 2520, 2535, 2544, 2550, 2565, 2574, 2580, 2589, 2596, 2603, 2609, 2610, 2625, 2636, 2640, 2651, 2660, 2667, 2670, 2685, 2694, 2700, 2730, 2743, 2747, 2751, 2755, 2759, 2760, 2765, 2769, 2775, 2781, 2787, 2790, 2795, 2800, 2805, 2811, 2818, 2820, 2850, 2880, 2910, 2940, 2961]
#         # s_frames=[0, 26, 30, 37, 47, 58, 60, 65, 70, 75, 81, 86, 90, 100, 112, 120, 126, 131, 136, 141, 144, 148, 150, 180, 193, 207, 210, 240, 270, 287, 300, 305, 309, 313, 317, 321, 325, 329, 330, 337, 344, 350, 355, 360, 390, 398, 405, 411, 417, 420, 437, 446, 450, 455, 458, 462, 465, 469, 473, 476, 480, 483, 487, 491, 495, 499, 503, 506, 510, 513, 517, 521, 524, 528, 532, 535, 539, 540, 543, 547, 551, 554, 558, 561, 565, 569, 570, 573, 577, 580, 583, 586, 590, 594, 597, 600, 613, 625, 630, 643, 656, 660, 690, 699, 712, 720, 730, 742, 750, 761, 770, 779, 780, 795, 807, 810, 826, 840, 865, 870, 881, 891, 900, 905, 909, 912, 916, 919, 922, 924, 927, 929, 930, 934, 937, 941, 945, 948, 952, 955, 959, 960, 964, 967, 971, 975, 978, 983, 987, 990, 997, 1005, 1013, 1020, 1024, 1027, 1031, 1035, 1038, 1042, 1045, 1049, 1050, 1064, 1078, 1080, 1093, 1105, 1110, 1120, 1131, 1140, 1152, 1163, 1170, 1184, 1198, 1200, 1214, 1228, 1230, 1244, 1259, 1260, 1274, 1287, 1290, 1302, 1313, 1320, 1334, 1348, 1350, 1374, 1380, 1403, 1410, 1435, 1440, 1466, 1470, 1494, 1500, 1519, 1530, 1550, 1560, 1584, 1590, 1614, 1620, 1644, 1650, 1672, 1680, 1700, 1710, 1716, 1721, 1726, 1731, 1736, 1740, 1755, 1768, 1770, 1782, 1793, 1800, 1819, 1830, 1841, 1852, 1860, 1872, 1885, 1890, 1910, 1920, 1931, 1941, 1950, 1968, 1980, 1997, 2010, 2025, 2040, 2053, 2065, 2070, 2086, 2100, 2108, 2116, 2123, 2130, 2152, 2160, 2179, 2190, 2212, 2220, 2231, 2243, 2250, 2269, 2280, 2289, 2297, 2305, 2310, 2318, 2326, 2333, 2340, 2343, 2346, 2348, 2351, 2354, 2358, 2362, 2365, 2369, 2370, 2373, 2377, 2380, 2384, 2387, 2389, 2392, 2396, 2400, 2407, 2417, 2425, 2430, 2438, 2449, 2460, 2463, 2465, 2468, 2471, 2473, 2476, 2479, 2481, 2483, 2485, 2487, 2489, 2490, 2499, 2507, 2513, 2520, 2530, 2541, 2550, 2560, 2570, 2580, 2586, 2595, 2604, 2610, 2618, 2625, 2631, 2639, 2640, 2648, 2655, 2660, 2664, 2668, 2670, 2676, 2682, 2690, 2698, 2700, 2703, 2706, 2709, 2712, 2715, 2718, 2721, 2724, 2727, 2730, 2760, 2770, 2779, 2788, 2790, 2799, 2808, 2817, 2820, 2850, 2854, 2859, 2863, 2866, 2870, 2871, 2874, 2877, 2880, 2910, 2919, 2927, 2934, 2940, 2963]
#         # SR_list = self.get_SRlist(s_frames, np.array(r_thres)[2])
#         # SR_list = self.get_SRlist(s_frames, np.array(r_thres)[int(120 / 30)])
#         #s_frames=[0, 29, 41, 55, 58, 63, 68, 73, 77, 82, 87, 96, 100, 105, 113, 116, 145, 174, 203, 232, 261, 290, 319, 348, 358, 376, 377, 406, 435, 444, 450, 458, 464, 474, 483, 493, 505, 517, 522, 531, 543, 551, 560, 570, 580, 590, 600, 609, 619, 629, 638, 648, 658, 667, 678, 689, 695, 696, 701, 710, 718, 725, 732, 741, 749, 750, 754, 762, 769, 776, 783, 807, 812, 823, 836, 841, 852, 866, 870, 877, 887, 896, 899, 902, 911, 919, 927, 928, 937, 947, 957, 968, 979, 986, 995, 1005, 1015, 1029, 1041, 1044, 1055, 1063, 1070, 1073, 1093, 1102, 1119, 1131, 1147, 1160, 1183, 1189, 1210, 1218, 1239, 1247, 1270, 1276, 1298, 1305, 1327, 1334, 1353, 1363, 1392, 1416, 1421, 1449, 1450, 1479, 1505, 1508, 1537, 1566, 1595, 1624, 1653, 1680, 1682, 1705, 1711, 1740, 1765, 1769, 1782, 1794, 1798, 1816, 1827, 1851, 1856, 1881, 1885, 1906, 1914, 1927, 1940, 1943, 1963, 1972, 1990, 2001, 2022, 2030, 2051, 2059, 2069, 2077, 2085, 2088, 2111, 2117, 2123, 2129, 2135, 2141, 2146, 2158, 2167, 2175, 2187, 2197, 2204, 2211, 2218, 2225, 2232, 2233, 2248, 2262, 2272, 2281, 2291, 2298, 2305, 2312, 2319, 2320, 2329, 2338, 2348, 2349, 2357, 2366, 2374, 2378, 2388, 2396, 2407, 2419, 2430, 2436, 2447, 2459, 2465, 2471, 2476, 2481, 2486, 2490, 2494, 2510, 2521, 2523, 2533, 2548, 2552, 2564, 2577, 2581, 2587, 2597, 2606, 2610, 2620, 2627, 2635, 2639, 2649, 2658, 2663, 2667, 2668, 2674, 2681, 2689, 2697, 2713, 2726, 2734, 2741, 2749, 2754, 2755, 2784, 2807, 2813, 2842, 2850, 2859, 2866, 2871, 2877, 2884, 2891, 2897, 2900, 2929, 2934, 2939, 2945, 2950, 2956, 2958, 2986]
#
#         #SR_list=[]
#         final_results = Results()
#         # ???s_frames??????????????????f1score
#         f1 = 0
#         s_f1 = 1
#         # results=self.result[self.last_frame]
#         # print(self.last_frame)
#         if s_frames:
#             self.last_frame = s_frames[-1]
#         time = 0
#         #all_time = 0
#         # print(self.idx-self.goplen)
#         start=T.time()
#         speedidx=[]
#         densityidx=[]
#         for frame_idx in range(self.idx - self.goplen, self.idx):
#             bbox=0
#             if frame_idx in s_frames:
#                 if frame_idx in SR_list:
#                     self.l_result = self.h_result[frame_idx]
#                     #self.l_result = self.result[frame_idx]
#                     for r in self.l_result:
#                         r=self.resize(r)
#                         label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
#                         # print(y)
#                         if w*h<0.2:
#                             bbox+=w*h
#                         r = Region(frame_idx, x, y, w, h, conf, label,
#                                    0, origin="mpeg")
#                         # ??????????????????client??????origin??????????????????low-res??? ??????get_first_phase_results
#                         if self.isdelete(r):
#                             final_results.append(r)
#                         #self.show_results.append(r)
#                     time += 0.21+self.times[frame_idx]
#                     densityidx.append(bbox)
#                 else:
#                     #for r in self.self.result[frame_idx]:
#                     #self.move(frame_idx)
#                     _d_result=Results()
#                     self.l_result = self.result[frame_idx]
#                     #self.clear_sr(frame_idx)
#                     time += self.times[frame_idx]
#                     # for r in self.l_result:
#                     #     label, conf, (x, y, w, h)=r.label,r.conf,(r.x,r.y,r.w,r.h)
#                     #     r = Region(frame_idx, x, y, w, h, conf, label,
#                     #                0, origin="mpeg")
#                     #     r = self.resize(r)
#                     #     if self.isdelete(r):
#                     #         _d_result.append(r)
#                     # self.l_result = merge_boxes_in_results(_d_result.regions_dict, 0.3, 0.3).regions_dict[frame_idx]
#
#                 #print(self.l_result)
#                 #print(self.l_results,self.result[frame_idx])
#                     for r in self.l_result:
#                         r = self.resize(r)
#                         label, conf, (x, y, w, h)=r.label,r.conf,(r.x,r.y,r.w,r.h)
#                         #print(y)
#
#                         r = Region(frame_idx, x, y, w, h, conf, label,
#                                    0, origin="mpeg")
#                         # ??????????????????client??????origin??????????????????low-res??? ??????get_first_phase_results
#                         r = self.resize(r)
#                         if self.isdelete(r):
#                             final_results.append(r)
#                             #self.show_results.append(r)
#
#                 #all_time += self.times[frame_idx]
#                 continue
#             # else:
#             #     for r in self.l_result:
#             #         label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
#             #         r = Region(frame_idx, x, y, w, h, conf, label,
#             #                    0, origin="mpeg")
#             #         # ??????????????????client??????origin??????????????????low-res??? ??????get_first_phase_results
#             #         final_results.append(r)
#             #     #all_time += self.times[frame_idx]
#             #     continue
#             if frame_idx + 1 in self.I_frame:
#                 for r in self.l_result:
#                     r = self.resize(r)
#                     label, conf, (x, y, w, h)=r.label,r.conf,(r.x,r.y,r.w,r.h)
#                     r = Region(frame_idx, x, y, w, h, conf, label,
#                                0, origin="mpeg")
#                     r = self.resize(r)
#                     # ??????????????????client??????origin??????????????????low-res??? ??????get_first_phase_results
#                     if self.isdelete(r):
#                         final_results.append(r)
#                     #self.show_results.append(r)
#             else:
#                 _d_result = Results()
#                 refer = find_refer(frame_idx + 1) - 1
#                 if refer + 1 == frame_idx:
#                     for r in self.l_result:
#                         r = self.resize(r)
#                         label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
#                         _x, _y = move_bbox(frame_idx + 1, x, y, w, h, refer + 1)
#                         _x, _y = _x / 4, _y / 4
#                         speedidx.append(math.sqrt(_x*_x+_y*_y))
#                         # _x, _y = 0, 0
#                         # if self.idx==1500:
#                         #     print(_x,_y)
#                         r = Region(frame_idx, (x - _x / 1280), (y - _y / 720), w, h, conf,
#                                    label,
#                                    0, origin="mpeg")
#                         r = self.resize(r)
#                         _d_result.append(r)
#                         # ??????????????????client??????origin??????????????????low-res??? ??????get_first_phase_results
#                         if self.isdelete(r):
#                             final_results.append(r)
#                         #self.show_results.append(r)
#                 elif refer == (find_refer(frame_idx) - 1) and find_refer(frame_idx)!=-1:
#                     for r in self.l_result:
#                         r = self.resize(r)
#                         label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
#                         _x, _y = move_bbox(frame_idx + 1, x, y, w, h, refer + 1)
#                         __x, __y = move_bbox(frame_idx, x, y, w, h, refer + 1)
#                         if _x != 0 and _y != 0:
#                             _x = _x - __x
#                             _y = _y - __y
#                         _x, _y = _x / 4, _y / 4
#                         speedidx.append(math.sqrt(_x * _x + _y * _y))
#                         # if self.idx==1500:
#                         #     print(_x,_y)
#                         # _x, _y = 0, 0
#                         r = Region(frame_idx, (x - _x / 1280), (y - _y / 720), w, h, conf,
#                                    label,
#                                    0, origin="mpeg")
#                         r = self.resize(r)
#                         _d_result.append(r)
#                         # ??????????????????client??????origin??????????????????low-res??? ??????get_first_phase_results
#                         if self.isdelete(r):
#                             final_results.append(r)
#                         #self.show_results.append(r)
#                 else:
#
#                     for r in self.l_result:
#                         r = self.resize(r)
#                         label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
#                         _x, _y = move_re_bbox(frame_idx, x, y, w, h, frame_idx)
#                         _x, _y = _x / 4, _y / 4
#                         speedidx.append(math.sqrt(_x * _x + _y * _y))
#                         r = Region(frame_idx, (x - _x / 1280), (y - _y / 720), w, h, conf, label,
#                                    0, origin="mpeg")
#                         r = self.resize(r)
#                         _d_result.append(r)
#                         # ??????????????????client??????origin??????????????????low-res??? ??????get_first_phase_results
#                         if self.isdelete(r):
#                             final_results.append(r)
#                         #self.show_results.append(r)
#                 self.l_result = _d_result.regions_dict[frame_idx]
#                 # print('f_r',final_results.regions_dict)
#             #all_time += self.times[frame_idx]
#             # print('result',results)
#         #final_results = merge_boxes_in_results(final_results.regions_dict, 0.3, 0.3)
#         #self.density.append(sum(densityidx)/len(densityidx))
#         #self.speed.append(sum(speedidx)/len(speedidx))
#         tp, fp, fn, _, _, _, f1 = evaluate(
#             self.idx - 1, final_results.regions_dict, self.ground_truth_dict,
#             0.3, 0.5, 0.4, 0.4)
#         # final_results = Results()
#         # # ???s_frames??????????????????f1score
#         # #print('f1:',f1)
#         # # print(self.idx-self.goplen)
#         # for frame_idx in range(self.idx - self.goplen, self.idx):
#         #
#         #     results = self.result[frame_idx]
#         #
#         #     # all_time+=self.times[frame_idx]
#         #     for r in results:
#         #         label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
#         #         r = Region(frame_idx, x, y, w, h, conf, label,
#         #                    0, origin="mpeg")
#         #         # ??????????????????client??????origin??????????????????low-res??? ??????get_first_phase_results
#         #         final_results.append(r)
#         #         # print('f_r',final_results.regions_dict)
#         # tp, fp, fn, _, _, _, s_f1 = evaluate(
#         #     self.idx - 1, final_results.regions_dict, self.ground_truth_dict,
#         #     0.3, 0.5, 0.4, 0.4)
#
#         # print(name,'f1',f1,"all_f1",s_f1,'Select',s_frames,len(s_frames),'time',time)
#
#         reward = self.a1 * (f1)  - self.a2 * (1 if time>0.8 else 0)
#
#         # ????????????
#         # self.queue =max(0,(self.queue+time-1))
#         ###
#         # ???????????????gop??????feature??????
#         state_ = copy.deepcopy(self.states[int(self.idx / self.goplen)])
#         res_ = copy.deepcopy(self.res[self.idx-self.goplen+1:self.idx])
#         if self.idx not in [150, 300, 450, 600, 750, 900, 1050, 1350, 1680, 2280, 2700]:
#             if s_frames:
#                 # print('?????????',self.state,self.state[(s_frames[-1]%30)+1:])
#                 self.diff_last = np.sum(self.state[(s_frames[-1] % 30) + 1:]) + self.diff_gop[int(self.idx / 30) - 1]
#                 state_.insert(0, self.diff_last)
#             else:
#                 self.diff_last += np.sum(self.state) + self.diff_gop[int(self.idx / 30) - 1]
#                 state_.insert(0, self.diff_last)
#             res_.insert(0,sum(self.res[self.last_sr_frame+1:self.idx-self.goplen+1]))
#         else:
#             res_.insert(0,0)
#             state_.insert(0, 0)
#         ###
#         self.state = np.array(state_)
#         self.idx += self.goplen
#         # return state_, reward, (self.idx==self.length)
#         if self.idx == 2970 and self.turn==2:
#             done = True
#         elif self.idx==2970:
#             self.turn+=1
#             self.changeP()
#             done = False
#             return self.changeP(), reward, done, None
#         #print(len(res_), self.idx)
#         state_ += self.pca.transform([self.features[self.idx - self.goplen]])[0].tolist()
#         state_+=res_
#         state_.append(self.turn)
#
#         # print('f1',f1)
#         #print(self.idx,done)
#         return np.array(state_), reward, done, None
# class Envs2mv:
#
#     def __init__(self, height, width, length, states, diff_gop, times, result, features):
#         # self.args=create_args()
#         with open('D:\\video\\v_t\\pro_mv\\I_frame.txt', "rb") as get_myprofile:
#             self.I_frame = pickle.load(get_myprofile)
#         #print(self.I_frame)
#         self.queue = 0
#         self.height = height
#         self.width = width
#         self.length = length
#         self.states = states
#         self.diff_gop = diff_gop
#         self.idx = 0
#         self.goplen = 30
#         self.a1 = 0.5
#         self.a2 = 0.5
#         self.state = []
#         self.last_frame = None
#         self.diff_last = None
#         self.l_result = None
#         self.model = Detector()
#         self.ground_truth_dict = read_results_dict(ground_path)
#         self.times = times
#         self.result = result
#         self.d_pro = diff_processor.DiffProcessor.str2class('edge')(0)
#         self.features = features
#         pca = PCA(n_components=256)  # ?????????
#         self.pca = pca.fit(np.array(features))  # ????????????
#         # self.server=Server(self.args)
#         # self.client=Client(self.args.hname, self.args, self.server)
#
#     def reset(self):
#         self.idx = 30
#         self.queue = 0
#         state_ = copy.deepcopy(self.states[0])
#         #
#         self.last_frame = 0
#         self.diff_last = 0
#         state_.insert(0, 0)
#         #
#         self.state = np.array(state_)
#         state_ += self.pca.transform([self.features[0]])[0].tolist()
#         # state_.append(self.queue)
#         # print(len(self.states[0]))
#         # print(len(state_))
#         return np.array(state_)
#
#     def step(self, action):
#
#         # print(action,self.state)
#         # print(self.state)
#         s_frames = self.d_pro.batch_diff_noobj_last(np.array(thres)[action], self.state, (self.idx - self.goplen),
#                                                     self.goplen)
#         # s_frames = self.d_pro.batch_diff_noobj_last(0.15, self.state, (self.idx - self.goplen),
#         #                                             self.goplen)
#         #print('thres:',0.5)
#         # print('state',self.state)
#         # print('s_frames',s_frames)
#         # s_frames = [0, 4, 7, 11, 13, 15, 17, 19, 21, 23, 24, 25, 27, 29, 30, 31, 32, 33, 35, 38, 42, 46, 50, 53, 56, 59, 60, 63, 66, 69, 74, 80, 87, 90, 102, 111, 116, 120, 131, 140, 147, 150, 180, 210, 240, 270, 300, 309, 316, 321, 325, 329, 330, 341, 348, 353, 358, 360, 367, 390, 405, 412, 416, 419, 420, 436, 443, 448, 450, 456, 459, 462, 465, 471, 479, 480, 486, 494, 500, 504, 507, 510, 516, 521, 529, 540, 546, 552, 559, 565, 570, 582, 589, 600, 607, 615, 621, 625, 629, 630, 637, 646, 655, 660, 667, 675, 687, 690, 697, 703, 710, 716, 720, 735, 747, 750, 759, 766, 775, 780, 792, 799, 804, 808, 810, 814, 818, 821, 825, 829, 833, 836, 839, 840, 842, 843, 845, 847, 849, 851, 854, 859, 867, 870, 880, 888, 894, 899, 900, 916, 927, 930, 940, 946, 952, 960, 970, 975, 984, 990, 1001, 1009, 1020, 1036, 1047, 1050, 1052, 1053, 1055, 1058, 1059, 1062, 1064, 1069, 1073, 1077, 1080, 1085, 1089, 1093, 1097, 1100, 1105, 1110, 1114, 1119, 1125, 1132, 1138, 1140, 1149, 1155, 1162, 1168, 1170, 1182, 1189, 1194, 1200, 1207, 1213, 1219, 1225, 1230, 1237, 1243, 1249, 1257, 1260, 1265, 1272, 1278, 1284, 1290, 1298, 1305, 1313, 1319, 1320, 1325, 1329, 1334, 1339, 1344, 1349, 1350, 1355, 1361, 1369, 1380, 1396, 1402, 1409, 1410, 1428, 1437, 1440, 1444, 1447, 1450, 1454, 1459, 1462, 1466, 1469, 1470, 1476, 1482, 1489, 1496, 1500, 1507, 1515, 1524, 1530, 1537, 1545, 1556, 1560, 1568, 1576, 1584, 1590, 1597, 1604, 1609, 1613, 1617, 1620, 1635, 1645, 1650, 1656, 1661, 1670, 1679, 1680, 1693, 1710, 1724, 1735, 1740, 1755, 1765, 1770, 1777, 1784, 1790, 1795, 1800, 1808, 1815, 1824, 1830, 1835, 1840, 1845, 1851, 1855, 1859, 1860, 1869, 1876, 1885, 1890, 1897, 1905, 1916, 1920, 1928, 1933, 1937, 1942, 1948, 1950, 1958, 1965, 1971, 1977, 1980, 1986, 1991, 1996, 2003, 2010, 2019, 2026, 2033, 2040, 2049, 2061, 2070, 2083, 2090, 2096, 2100, 2107, 2116, 2123, 2130, 2138, 2148, 2158, 2160, 2165, 2170, 2174, 2180, 2184, 2186, 2189, 2190, 2197, 2203, 2208, 2214, 2220, 2228, 2235, 2242, 2249, 2250, 2258, 2266, 2273, 2280, 2292, 2299, 2306, 2310, 2325, 2333, 2340, 2342, 2344, 2346, 2347, 2349, 2352, 2355, 2360, 2365, 2370, 2380, 2386, 2394, 2400, 2410, 2417, 2424, 2429, 2430, 2434, 2437, 2440, 2445, 2457, 2460, 2474, 2483, 2490, 2500, 2505, 2516, 2520, 2535, 2544, 2550, 2565, 2574, 2580, 2589, 2596, 2603, 2609, 2610, 2625, 2636, 2640, 2651, 2660, 2667, 2670, 2685, 2694, 2700, 2730, 2743, 2747, 2751, 2755, 2759, 2760, 2765, 2769, 2775, 2781, 2787, 2790, 2795, 2800, 2805, 2811, 2818, 2820, 2850, 2880, 2910, 2940, 2961]
#
#         final_results = Results()
#         # ???s_frames??????????????????f1score
#         f1 = 0
#         s_f1 = 1
#         # results=self.result[self.last_frame]
#         # print(self.last_frame)
#         if s_frames:
#             self.last_frame = s_frames[-1]
#         time = 0
#         all_time = 0
#         # print(self.idx-self.goplen)
#
#         for frame_idx in range(self.idx - self.goplen, self.idx):
#             if frame_idx in s_frames:
#                 self.l_result = self.result[frame_idx]
#                 time += self.times[frame_idx]
#                 print(self.l_result)
#                 #print(self.l_results,self.result[frame_idx])
#                 for r in self.l_result:
#                     label, conf, (x, y, w, h)=r.label,r.conf,(r.x,r.y,r.w,r.h)
#                     r = Region(frame_idx, x, y, w, h, conf, label,
#                                0, origin="mpeg")
#                     # ??????????????????client??????origin??????????????????low-res??? ??????get_first_phase_results
#                     final_results.append(r)
#                 continue
#             if frame_idx + 1 in self.I_frame:
#                 for r in self.l_result:
#                     label, conf, (x, y, w, h)=r.label,r.conf,(r.x,r.y,r.w,r.h)
#                     r = Region(frame_idx, x.clip(0, 1), y.clip(0, 1), w, h, conf, label,
#                                0, origin="mpeg")
#                     # ??????????????????client??????origin??????????????????low-res??? ??????get_first_phase_results
#                     final_results.append(r)
#             else:
#                 _d_result = []
#                 refer = find_refer(frame_idx + 1) - 1
#                 if refer + 1 == frame_idx:
#                     for r in self.l_result:
#                         label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
#                         _x, _y = move_bbox(frame_idx + 1, x, y, w, h, refer + 1)
#                         _x, _y = _x / 4, _y / 4
#                         # _x, _y = 0, 0
#                         _d_result.append(
#                             (label, conf, ((x - _x / 1920).clip(0, 1), (y - _y / 1080).clip(0, 1), w, h)))
#                         r = Region(frame_idx, (x - _x / 1920).clip(0, 1), (y - _y / 1080).clip(0, 1), w, h, conf,
#                                    label,
#                                    0, origin="mpeg")
#                         # ??????????????????client??????origin??????????????????low-res??? ??????get_first_phase_results
#                         final_results.append(r)
#                 elif refer == (find_refer(frame_idx) - 1) and find_refer(frame_idx)!=-1:
#                     for r in self.l_result:
#                         label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
#                         _x, _y = move_bbox(frame_idx + 1, x, y, w, h, refer + 1)
#                         __x, __y = move_bbox(frame_idx, x, y, w, h, refer + 1)
#                         if _x != 0 and _y != 0:
#                             _x = _x - __x
#                             _y = _y - __y
#                         _x, _y = _x / 4, _y / 4
#
#                         # _x, _y = 0, 0
#                         _d_result.append(
#                             (label, conf, ((x - _x / 1920).clip(0, 1), (y - _y / 1080).clip(0, 1), w, h)))
#                         r = Region(frame_idx, (x - _x / 1920).clip(0, 1), (y - _y / 1080).clip(0, 1), w, h, conf,
#                                    label,
#                                    0, origin="mpeg")
#                         # ??????????????????client??????origin??????????????????low-res??? ??????get_first_phase_results
#                         final_results.append(r)
#                 else:
#
#                     for r in self.l_result:
#                         label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
#                         _x, _y = move_re_bbox(frame_idx, x, y, w, h, frame_idx)
#                         _x, _y = _x / 4, _y / 4
#
#                         # _x, _y =0,0
#                         _d_result.append((label, conf, ((x + _x / 1920).clip(0, 1), (y + _y / 1080).clip(0, 1), w, h)))
#                         r = Region(frame_idx, (x - _x / 1920).clip(0, 1), (y - _y / 1080).clip(0, 1), w, h, conf, label,
#                                    0, origin="mpeg")
#                         # ??????????????????client??????origin??????????????????low-res??? ??????get_first_phase_results
#                         final_results.append(r)
#
#                 self.l_result = _d_result
#
#                 # print('f_r',final_results.regions_dict)
#             all_time += self.times[frame_idx]
#             # print('result',results)
#
#         tp, fp, fn, _, _, _, f1 = evaluate(
#             self.idx - 1, final_results.regions_dict, self.ground_truth_dict,
#             0.3, 0.5, 0.4, 0.4)
#         final_results = Results()
#         # ???s_frames??????????????????f1score
#         #print('f1:',f1)
#         # print(self.idx-self.goplen)
#         for frame_idx in range(self.idx - self.goplen, self.idx):
#
#             results = self.result[frame_idx]
#
#             # all_time+=self.times[frame_idx]
#             for label, conf, (x, y, w, h) in results:
#                 r = Region(frame_idx, x, y, w, h, conf, label,
#                            0, origin="mpeg")
#                 # ??????????????????client??????origin??????????????????low-res??? ??????get_first_phase_results
#                 final_results.append(r)
#                 # print('f_r',final_results.regions_dict)
#         tp, fp, fn, _, _, _, s_f1 = evaluate(
#             self.idx - 1, final_results.regions_dict, self.ground_truth_dict,
#             0.3, 0.5, 0.4, 0.4)
#
#         # print(name,'f1',f1,"all_f1",s_f1,'Select',s_frames,len(s_frames),'time',time)
#         reward = self.a1 * (f1) / s_f1 - self.a2 * (time + self.queue) / (all_time)
#
#         # ????????????
#         # self.queue =max(0,(self.queue+time-1))
#         ###
#         # ???????????????gop??????feature??????
#         state_ = copy.deepcopy(self.states[int(self.idx / self.goplen)])
#         if self.idx not in [150, 300, 450, 600, 750, 900, 1050, 1350, 1680, 2280, 2700]:
#             if s_frames:
#                 # print('?????????',self.state,self.state[(s_frames[-1]%30)+1:])
#                 self.diff_last = np.sum(self.state[(s_frames[-1] % 30) + 1:]) + self.diff_gop[int(self.idx / 30) - 1]
#                 state_.insert(0, self.diff_last)
#             else:
#                 self.diff_last += np.sum(self.state) + self.diff_gop[int(self.idx / 30) - 1]
#                 state_.insert(0, self.diff_last)
#         else:
#             state_.insert(0, 0)
#         ###
#         self.state = np.array(state_)
#         self.idx += self.goplen
#         # return state_, reward, (self.idx==self.length)
#         if self.idx == 2970:
#             done = True
#         else:
#             done = False
#         state_ += self.pca.transform([self.features[self.idx - self.goplen]])[0].tolist()
#         # state_.append(self.queue)
#         return np.array(state_), reward, done, None

class Envs2mv:

    def __init__(self, height, width, length, states, diff_gop, times, result, features):
        # self.args=create_args()
        with open('D:\\video\\v_t\\pro_mv\\I_frame.txt', "rb") as get_myprofile:
            self.I_frame = pickle.load(get_myprofile)
        #print(self.I_frame)
        self.queue = 0
        self.height = height
        self.width = width
        self.length = length
        self.states = states
        self.diff_gop = diff_gop
        self.idx = 0
        self.goplen = 30
        self.a1 = 0.5
        self.a2 = 0.5
        self.state = []
        self.last_frame = None
        self.diff_last = None
        self.l_result = None
        #self.model = Detector()
        self.ground_truth_dict = read_results_dict(ground_path)
        self.times = times
        self.result = result
        self.d_pro = diff_processor.DiffProcessor.str2class('edge')(0)
        self.features = features
        pca = PCA(n_components=128)  # ?????????
        self.pca = pca.fit(np.array(features))  # ????????????
        # self.server=Server(self.args)
        # self.client=Client(self.args.hname, self.args, self.server)

    def reset(self):
        self.idx = 30
        self.queue = 0
        state_ = copy.deepcopy(self.states[0])
        #
        self.last_frame = 0
        self.diff_last = 0
        state_.insert(0, 0)
        #
        self.state = np.array(state_)
        state_ += self.pca.transform([self.features[0]])[0].tolist()
        # state_.append(self.queue)
        # print(len(self.states[0]))
        # print(len(state_))
        return np.array(state_)

    def step(self, action):

        # print(action,self.state)
        # print(self.state)

        s_frames = self.d_pro.batch_diff_noobj_last(np.array(thres)[action], self.state, (self.idx - self.goplen),
                                                    self.goplen)
        print(np.array(thres)[action])
        # s_frames = self.d_pro.batch_diff_noobj_last(0.15, self.state, (self.idx - self.goplen),
        #                                             self.goplen)
        #print('thres:',0.5)
        # print('state',self.state)
        # print('s_frames',s_frames)
        # s_frames = [0, 4, 7, 11, 13, 15, 17, 19, 21, 23, 24, 25, 27, 29, 30, 31, 32, 33, 35, 38, 42, 46, 50, 53, 56, 59, 60, 63, 66, 69, 74, 80, 87, 90, 102, 111, 116, 120, 131, 140, 147, 150, 180, 210, 240, 270, 300, 309, 316, 321, 325, 329, 330, 341, 348, 353, 358, 360, 367, 390, 405, 412, 416, 419, 420, 436, 443, 448, 450, 456, 459, 462, 465, 471, 479, 480, 486, 494, 500, 504, 507, 510, 516, 521, 529, 540, 546, 552, 559, 565, 570, 582, 589, 600, 607, 615, 621, 625, 629, 630, 637, 646, 655, 660, 667, 675, 687, 690, 697, 703, 710, 716, 720, 735, 747, 750, 759, 766, 775, 780, 792, 799, 804, 808, 810, 814, 818, 821, 825, 829, 833, 836, 839, 840, 842, 843, 845, 847, 849, 851, 854, 859, 867, 870, 880, 888, 894, 899, 900, 916, 927, 930, 940, 946, 952, 960, 970, 975, 984, 990, 1001, 1009, 1020, 1036, 1047, 1050, 1052, 1053, 1055, 1058, 1059, 1062, 1064, 1069, 1073, 1077, 1080, 1085, 1089, 1093, 1097, 1100, 1105, 1110, 1114, 1119, 1125, 1132, 1138, 1140, 1149, 1155, 1162, 1168, 1170, 1182, 1189, 1194, 1200, 1207, 1213, 1219, 1225, 1230, 1237, 1243, 1249, 1257, 1260, 1265, 1272, 1278, 1284, 1290, 1298, 1305, 1313, 1319, 1320, 1325, 1329, 1334, 1339, 1344, 1349, 1350, 1355, 1361, 1369, 1380, 1396, 1402, 1409, 1410, 1428, 1437, 1440, 1444, 1447, 1450, 1454, 1459, 1462, 1466, 1469, 1470, 1476, 1482, 1489, 1496, 1500, 1507, 1515, 1524, 1530, 1537, 1545, 1556, 1560, 1568, 1576, 1584, 1590, 1597, 1604, 1609, 1613, 1617, 1620, 1635, 1645, 1650, 1656, 1661, 1670, 1679, 1680, 1693, 1710, 1724, 1735, 1740, 1755, 1765, 1770, 1777, 1784, 1790, 1795, 1800, 1808, 1815, 1824, 1830, 1835, 1840, 1845, 1851, 1855, 1859, 1860, 1869, 1876, 1885, 1890, 1897, 1905, 1916, 1920, 1928, 1933, 1937, 1942, 1948, 1950, 1958, 1965, 1971, 1977, 1980, 1986, 1991, 1996, 2003, 2010, 2019, 2026, 2033, 2040, 2049, 2061, 2070, 2083, 2090, 2096, 2100, 2107, 2116, 2123, 2130, 2138, 2148, 2158, 2160, 2165, 2170, 2174, 2180, 2184, 2186, 2189, 2190, 2197, 2203, 2208, 2214, 2220, 2228, 2235, 2242, 2249, 2250, 2258, 2266, 2273, 2280, 2292, 2299, 2306, 2310, 2325, 2333, 2340, 2342, 2344, 2346, 2347, 2349, 2352, 2355, 2360, 2365, 2370, 2380, 2386, 2394, 2400, 2410, 2417, 2424, 2429, 2430, 2434, 2437, 2440, 2445, 2457, 2460, 2474, 2483, 2490, 2500, 2505, 2516, 2520, 2535, 2544, 2550, 2565, 2574, 2580, 2589, 2596, 2603, 2609, 2610, 2625, 2636, 2640, 2651, 2660, 2667, 2670, 2685, 2694, 2700, 2730, 2743, 2747, 2751, 2755, 2759, 2760, 2765, 2769, 2775, 2781, 2787, 2790, 2795, 2800, 2805, 2811, 2818, 2820, 2850, 2880, 2910, 2940, 2961]
        #s_frames=[0, 26, 30, 37, 47, 58, 60, 65, 70, 75, 81, 86, 90, 100, 112, 120, 126, 131, 136, 141, 144, 148, 150, 180, 193, 207, 210, 240, 270, 287, 300, 305, 309, 313, 317, 321, 325, 329, 330, 337, 344, 350, 355, 360, 390, 398, 405, 411, 417, 420, 437, 446, 450, 455, 458, 462, 465, 469, 473, 476, 480, 483, 487, 491, 495, 499, 503, 506, 510, 513, 517, 521, 524, 528, 532, 535, 539, 540, 543, 547, 551, 554, 558, 561, 565, 569, 570, 573, 577, 580, 583, 586, 590, 594, 597, 600, 613, 625, 630, 643, 656, 660, 690, 699, 712, 720, 730, 742, 750, 761, 770, 779, 780, 795, 807, 810, 826, 840, 865, 870, 881, 891, 900, 905, 909, 912, 916, 919, 922, 924, 927, 929, 930, 934, 937, 941, 945, 948, 952, 955, 959, 960, 964, 967, 971, 975, 978, 983, 987, 990, 997, 1005, 1013, 1020, 1024, 1027, 1031, 1035, 1038, 1042, 1045, 1049, 1050, 1064, 1078, 1080, 1093, 1105, 1110, 1120, 1131, 1140, 1152, 1163, 1170, 1184, 1198, 1200, 1214, 1228, 1230, 1244, 1259, 1260, 1274, 1287, 1290, 1302, 1313, 1320, 1334, 1348, 1350, 1374, 1380, 1403, 1410, 1435, 1440, 1466, 1470, 1494, 1500, 1519, 1530, 1550, 1560, 1584, 1590, 1614, 1620, 1644, 1650, 1672, 1680, 1700, 1710, 1716, 1721, 1726, 1731, 1736, 1740, 1755, 1768, 1770, 1782, 1793, 1800, 1819, 1830, 1841, 1852, 1860, 1872, 1885, 1890, 1910, 1920, 1931, 1941, 1950, 1968, 1980, 1997, 2010, 2025, 2040, 2053, 2065, 2070, 2086, 2100, 2108, 2116, 2123, 2130, 2152, 2160, 2179, 2190, 2212, 2220, 2231, 2243, 2250, 2269, 2280, 2289, 2297, 2305, 2310, 2318, 2326, 2333, 2340, 2343, 2346, 2348, 2351, 2354, 2358, 2362, 2365, 2369, 2370, 2373, 2377, 2380, 2384, 2387, 2389, 2392, 2396, 2400, 2407, 2417, 2425, 2430, 2438, 2449, 2460, 2463, 2465, 2468, 2471, 2473, 2476, 2479, 2481, 2483, 2485, 2487, 2489, 2490, 2499, 2507, 2513, 2520, 2530, 2541, 2550, 2560, 2570, 2580, 2586, 2595, 2604, 2610, 2618, 2625, 2631, 2639, 2640, 2648, 2655, 2660, 2664, 2668, 2670, 2676, 2682, 2690, 2698, 2700, 2703, 2706, 2709, 2712, 2715, 2718, 2721, 2724, 2727, 2730, 2760, 2770, 2779, 2788, 2790, 2799, 2808, 2817, 2820, 2850, 2854, 2859, 2863, 2866, 2870, 2871, 2874, 2877, 2880, 2910, 2919, 2927, 2934, 2940, 2963]
        final_results = Results()
        # ???s_frames??????????????????f1score
        f1 = 0
        s_f1 = 1
        # results=self.result[self.last_frame]
        # print(self.last_frame)
        if s_frames:
            self.last_frame = s_frames[-1]
        time = 0
        all_time = 0
        # print(self.idx-self.goplen)

        for frame_idx in range(self.idx - self.goplen, self.idx):
            if frame_idx in s_frames:
                self.l_result = self.result[frame_idx]
                time += self.times[frame_idx]
                #print(self.l_result)
                #print(self.l_results,self.result[frame_idx])
                for r in self.l_result:
                    label, conf, (x, y, w, h)=r.label,r.conf,(r.x,r.y,r.w,r.h)
                    r = Region(frame_idx, x, y, w, h, conf, label,
                               0, origin="mpeg")
                    # ??????????????????client??????origin??????????????????low-res??? ??????get_first_phase_results
                    final_results.append(r)
                all_time += self.times[frame_idx]
                continue
            # else:
            #     for r in self.l_result:
            #         label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
            #         r = Region(frame_idx, x, y, w, h, conf, label,
            #                    0, origin="mpeg")
            #         # ??????????????????client??????origin??????????????????low-res??? ??????get_first_phase_results
            #         final_results.append(r)
            #     all_time += self.times[frame_idx]
            #     continue
            if frame_idx + 1 in self.I_frame:
                for r in self.l_result:
                    label, conf, (x, y, w, h)=r.label,r.conf,(r.x,r.y,r.w,r.h)
                    r = Region(frame_idx, x, y, w, h, conf, label,
                               0, origin="mpeg")
                    # ??????????????????client??????origin??????????????????low-res??? ??????get_first_phase_results
                    final_results.append(r)
            else:
                _d_result = Results()
                refer = find_refer(frame_idx + 1) - 1
                if refer + 1 == frame_idx:
                    for r in self.l_result:
                        label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                        _x, _y = move_bbox(frame_idx + 1, x, y, w, h, refer + 1)
                        _x, _y = _x / 4, _y / 4
                        # _x, _y = 0, 0

                        r = Region(frame_idx, (x - _x / 1280), (y - _y / 720), w, h, conf,
                                   label,
                                   0, origin="mpeg")
                        _d_result.append(r)
                        # ??????????????????client??????origin??????????????????low-res??? ??????get_first_phase_results
                        final_results.append(r)
                elif refer == (find_refer(frame_idx) - 1) and find_refer(frame_idx)!=-1:
                    for r in self.l_result:
                        label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                        _x, _y = move_bbox(frame_idx + 1, x, y, w, h, refer + 1)
                        __x, __y = move_bbox(frame_idx, x, y, w, h, refer + 1)
                        if _x != 0 and _y != 0:
                            _x = _x - __x
                            _y = _y - __y
                        _x, _y = _x / 4, _y / 4

                        # _x, _y = 0, 0

                        r = Region(frame_idx, (x - _x / 1280), (y - _y / 720), w, h, conf,
                                   label,
                                   0, origin="mpeg")
                        _d_result.append(r)
                        # ??????????????????client??????origin??????????????????low-res??? ??????get_first_phase_results
                        final_results.append(r)
                else:

                    for r in self.l_result:
                        label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                        _x, _y = move_re_bbox(frame_idx, x, y, w, h, frame_idx)
                        _x, _y = _x / 4, _y / 4

                        # _x, _y =0,0

                        r = Region(frame_idx, (x - _x / 1280), (y - _y / 720), w, h, conf, label,
                                   0, origin="mpeg")
                        _d_result.append(r)
                        # ??????????????????client??????origin??????????????????low-res??? ??????get_first_phase_results
                        final_results.append(r)
                self.l_result = _d_result.regions_dict[frame_idx]
                # print('f_r',final_results.regions_dict)
            all_time += self.times[frame_idx]
            # print('result',results)

        tp, fp, fn, _, _, _, f1 = evaluate(
            self.idx - 1, final_results.regions_dict, self.ground_truth_dict,
            0.3, 0.5, 0.4, 0.4)
        #final_results = Results()
        # ???s_frames??????????????????f1score
        #print('f1:',f1)
        # print(self.idx-self.goplen)
        # for frame_idx in range(self.idx - self.goplen, self.idx):
        #
        #     results = self.result[frame_idx]
        #
        #     # all_time+=self.times[frame_idx]
        #     for r in results:
        #         label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
        #         r = Region(frame_idx, x, y, w, h, conf, label,
        #                    0, origin="mpeg")
        #         # ??????????????????client??????origin??????????????????low-res??? ??????get_first_phase_results
        #         final_results.append(r)
        #         # print('f_r',final_results.regions_dict)
        # tp, fp, fn, _, _, _, s_f1 = evaluate(
        #     self.idx - 1, final_results.regions_dict, self.ground_truth_dict,
        #     0.3, 0.5, 0.4, 0.4)

        # print(name,'f1',f1,"all_f1",s_f1,'Select',s_frames,len(s_frames),'time',time)
        reward = self.a1 * (f1)  - self.a2 * (time + self.queue) / (all_time)

        # ????????????
        # self.queue =max(0,(self.queue+time-1))
        ###
        # ???????????????gop??????feature??????
        state_ = copy.deepcopy(self.states[int(self.idx / self.goplen)])
        if self.idx not in [150, 300, 450, 600, 750, 900, 1050, 1350, 1680, 2280, 2700]:
            if s_frames:
                # print('?????????',self.state,self.state[(s_frames[-1]%30)+1:])
                self.diff_last = np.sum(self.state[(s_frames[-1] % 30) + 1:]) + self.diff_gop[int(self.idx / 30) - 1]
                state_.insert(0, self.diff_last)
            else:
                self.diff_last += np.sum(self.state) + self.diff_gop[int(self.idx / 30) - 1]
                state_.insert(0, self.diff_last)
        else:
            state_.insert(0, 0)
        ###
        self.state = np.array(state_)
        self.idx += self.goplen
        # return state_, reward, (self.idx==self.length)
        if self.idx == 2970:
            done = True
        else:
            done = False
        state_ += self.pca.transform([self.features[self.idx - self.goplen]])[0].tolist()
        # state_.append(self.queue)
        return np.array(state_), reward, done, None,(f1,time)

class Envs2:

    def __init__(self, height,width,length,states,diff_gop,times,result,features):
        #self.args=create_args()
        # with open('D:\\video\\pro_mv\\I_frame.txt', "rb") as get_myprofile:
        #     self.I_frame = pickle.load(get_myprofile)
        self.queue=0
        self.height=height
        self.width=width
        self.length=length
        self.states=states
        self.diff_gop=diff_gop
        self.idx=0
        self.goplen=30
        self.a1=0.5
        self.a2=0.5
        self.state=[]
        self.last_frame=None
        self.diff_last=None
        #self.l_result=None
        #self.model=Detector()
        self.ground_truth_dict = read_results_dict(ground_path)
        self.times=times
        self.result=result
        self.d_pro=diff_processor.DiffProcessor.str2class('edge')(0)
        self.features = features
        pca = PCA(n_components=256)  # ?????????
        self.pca = pca.fit(np.array(features))  # ????????????
        #self.server=Server(self.args)
        #self.client=Client(self.args.hname, self.args, self.server)

    def reset(self):
        self.idx=30
        self.queue = 0
        state_ =copy.deepcopy(self.states[0])
        #
        self.last_frame =0
        self.diff_last =0
        state_.insert(0,0)
        #
        self.state =np.array(state_)
        state_ += self.pca.transform([self.features[0]])[0].tolist()
        #state_.append(self.queue)
        #print(len(self.states[0]))
        #print(len(state_))
        return np.array(state_)

    def step(self,action):


        #print(action,self.state)
        #print(self.state)
        s_frames=self.d_pro.batch_diff_noobj_last(np.array(thres)[action],self.state,(self.idx-self.goplen),self.goplen)
        #
        # print('state',self.state)
        # print('s_frames',s_frames)
        #s_frames = [0, 4, 7, 11, 13, 15, 17, 19, 21, 23, 24, 25, 27, 29, 30, 31, 32, 33, 35, 38, 42, 46, 50, 53, 56, 59, 60, 63, 66, 69, 74, 80, 87, 90, 102, 111, 116, 120, 131, 140, 147, 150, 180, 210, 240, 270, 300, 309, 316, 321, 325, 329, 330, 341, 348, 353, 358, 360, 367, 390, 405, 412, 416, 419, 420, 436, 443, 448, 450, 456, 459, 462, 465, 471, 479, 480, 486, 494, 500, 504, 507, 510, 516, 521, 529, 540, 546, 552, 559, 565, 570, 582, 589, 600, 607, 615, 621, 625, 629, 630, 637, 646, 655, 660, 667, 675, 687, 690, 697, 703, 710, 716, 720, 735, 747, 750, 759, 766, 775, 780, 792, 799, 804, 808, 810, 814, 818, 821, 825, 829, 833, 836, 839, 840, 842, 843, 845, 847, 849, 851, 854, 859, 867, 870, 880, 888, 894, 899, 900, 916, 927, 930, 940, 946, 952, 960, 970, 975, 984, 990, 1001, 1009, 1020, 1036, 1047, 1050, 1052, 1053, 1055, 1058, 1059, 1062, 1064, 1069, 1073, 1077, 1080, 1085, 1089, 1093, 1097, 1100, 1105, 1110, 1114, 1119, 1125, 1132, 1138, 1140, 1149, 1155, 1162, 1168, 1170, 1182, 1189, 1194, 1200, 1207, 1213, 1219, 1225, 1230, 1237, 1243, 1249, 1257, 1260, 1265, 1272, 1278, 1284, 1290, 1298, 1305, 1313, 1319, 1320, 1325, 1329, 1334, 1339, 1344, 1349, 1350, 1355, 1361, 1369, 1380, 1396, 1402, 1409, 1410, 1428, 1437, 1440, 1444, 1447, 1450, 1454, 1459, 1462, 1466, 1469, 1470, 1476, 1482, 1489, 1496, 1500, 1507, 1515, 1524, 1530, 1537, 1545, 1556, 1560, 1568, 1576, 1584, 1590, 1597, 1604, 1609, 1613, 1617, 1620, 1635, 1645, 1650, 1656, 1661, 1670, 1679, 1680, 1693, 1710, 1724, 1735, 1740, 1755, 1765, 1770, 1777, 1784, 1790, 1795, 1800, 1808, 1815, 1824, 1830, 1835, 1840, 1845, 1851, 1855, 1859, 1860, 1869, 1876, 1885, 1890, 1897, 1905, 1916, 1920, 1928, 1933, 1937, 1942, 1948, 1950, 1958, 1965, 1971, 1977, 1980, 1986, 1991, 1996, 2003, 2010, 2019, 2026, 2033, 2040, 2049, 2061, 2070, 2083, 2090, 2096, 2100, 2107, 2116, 2123, 2130, 2138, 2148, 2158, 2160, 2165, 2170, 2174, 2180, 2184, 2186, 2189, 2190, 2197, 2203, 2208, 2214, 2220, 2228, 2235, 2242, 2249, 2250, 2258, 2266, 2273, 2280, 2292, 2299, 2306, 2310, 2325, 2333, 2340, 2342, 2344, 2346, 2347, 2349, 2352, 2355, 2360, 2365, 2370, 2380, 2386, 2394, 2400, 2410, 2417, 2424, 2429, 2430, 2434, 2437, 2440, 2445, 2457, 2460, 2474, 2483, 2490, 2500, 2505, 2516, 2520, 2535, 2544, 2550, 2565, 2574, 2580, 2589, 2596, 2603, 2609, 2610, 2625, 2636, 2640, 2651, 2660, 2667, 2670, 2685, 2694, 2700, 2730, 2743, 2747, 2751, 2755, 2759, 2760, 2765, 2769, 2775, 2781, 2787, 2790, 2795, 2800, 2805, 2811, 2818, 2820, 2850, 2880, 2910, 2940, 2961]

        final_results=Results()
        #???s_frames??????????????????f1score
        f1=0
        s_f1=1
        results=self.result[self.last_frame]
        #print(self.last_frame)
        if s_frames:
            self.last_frame=s_frames[-1]
        time=0
        all_time=0
        #print(self.idx-self.goplen)
        for frame_idx in range(self.idx-self.goplen,self.idx):
            if frame_idx in s_frames:
                #print(raw_images_path,f"{str(frame_idx).zfill(10)}.png")
                # frame=cv.imread(raw_images_path+f"{str(int(frame_idx)).zfill(10)}.png")
                # starttime = T.time()
                # results=self.model.infer(frame)
                # endtime = T.time()
                # time += float(endtime - starttime)
                # #print(results)
                results=self.result[frame_idx]
                time+=self.times[frame_idx]
            all_time+=self.times[frame_idx]
            #print('result',results)
            for label, conf, (x, y, w, h) in results:
                r = Region(frame_idx, x, y, w, h, conf, label,
                           0, origin="mpeg")
                # ??????????????????client??????origin??????????????????low-res??? ??????get_first_phase_results
                final_results.append(r)
                #print('f_r',final_results.regions_dict)


        tp, fp, fn, _, _, _, f1 = evaluate(
            self.idx-1, final_results.regions_dict, self.ground_truth_dict,
            0.3, 0.5, 0.4, 0.4)
        final_results=Results()
        #???s_frames??????????????????f1score

        #print(self.idx-self.goplen)
        for frame_idx in range(self.idx-self.goplen,self.idx):

            results=self.result[frame_idx]

            #all_time+=self.times[frame_idx]
            for label, conf, (x, y, w, h) in results:
                r = Region(frame_idx, x, y, w, h, conf, label,
                           0, origin="mpeg")
                # ??????????????????client??????origin??????????????????low-res??? ??????get_first_phase_results
                final_results.append(r)
                #print('f_r',final_results.regions_dict)
        tp, fp, fn, _, _, _, s_f1 = evaluate(
            self.idx-1, final_results.regions_dict, self.ground_truth_dict,
            0.3, 0.5, 0.4, 0.4)

        #print(name,'f1',f1,"all_f1",s_f1,'Select',s_frames,len(s_frames),'time',time)
        reward=self.a1*(f1)/s_f1-self.a2*(time+self.queue)/(all_time)
        #reward = self.a1 * (f1) / s_f1 - self.a2 * (time)
        # ????????????
        #self.queue =max(0,(self.queue+time-1))
        ###
        #???????????????gop??????feature??????
        state_ = copy.deepcopy(self.states[int(self.idx/self.goplen)])
        if self.idx not in [150,300,450,600,750,900,1050,1350,1680,2280,2700]:
            if s_frames:
                #print('?????????',self.state,self.state[(s_frames[-1]%30)+1:])
                self.diff_last=np.sum(self.state[(s_frames[-1]%30)+1:])+self.diff_gop[int(self.idx/30)-1]
                state_.insert(0,self.diff_last)
            else:
                self.diff_last+=np.sum(self.state)+self.diff_gop[int(self.idx/30)-1]
                state_.insert(0,self.diff_last)
        else:
            state_.insert(0, 0)
        ###
        self.state = np.array(state_)
        self.idx += self.goplen
        #return state_, reward, (self.idx==self.length)
        if self.idx==2970:
            done=True
        else:
            done=False
        state_ += self.pca.transform([self.features[self.idx - self.goplen]])[0].tolist()
        #state_.append(self.queue)
        return np.array(state_),reward,done,None


class Envs1:

    def __init__(self, height,width,length,states,diff_gop,times,result):
        #self.args=create_args()
        self.queue=0
        self.height=height
        self.width=width
        self.length=length
        self.states=states
        self.diff_gop=diff_gop
        self.idx=0
        self.goplen=30
        self.a1=0.5
        self.a2=0.5
        self.state=[]
        self.last_frame=None
        self.diff_last=None
        #self.model=Detector()
        self.ground_truth_dict = read_results_dict(ground_path)
        self.times=times
        self.result=result
        self.d_pro=diff_processor.DiffProcessor.str2class('edge')(0)
        #self.server=Server(self.args)
        #self.client=Client(self.args.hname, self.args, self.server)

    def reset(self):
        self.idx=30
        self.queue = 0
        state_ =copy.deepcopy(self.states[0])
        #
        self.last_frame =0
        self.diff_last =0
        state_.insert(0,0)
        #
        self.state =np.array(state_)
        #state_.append(self.queue)
        #print(len(self.states[0]))
        return np.array(state_)

    def step(self,action):


        #print(action,self.state)
        #print(self.state)
        s_frames=self.d_pro.batch_diff_noobj_last(np.array(thres)[action],self.state,(self.idx-self.goplen),self.goplen)
        #
        # print('state',self.state)
        # print('s_frames',s_frames)
        #s_frames = [0, 4, 7, 11, 13, 15, 17, 19, 21, 23, 24, 25, 27, 29, 30, 31, 32, 33, 35, 38, 42, 46, 50, 53, 56, 59, 60, 63, 66, 69, 74, 80, 87, 90, 102, 111, 116, 120, 131, 140, 147, 150, 180, 210, 240, 270, 300, 309, 316, 321, 325, 329, 330, 341, 348, 353, 358, 360, 367, 390, 405, 412, 416, 419, 420, 436, 443, 448, 450, 456, 459, 462, 465, 471, 479, 480, 486, 494, 500, 504, 507, 510, 516, 521, 529, 540, 546, 552, 559, 565, 570, 582, 589, 600, 607, 615, 621, 625, 629, 630, 637, 646, 655, 660, 667, 675, 687, 690, 697, 703, 710, 716, 720, 735, 747, 750, 759, 766, 775, 780, 792, 799, 804, 808, 810, 814, 818, 821, 825, 829, 833, 836, 839, 840, 842, 843, 845, 847, 849, 851, 854, 859, 867, 870, 880, 888, 894, 899, 900, 916, 927, 930, 940, 946, 952, 960, 970, 975, 984, 990, 1001, 1009, 1020, 1036, 1047, 1050, 1052, 1053, 1055, 1058, 1059, 1062, 1064, 1069, 1073, 1077, 1080, 1085, 1089, 1093, 1097, 1100, 1105, 1110, 1114, 1119, 1125, 1132, 1138, 1140, 1149, 1155, 1162, 1168, 1170, 1182, 1189, 1194, 1200, 1207, 1213, 1219, 1225, 1230, 1237, 1243, 1249, 1257, 1260, 1265, 1272, 1278, 1284, 1290, 1298, 1305, 1313, 1319, 1320, 1325, 1329, 1334, 1339, 1344, 1349, 1350, 1355, 1361, 1369, 1380, 1396, 1402, 1409, 1410, 1428, 1437, 1440, 1444, 1447, 1450, 1454, 1459, 1462, 1466, 1469, 1470, 1476, 1482, 1489, 1496, 1500, 1507, 1515, 1524, 1530, 1537, 1545, 1556, 1560, 1568, 1576, 1584, 1590, 1597, 1604, 1609, 1613, 1617, 1620, 1635, 1645, 1650, 1656, 1661, 1670, 1679, 1680, 1693, 1710, 1724, 1735, 1740, 1755, 1765, 1770, 1777, 1784, 1790, 1795, 1800, 1808, 1815, 1824, 1830, 1835, 1840, 1845, 1851, 1855, 1859, 1860, 1869, 1876, 1885, 1890, 1897, 1905, 1916, 1920, 1928, 1933, 1937, 1942, 1948, 1950, 1958, 1965, 1971, 1977, 1980, 1986, 1991, 1996, 2003, 2010, 2019, 2026, 2033, 2040, 2049, 2061, 2070, 2083, 2090, 2096, 2100, 2107, 2116, 2123, 2130, 2138, 2148, 2158, 2160, 2165, 2170, 2174, 2180, 2184, 2186, 2189, 2190, 2197, 2203, 2208, 2214, 2220, 2228, 2235, 2242, 2249, 2250, 2258, 2266, 2273, 2280, 2292, 2299, 2306, 2310, 2325, 2333, 2340, 2342, 2344, 2346, 2347, 2349, 2352, 2355, 2360, 2365, 2370, 2380, 2386, 2394, 2400, 2410, 2417, 2424, 2429, 2430, 2434, 2437, 2440, 2445, 2457, 2460, 2474, 2483, 2490, 2500, 2505, 2516, 2520, 2535, 2544, 2550, 2565, 2574, 2580, 2589, 2596, 2603, 2609, 2610, 2625, 2636, 2640, 2651, 2660, 2667, 2670, 2685, 2694, 2700, 2730, 2743, 2747, 2751, 2755, 2759, 2760, 2765, 2769, 2775, 2781, 2787, 2790, 2795, 2800, 2805, 2811, 2818, 2820, 2850, 2880, 2910, 2940, 2961]

        final_results=Results()
        #???s_frames??????????????????f1score
        f1=0
        s_f1=1
        results=self.result[self.last_frame]
        #print(self.last_frame)
        if s_frames:
            self.last_frame=s_frames[-1]
        time=0
        all_time=0
        #print(self.idx-self.goplen)

        for frame_idx in range(self.idx-self.goplen,self.idx):
            if frame_idx in s_frames:
                #print(raw_images_path,f"{str(frame_idx).zfill(10)}.png")
                # frame=cv.imread(raw_images_path+f"{str(int(frame_idx)).zfill(10)}.png")
                # starttime = T.time()
                # results=self.model.infer(frame)
                # endtime = T.time()
                # time += float(endtime - starttime)
                # #print(results)
                results=self.result[frame_idx]
                time+=self.times[frame_idx]
            all_time+=self.times[frame_idx]
            #print('result',results)
            for label, conf, (x, y, w, h) in results:
                r = Region(frame_idx, x, y, w, h, conf, label,
                           0, origin="mpeg")
                # ??????????????????client??????origin??????????????????low-res??? ??????get_first_phase_results
                final_results.append(r)
                #print('f_r',final_results.regions_dict)
        tp, fp, fn, _, _, _, f1 = evaluate(
            self.idx-1, final_results.regions_dict, self.ground_truth_dict,
            0.3, 0.5, 0.4, 0.4)
        final_results=Results()
        #???s_frames??????????????????f1score

        #print(self.idx-self.goplen)
        for frame_idx in range(self.idx-self.goplen,self.idx):

            results=self.result[frame_idx]

            #all_time+=self.times[frame_idx]
            for label, conf, (x, y, w, h) in results:
                r = Region(frame_idx, x, y, w, h, conf, label,
                           0, origin="mpeg")
                # ??????????????????client??????origin??????????????????low-res??? ??????get_first_phase_results
                final_results.append(r)
                #print('f_r',final_results.regions_dict)
        tp, fp, fn, _, _, _, s_f1 = evaluate(
            self.idx-1, final_results.regions_dict, self.ground_truth_dict,
            0.3, 0.5, 0.4, 0.4)

        #print(name,'f1',f1,"all_f1",s_f1,'Select',s_frames,len(s_frames),'time',time)
        reward=self.a1*(f1)/s_f1-self.a2*(time+self.queue)/(all_time)

        # ????????????
        #self.queue =max(0,(self.queue+time-1))
        ###
        #???????????????gop??????feature??????
        state_ = copy.deepcopy(self.states[int(self.idx/self.goplen)])
        if self.idx not in [150,300,450,600,750,900,1050,1350,1680,2280,2700]:
            if s_frames:
                #print('?????????',self.state,self.state[(s_frames[-1]%30)+1:])
                self.diff_last=np.sum(self.state[(s_frames[-1]%30)+1:])+self.diff_gop[int(self.idx/30)-1]
                state_.insert(0,self.diff_last)
            else:
                self.diff_last+=np.sum(self.state)+self.diff_gop[int(self.idx/30)-1]
                state_.insert(0,self.diff_last)
        else:
            state_.insert(0, 0)
        ###
        self.state = np.array(state_)
        self.idx += self.goplen
        #return state_, reward, (self.idx==self.length)
        if self.idx==2970:
            done=True
        else:
            done=False
        #state_.append(self.queue)
        return np.array(state_),reward,done,None,(f1,time)


class Envs:

    def __init__(self, height,width,length,video_path,times,result):
        #self.args=create_args()
        self.height=height
        self.width=width
        self.length=length
        self.path=video_path
        self.idx=0
        self.goplen=30
        self.a1=0.5
        self.a2=0.5
        self.state=[]
        #self.model=Detector()
        self.ground_truth_dict = read_results_dict(ground_path)
        self.times=times
        self.result=result
        #self.server=Server(self.args)
        #self.client=Client(self.args.hname, self.args, self.server)

    def reset(self):
        self.idx=30
        d_pro = diff_processor.DiffProcessor.str2class('area')(0)
        state_ = d_pro.get_diff_vector(self.path, 0, self.goplen)
        self.state =np.array(state_)
        return np.array(state_)

    def step(self,action):

        d_pro=diff_processor.DiffProcessor.str2class('area')(0)

        #print(self.state)
        s_frames=d_pro.batch_diff_noobj(thres[action],self.state,(self.idx-self.goplen),self.goplen)

        # print(s_frames)
        #s_frames = [0, 4, 7, 11, 13, 15, 17, 19, 21, 23, 24, 25, 27, 29, 30, 31, 32, 33, 35, 38, 42, 46, 50, 53, 56, 59, 60, 63, 66, 69, 74, 80, 87, 90, 102, 111, 116, 120, 131, 140, 147, 150, 180, 210, 240, 270, 300, 309, 316, 321, 325, 329, 330, 341, 348, 353, 358, 360, 367, 390, 405, 412, 416, 419, 420, 436, 443, 448, 450, 456, 459, 462, 465, 471, 479, 480, 486, 494, 500, 504, 507, 510, 516, 521, 529, 540, 546, 552, 559, 565, 570, 582, 589, 600, 607, 615, 621, 625, 629, 630, 637, 646, 655, 660, 667, 675, 687, 690, 697, 703, 710, 716, 720, 735, 747, 750, 759, 766, 775, 780, 792, 799, 804, 808, 810, 814, 818, 821, 825, 829, 833, 836, 839, 840, 842, 843, 845, 847, 849, 851, 854, 859, 867, 870, 880, 888, 894, 899, 900, 916, 927, 930, 940, 946, 952, 960, 970, 975, 984, 990, 1001, 1009, 1020, 1036, 1047, 1050, 1052, 1053, 1055, 1058, 1059, 1062, 1064, 1069, 1073, 1077, 1080, 1085, 1089, 1093, 1097, 1100, 1105, 1110, 1114, 1119, 1125, 1132, 1138, 1140, 1149, 1155, 1162, 1168, 1170, 1182, 1189, 1194, 1200, 1207, 1213, 1219, 1225, 1230, 1237, 1243, 1249, 1257, 1260, 1265, 1272, 1278, 1284, 1290, 1298, 1305, 1313, 1319, 1320, 1325, 1329, 1334, 1339, 1344, 1349, 1350, 1355, 1361, 1369, 1380, 1396, 1402, 1409, 1410, 1428, 1437, 1440, 1444, 1447, 1450, 1454, 1459, 1462, 1466, 1469, 1470, 1476, 1482, 1489, 1496, 1500, 1507, 1515, 1524, 1530, 1537, 1545, 1556, 1560, 1568, 1576, 1584, 1590, 1597, 1604, 1609, 1613, 1617, 1620, 1635, 1645, 1650, 1656, 1661, 1670, 1679, 1680, 1693, 1710, 1724, 1735, 1740, 1755, 1765, 1770, 1777, 1784, 1790, 1795, 1800, 1808, 1815, 1824, 1830, 1835, 1840, 1845, 1851, 1855, 1859, 1860, 1869, 1876, 1885, 1890, 1897, 1905, 1916, 1920, 1928, 1933, 1937, 1942, 1948, 1950, 1958, 1965, 1971, 1977, 1980, 1986, 1991, 1996, 2003, 2010, 2019, 2026, 2033, 2040, 2049, 2061, 2070, 2083, 2090, 2096, 2100, 2107, 2116, 2123, 2130, 2138, 2148, 2158, 2160, 2165, 2170, 2174, 2180, 2184, 2186, 2189, 2190, 2197, 2203, 2208, 2214, 2220, 2228, 2235, 2242, 2249, 2250, 2258, 2266, 2273, 2280, 2292, 2299, 2306, 2310, 2325, 2333, 2340, 2342, 2344, 2346, 2347, 2349, 2352, 2355, 2360, 2365, 2370, 2380, 2386, 2394, 2400, 2410, 2417, 2424, 2429, 2430, 2434, 2437, 2440, 2445, 2457, 2460, 2474, 2483, 2490, 2500, 2505, 2516, 2520, 2535, 2544, 2550, 2565, 2574, 2580, 2589, 2596, 2603, 2609, 2610, 2625, 2636, 2640, 2651, 2660, 2667, 2670, 2685, 2694, 2700, 2730, 2743, 2747, 2751, 2755, 2759, 2760, 2765, 2769, 2775, 2781, 2787, 2790, 2795, 2800, 2805, 2811, 2818, 2820, 2850, 2880, 2910, 2940, 2961]

        final_results=Results()
        #???s_frames??????????????????f1score
        f1=0
        s_f1=1
        results=None
        time=0
        all_time=1
        #print(self.idx-self.goplen)

        for frame_idx in range(self.idx-self.goplen,self.idx):
            if frame_idx in s_frames:
                #print(raw_images_path,f"{str(frame_idx).zfill(10)}.png")
                # frame=cv.imread(raw_images_path+f"{str(int(frame_idx)).zfill(10)}.png")
                # starttime = T.time()
                # results=self.model.infer(frame)
                # endtime = T.time()
                # time += float(endtime - starttime)
                # #print(results)
                results=self.result[frame_idx]
                time+=self.times[frame_idx]
            all_time+=self.times[frame_idx]
            for label, conf, (x, y, w, h) in results:
                r = Region(frame_idx, x, y, w, h, conf, label,
                           0, origin="mpeg")
                # ??????????????????client??????origin??????????????????low-res??? ??????get_first_phase_results
                final_results.append(r)
                #print('f_r',final_results.regions_dict)
        tp, fp, fn, _, _, _, f1 = evaluate(
            self.idx-1, final_results.regions_dict, self.ground_truth_dict,
            0.3, 0.5, 0.4, 0.4)
        final_results=Results()
        #???s_frames??????????????????f1score

        #print(self.idx-self.goplen)
        for frame_idx in range(self.idx-self.goplen,self.idx):

            results=self.result[frame_idx]

            #all_time+=self.times[frame_idx]
            for label, conf, (x, y, w, h) in results:
                r = Region(frame_idx, x, y, w, h, conf, label,
                           0, origin="mpeg")
                # ??????????????????client??????origin??????????????????low-res??? ??????get_first_phase_results
                final_results.append(r)
                #print('f_r',final_results.regions_dict)
        tp, fp, fn, _, _, _, s_f1 = evaluate(
            self.idx-1, final_results.regions_dict, self.ground_truth_dict,
            0.3, 0.5, 0.4, 0.4)




        print('f1',f1,"all_f1",s_f1,'Select',s_frames,len(s_frames),'time',time)
        reward=self.a1*(f1)/s_f1-self.a2*time/(all_time)
        ra=self.a1*(f1)/s_f1
        rt=reward-ra
        ###
        state_ = np.array(d_pro.get_diff_vector(self.path, self.idx, self.goplen))
        #state_=self.state
        self.state = state_
        self.idx += self.goplen
        #return state_, reward, (self.idx==self.length)
        if self.idx==2970:
            done=True
        else:
            done=False
        return state_,reward,done,None,ra,rt





# import pickle
# with open("features_res50_1000.txt", "rb") as get_myprofile:
#     features = pickle.load(get_myprofile)
# #pca = PCA(n_components=256)
# X=[]
# X=features
# X=np.array(X)
# print(X.shape)
# pca = PCA(n_components=256) #?????????
# pca = pca.fit(X) #????????????
# for i in range(3000):
#     x=[]
#     x.append((features[0]))
#     newX = pca.transform(x)
#     print('finish',i,newX[0].tolist())

# import numpy as np
# from sklearn import decomposition,datasets
# iris=datasets.load_iris()#????????????
# X=iris['data']
# model=decomposition.PCA(n_components=2)
# print(X.shape)
# model.fit(X)
# X_new=model.fit_transform(X)
# Maxcomponent=model.components_
# ratio=model.explained_variance_ratio_
# score=model.score(X)
# print('??????????????????:',X_new)
# print('?????????????????????????????????:',Maxcomponent)
# print('?????????????????????????????????:',ratio)
# print('???????????????log???????????????:',score)
# print('?????????:',model.singular_values_)
# print('???????????????:',model.noise_variance_)


# pi0=nn.LSTM(30, 128, batch_first = False)
# print(pi0.mode)
# for name, param in pi0.named_parameters():
#     if name.startswith("weight"):
#         nn.init.xavier_normal_(param)
#     else:
#         nn.init.zeros_(param)
#

# d_pro = diff_processor.DiffProcessor.str2class('edge')(0)
#
# states,diff_gop= d_pro.get_all_diff_vector(
#     "D:\\shiyan\\server\\server\\my_dds_sr_619\\dataset\\video_test\\src\\video_test.mp4", 30)

