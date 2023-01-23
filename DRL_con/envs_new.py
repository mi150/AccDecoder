import copy
from sklearn.decomposition import PCA
from DRL_con.diff_processor import DiffProcessor
import pickle
import numpy as np
from gym import spaces
from gym.utils import seeding
from DRL.utils import v_wrap
from discrete_A3C import Net
import torch
thres=range(50,800,50)
thres=[i/1000 for i in thres]
r_thres=range(0,3000,600)
r_thres=[i/1000 for i in r_thres]




class Envs:

    def __init__(self, height, width, length):
        self.d_pro = DiffProcessor.str2class('edge')(0)
        with open("features.txt", "rb") as get_myprofile:
            feature = pickle.load(get_myprofile)
        with open("res.txt", "rb") as get_myprofile:
            res = pickle.load(get_myprofile)
        self.features=feature

        states, diff_gop = self.d_pro.get_all_diff_vector(
            "D:\\VASRL\\server\\server\\my_dds_sr_619\\dataset\\video_test\\src\\video_test.mp4", 30)
        #print(self.I_frame)
        self.environment_title='video_V0'
        self.action_space=spaces.Discrete(75)
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
        self.speed=[]
        self.density=[]
        self.state = []
        self.last_frame = None
        self.last_sr_frame=None
        self.diff_last = None
        self.l_result = []
        self._max_episode_steps=100
        self.res=res
        pca = PCA(n_components=128)  # 实例化
        self.pca = pca.fit(np.array(self.features))  # 拟合模型

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

    def get_SRlist(self,s_frames,thre):
        srlist=[]
        for fra in s_frames:
            if sum(self.res[self.last_sr_frame:fra+1])>thre or fra in [0,150, 300, 450, 600, 750, 900, 1050, 1350, 1680, 2280, 2700]:
                self.last_sr_frame=fra
                srlist.append(fra)
        return srlist

    def step(self, action):

        s_frames = self.d_pro.batch_diff_noobj_last(thres[np.array(action%15)], self.state, (self.idx - self.goplen),
                                                    self.goplen)
        SR_list=self.get_SRlist(s_frames,r_thres[int(action/15)])
        if self.idx == 2970:
            done = True
            return [],done,SR_list,s_frames
        else:
            done=False
        if s_frames:
            self.last_frame = s_frames[-1]
        state_ = copy.deepcopy(self.states[int(self.idx / self.goplen)])
        res_ = copy.deepcopy(self.res[self.idx-self.goplen+1:self.idx])
        if self.idx not in [150, 300, 450, 600, 750, 900, 1050, 1350, 1680, 2280, 2700]:
            if s_frames:
                # print('余下的',self.state,self.state[(s_frames[-1]%30)+1:])
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

        state_ += self.pca.transform([self.features[self.idx - self.goplen]])[0].tolist()
        state_+=res_
        return np.array(state_),done,SR_list,s_frames



env=Envs(1,1,1)
model=torch.load("newa3c.pth")
s=env.reset()
done=False
while not done:
    a = model.choose_action(v_wrap(s[None, :]))
    s,done,SR_list,Infer_list= env.step(a)


