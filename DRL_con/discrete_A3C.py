import pickle
import torchvision.models as models
from torch.autograd import Variable
import diff_processor
import torch
import torch.nn as nn
from utils import v_wrap, set_init, push_and_pull, record,set_init_LSTM
import torch.nn.functional as F
import torch.multiprocessing as mp
from shared_adam import SharedAdam
from envs import Envs,Envs1,Envs2,Envs2mv,Envs3
import os
from dds_utils import merge_boxes_in_results,Region,Results,read_results_dict
from backend.object_detector import Detector
import cv2 as cv
import time as T
from multicate import multi_categorical_maker
os.environ["OMP_NUM_THREADS"] = "20"

UPDATE_GLOBAL_ITER = 5
GAMMA = 0.9
MAX_EP = 2000

N_S=30+128+30
N_A=4









class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        #self.pi0=nn.LSTM(s_dim, 128, batch_first = False)
        self.pi1 = nn.Linear(self.s_dim, 128)
        self.pi2 = nn.Linear(128, a_dim)
        #self.pi3 = nn.Linear(128, a_dim)
        #self.v0=nn.LSTM(s_dim, 128, batch_first = False)
        self.v1 = nn.Linear(s_dim, 128)
        self.v2 = nn.Linear(128, 1)
#        set_init_LSTM([self.pi0,self.v0])
        #self.vgg16 = models.vgg16(pretrained=True)
        set_init([self.pi1, self.pi2, self.v1, self.v2])
        self.distribution = torch.distributions.Categorical
        #self.distribution=multi_categorical_maker([15, 15])
        # self.a_hx = None
        # self.a_cx = None
        # self.c_hx = None
        # self.c_cx = None
        self.a_hidden=None
        self.c_hidden = None

        # self.a_hidden = self.init_hidden()
        # self.c_hidden = self.init_hidden()

    def init_hidden(self):
        # 开始时刻, 没有隐状态
        # 关于维度设置的详情,请参考 Pytorch 文档
        # 各个维度的含义是 (Seguence, minibatch_size, hidden_dim)
        return (torch.zeros(128).unsqueeze(0).unsqueeze(0),
                torch.zeros(128).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        # print(x.size())
        #print("x",x)
        #x = x.view(len(x), 1, -1)
        #print(x)
        #print('a',(x))

        ###
        #pi0,self.a_hidden=self.pi0(x,self.a_hidden)
        #print(self.a_hidden)
        #self.a_hidden=(self.a_hidden[0].data,self.a_hidden[1].data)
        #pi0,_=self.pi0(x)
        #print("tuple",pi0)
        #pi0,_=self.pi0(x)
        #print("notuple",pi0)
        #pi0=torch.tanh(pi0)
        pi1 = torch.tanh(self.pi1(x))
        #print("pi1",pi1)
        logits = self.pi2(pi1)
        # logits1 = self.pi3(pi1)
        # logits=torch.cat([logits0, logits1], dim=1)
        #print('log',logits,logits.shape)
        #logits1 = self.pi3(pi1)
        #logits=
        #v0,self.c_hidden=self.v0(x,self.c_hidden)
        #self.c_hidden=(self.c_hidden[0].data,self.c_hidden[1].data)
        #self.c_hidden = self.c_hidden.data
        #v0,_=self.v0(x)
        v1 = torch.tanh(self.v1(x))
        values = self.v2(v1)
        return logits, values

    def choose_action(self, s):
        self.eval()
        logits, _ = self.forward(s)
        #logits0,logits1=logits.split([50, 25], dim=1)
        #print("logits,]",logits)
        #print(logits)
        #prob = F.softmax(logits[0], dim=1).data
        prob = F.softmax(logits, dim=1).data
        # print(torch.rand(2, 75).shape)
        # print(prob.shape)
        m = self.distribution(prob)
        #print(m.sample().numpy()[0])
        # prob1 = F.softmax(logits1, dim=1).data
        # m1 = self.distribution(prob1)
        #print("prob",prob)
        #print((m0.sample().numpy()[0],m1.sample().numpy()[0]))
        #return int(torch.argmax(prob))
        return m.sample().numpy()[0]

    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)
        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss


class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name,env):
        super(Worker, self).__init__()
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = Net(N_S, N_A)           # local network
        #self.env = gym.make('CartPole-v0').unwrapped
        self.env=env
    def run(self):
        total_step = 1

        #print(self.g_ep.value, ' start')
        while self.g_ep.value < MAX_EP:
            #print(self.name, ' into1')

            s = self.env.reset()
            # self.lnet.a_hx = torch.zeros(128).unsqueeze(0).unsqueeze(0)
            # self.lnet.a_cx = torch.zeros(128).unsqueeze(0).unsqueeze(0)
            # self.lnet.c_hx = torch.zeros(128).unsqueeze(0).unsqueeze(0)
            # self.lnet.c_cx = torch.zeros(128).unsqueeze(0).unsqueeze(0)
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            while True:
                #print(self.name, ' into2')
                # if self.name == 'w00':
                #     self.env.render()
                #print(s,"v_wrap(s)",v_wrap(s))
                a = self.lnet.choose_action(v_wrap(s[None,:]))
                s_, r, done, _ = self.env.step(a)
                #print("all",s_,r,done,a)
                #if done: r = -1
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync

                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)

                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break
                s = s_
                total_step += 1

        #print(self.name, ' over')
        self.res_queue.put(None)


if __name__ == "__main__":
    gnet = Net(N_S, N_A)        # global network
    gnet.share_memory()         # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.92, 0.999))      # global optimizer
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()


    times = []
    result = []
    #model = Detector()
    # for frame_idx in range(0, 181):
    #     # print(raw_images_path,f"{str(frame_idx).zfill(10)}.png")
    #     frame = cv.imread(raw_images_path + f"{str(int(frame_idx)).zfill(10)}.png")
    #     starttime = T.time()
    #     result.append(model.infer(frame))
    #     #result.append([])
    #     endtime = T.time()
    #     times.append(float(endtime - starttime))
    #env = Envs(1080, 1920, 2000, "trafficcam_1.mp4")
    with open("dds_results540.txt", "rb") as get_myprofile:
        result = pickle.load(get_myprofile)
    with open("dds_results.txt", "rb") as get_myprofile:
        h_result = pickle.load(get_myprofile)
    with open("times.txt", "rb") as get_myprofile:
        times = pickle.load(get_myprofile)
    with open("features.txt", "rb") as get_myprofile:
        features = pickle.load(get_myprofile)
    d_pro = diff_processor.DiffProcessor.str2class('edge')(0)
    
    with open("res.txt", "rb") as get_myprofile:
        res = pickle.load(get_myprofile)
        #print('ok')
    #print(res)

    states,diff_gop= d_pro.get_all_diff_vector(
        "video_test.mp4", 30)
    for id in result.regions_dict:
        for r in result.regions_dict[id]:
            r.y = (r.y - 0.077) / 0.84583333
    # result = read_results_dict('video_test_yolo_540p')
    # result = merge_boxes_in_results(result, 0.3, 0.3)
    # h_result = read_results_dict('video_test_yolo_720p')
    # result.regions_dict+=h_result
    # print(result.regions_dict)
    #print(h_result)
    # parallel training

    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i, Envs3(540, 960, 2000,
                                                                             states,diff_gop,
                                                                             times, result.regions_dict,
                                                                             h_result.regions_dict,
                                                                             res,features)) for i in range(8)]

    start = T.time()
    [w.start() for w in workers]

    res = []                    # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]
    end = T.time()
    print('t',(end-start))
    torch.save(gnet, "newa3c_4.pth")
    import matplotlib

    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    print(res)
    # with open("a3c_res4.txt", "wb") as myprofile:
    #     pickle.dump(res, myprofile)
    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()

