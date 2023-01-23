import pickle
import torchvision.models as models
from torch.autograd import Variable
import diff_processor
import torch
import torch.nn as nn
from utils import v_wrap, set_init, push_and_pull, record,set_init_LSTM,push_and_pull_vgg
import torch.nn.functional as F
import torch.multiprocessing as mp
from shared_adam import SharedAdam
from envs import Envs,Envs1,Envs2,Envs3
import os
from backend.object_detector import Detector
import cv2 as cv
import time as T
os.environ["OMP_NUM_THREADS"] = "20"

UPDATE_GLOBAL_ITER = 1
GAMMA = 0.9
MAX_EP = 10000
#
# env = gym.make('CartPole-v0')
# N_S = env.observation_space.shape[0]
# N_A = env.action_space.n
# print(N_S,N_A)
#raw_images_path='D:\\shiyan\\server\\server\\my_dds_sr_619\\dataset\\trafficcam_1\\src\\'

N_S=30+1000
N_A=30



class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.vgg1=models.vgg16(pretrained=True)
        self.vgg2 = models.vgg16(pretrained=True)
        self.pi1 = nn.Linear(self.s_dim, 128)
        self.pi2 = nn.Linear(128, a_dim)
        self.v1 = nn.Linear(s_dim, 128)
        self.v2 = nn.Linear(128, 1)
        set_init([self.pi1, self.pi2, self.v1, self.v2])
        self.distribution = torch.distributions.Categorical


        # self.a_hidden = self.init_hidden()
        # self.c_hidden = self.init_hidden()

    def init_hidden(self):
        # 开始时刻, 没有隐状态
        # 关于维度设置的详情,请参考 Pytorch 文档
        # 各个维度的含义是 (Seguence, minibatch_size, hidden_dim)
        return (torch.zeros(128).unsqueeze(0).unsqueeze(0),
                torch.zeros(128).unsqueeze(0).unsqueeze(0))

    def forward(self, s,tensor):
        #print(Variable(tensor).size())
        #tensor = torch.unsqueeze(tensor, dim=0)

        f1=self.vgg1(Variable(tensor))
        x1=torch.cat([f1,s],dim=1)
        pi1 = torch.tanh(self.pi1(x1))
        #print("pi1",pi1)
        logits = self.pi2(pi1)
        f2=self.vgg2(Variable(tensor))
        x2=torch.cat([f2,s],dim=1)
        v1 = torch.tanh(self.v1(x2))
        values = self.v2(v1)
        return logits, values

    def choose_action(self, s,tensor):
        self.eval()
        logits, _ = self.forward(s,tensor)
        #print("logits,]",logits)
        #print(logits)
        #prob = F.softmax(logits[0], dim=1).data
        prob = F.softmax(logits, dim=1).data
        m = self.distribution(prob)
        #print("prob",prob)
        #return int(torch.argmax(prob))
        return m.sample().numpy()[0]

    def loss_func(self, s,tensor, a, v_t):
        self.train()
        logits, values = self.forward(s,tensor)
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

            s,tensor = self.env.reset()
            # self.lnet.a_hx = torch.zeros(128).unsqueeze(0).unsqueeze(0)
            # self.lnet.a_cx = torch.zeros(128).unsqueeze(0).unsqueeze(0)
            # self.lnet.c_hx = torch.zeros(128).unsqueeze(0).unsqueeze(0)
            # self.lnet.c_cx = torch.zeros(128).unsqueeze(0).unsqueeze(0)
            buffer_s, buffer_t,buffer_a, buffer_r = [], [], [],[]
            ep_r = 0.
            while True:
                #print(self.name, ' into2')
                # if self.name == 'w00':
                #     self.env.render()
                #print(s,"v_wrap(s)",v_wrap(s))
                a = self.lnet.choose_action(v_wrap(s[None,:]),tensor)
                s_,tensor_, r, done, _ = self.env.step(a)
                #print("all",s_,r,done,a)
                #if done: r = -1
                ep_r += r
                buffer_a.append(a)
                buffer_t.append(tensor)
                buffer_s.append(s)
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync

                    push_and_pull_vgg(self.opt, self.lnet, self.gnet, done, s_,tensor_, buffer_s,buffer_t, buffer_a, buffer_r, GAMMA)

                    buffer_s, buffer_t,buffer_a, buffer_r = [], [], [],[]

                    if done:  # done and print information
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break
                s = s_
                tensor=tensor_
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
    model = Detector()
    # for frame_idx in range(0, 181):
    #     # print(raw_images_path,f"{str(frame_idx).zfill(10)}.png")
    #     frame = cv.imread(raw_images_path + f"{str(int(frame_idx)).zfill(10)}.png")
    #     starttime = T.time()
    #     result.append(model.infer(frame))
    #     #result.append([])
    #     endtime = T.time()
    #     times.append(float(endtime - starttime))
    #env = Envs(1080, 1920, 2000, "D:\\shiyan\\server\\server\\my_dds_sr_619\\visual_f1score\\trafficcam_1.mp4")
    with open("result_long.txt", "rb") as get_myprofile:
        result = pickle.load(get_myprofile)
    with open("times.txt", "rb") as get_myprofile:
        times = pickle.load(get_myprofile)
    with open("features_res50_1000.txt", "rb") as get_myprofile:
        features = pickle.load(get_myprofile)
    d_pro = diff_processor.DiffProcessor.str2class('edge')(0)

    states,diff_gop= d_pro.get_all_diff_vector(
        "D:\\shiyan\\server\\server\\my_dds_sr_619\\dataset\\video_test\\src\\video_test.mp4", 30)
    # print('states',states)
    # print('diff_gop',diff_gop)
    #print(len(states[0]))
    # parallel training
    #workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i,Envs(720, 1280, 2000, "D:\\shiyan\\server\\server\\my_dds_sr_619\\dataset\\video_test\\src\\video_test.mp4",times,result)) for i in range(2)]
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i, Envs3(720, 1280, 2000,
                                                                             states,diff_gop,
                                                                             times, result,features)) for i in range(1)]
    # workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i, Envs1(720, 1280, 2000,
    #                                                                         states,diff_gop,
    #                                                                         times, result)) for i in range(20)]

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
    torch.save(gnet, "/DRL_con\\long_model_连续state_vgg16_1000_反.pth")
    import matplotlib

    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    print(res)
    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()
