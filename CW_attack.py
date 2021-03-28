import torch
from data_loader import test_loader,train_loader,test_dataset,train_dataset
import tqdm
import time
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torchvision import transforms
from Classfication_model import THNet,MNIST_Model
from torchattacks.attacks.cw import CW
'''
    https://github.com/FenHua/Adversarial-Examples
    https://github.com/Carco-git/CW_Attack_on_MNIST/blob/master/CW_Attack_l2.ipynb
'''
class CW_attack():
    def __init__(self, model, is_target=True, target_label=9, default_label=8, c=1e-3, kappa=0, steps=1000,
                 lr=0.01, binary_number=9, min_loss=100000, device=None):
        super(CW_attack, self).__init__()
        self.model = model
        self.is_target = is_target          # 是否进行目标攻击
        self.target_label = target_label    # 攻击的目标标签
        self.c = c                          # 常数 初始化为0.001
        self.kappa = kappa                  # f函数中的k 论文中取值为0
        self.steps = steps                  # 攻击的迭代次数
        self.lr = lr                        # 学习率
        self.binary_number = binary_number  # 二分搜索次数
        self.minc = 0                       # 寻找c值 [0.01-100]
        self.maxc = 1e10
        self.min_loss = min_loss            # 寻找最小的扰动，loss初始为大值
        self.default_label = default_label  # 如果不是target攻击，默认向label=8进行转换
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # 计算L2范数的另一种方法（手写）
    # def L2_norm(self, img, adv_img):
    #     MSEloss = nn.MSELoss(reduction='none')
    #     Flatten = nn.Flatten()
    #     current_l2 = MSEloss(Flatten(adv_img),Flatten(img)).sum(dim=1)
    #     L2_loss = current_l2.sum()
    #     return L2_loss

    # # 计算l2范数
    def L2_norm(self, img, adv_img):
        return torch.dist(img, adv_img, p=2)

    def f_loss(self, outputs, target_label):
        one_hot_labels = torch.eye(len(outputs[0]))[target_label]
        # function = max( (max( Z(x')i - Z(x')t ), -k)
        real_pre = torch.max(outputs * one_hot_labels)
        print(outputs)
        print(one_hot_labels)
        print(outputs * one_hot_labels)
        print("real_pre:", real_pre)
        adv_pre = torch.max((1 - one_hot_labels) * outputs, dim=1)[0]
        print("adv_pre:", adv_pre)
        if(self.is_target):
            func = torch.max(adv_pre - real_pre, torch.Tensor(-self.kappa))
        else:
            func = torch.max(real_pre - adv_pre, torch.Tensor(-self.kappa))
        print("func:", func)
        return func

    def arctanh(self, imgs, eps=1e-6):
        imgs = torch.clamp(imgs, max=1, min=0)
        print("arctanh:",imgs)
        imgs = imgs*(1-eps)
        return 0.5*torch.log10((1.0+imgs)/(1.0-imgs))

    def invers_tanh(self, imgs):
        return self.arctanh(imgs*2-1)

    def tanh(self, imgs_atanh):
        return 0.5*(torch.tanh(imgs_atanh+1))

    def forward(self, imgs):
        imgs = imgs
        # target_label = self.target_label.to(self.device)

        # w = Variable(torch.from_numpy(np.arctanh((imgs.numpy()*2-1) * 0.99999)).float())
        # np_img = torch.from_numpy(np.arctanh((imgs.numpy()*2-1) * 0.99999))
        w = self.invers_tanh(imgs)
        w.requires_grad = True

        if(not self.is_target):
            self.target_label = self.default_label

        # 二分查找c
        for binary_index in range(self.binary_number):
            print("Start {} search,current c is {}.".format(binary_index, self.c))


            # imgs_pert = torch.zeros_like(imgs_arctanh)
            # imgs_pert.requires_grad = True
            optimizer = optim.Adam([w], lr=self.lr)
            isSuccessfulAttack = False

            for step in range(self.steps):

                adv_imgs = self.tanh(w)
                self.model.eval()
                output = self.model(adv_imgs)

                Distence_loss = self.L2_norm(imgs, adv_imgs)
                F_loss = self.c * self.f_loss(output, self.target_label)
                cw_loss = Distence_loss + F_loss

                optimizer.zero_grad()
                cw_loss.backward()
                optimizer.step()

                if step % 100 == 0:
                    print("Step: [{}/{}]\tcw_loss:{} L2_norm:{} F_loss:{}".format(
                        step, self.steps, cw_loss.item(), Distence_loss.item(), F_loss.item()
                    ))

                adv_pre = output.argmax(1, keepdim=True).item()

                if(self.is_target) :
                    if(Distence_loss < self.min_loss and adv_pre == self.target_label):
                        flag = False
                        for i in range(20):
                            output = model(adv_imgs)
                            adv_pre = output.argmax(1,keepdim=True).item()
                            if(adv_pre != self.target_label):
                                flag = True
                                break
                        if(flag):continue
                        self.min_loss = Distence_loss
                        min_loss_img = adv_imgs
                        print("success when loss")
                        isSuccessfulAttack = True
                else:
                    if(Distence_loss < self.min_loss and adv_pre != self.target_label):
                        flag = False
                        for i in range(50):
                            output = model(adv_imgs)
                            adv_pre = output.argmax(1, keepdim=True).item()
                            if (adv_pre == self.target_label):
                                flag = True
                                break
                        if (flag): continue
                        self.min_loss = Distence_loss
                        min_loss_img = adv_imgs
                        print("success when loss")
                        isSuccessfulAttack = True
                if(isSuccessfulAttack):
                    self.maxc = min(self.maxc, self.c)
                    if(maxc<1e9):
                        self.c = (self.minc+self.maxc)/2
                else:
                    self.minc = max(self.minc, self.c)
                    if(self.maxc<1e9):
                        self.c = (self.maxc+self.minc)/2
                    else:
                        self.c = self.c*10
        return min_loss_img


if __name__ == "__main__":
    target_model = MNIST_Model()
    target_model.load_state_dict(torch.load("AlexNet_model.pth", map_location=torch.device('cpu')))
    target_model.eval()
    test_data = datasets.MNIST(root='./mnist_data/', train=False, download=False, transform=transforms.ToTensor())
    # test_loader = torch.utils.data.DataLoader(test_data, batch_size=1)
    target_model.eval()
    test_img, test_label = test_data[6666]
    img = transforms.ToPILImage()(test_img).convert('RGB')
    plt.imshow(img)
    plt.show()
    inputimg = Variable(torch.unsqueeze(test_img, 0))
    output = target_model(inputimg)
    print("The pred result is ", output.argmax(1, keepdim=True).item())
    victim_img = test_img
    victim_label = test_label
    print("Correct Label is ", victim_label)


    victim_img_input = torch.unsqueeze(victim_img, 0)

    print(victim_img_input.shape)
    attack = CW_attack(target_model)
    attack_img = attack.forward(victim_img_input)

    attack_imgs = Variable(torch.squeeze(attack_img, 0))
    imgs = transforms.ToPILImage()(attack_imgs).convert('RGB')
    plt.imshow(imgs)
    plt.show()
    inputimg = Variable(torch.unsqueeze(attack_imgs, 0))
    output = target_model(inputimg)
    print("The pred result is ", output.argmax(1, keepdim=True).item())
