import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
from Classfication_model import MNIST_Model


def L2_norm(img, adv_img):
    return torch.dist(adv_img, 0.5*(torch.tanh(img)+1), p=2)

def f_loss(output, target_label, target, k=0):
    # f函数选用作者觉得最好的f6(x)=max((max(Z(x')i)-Z(x')t),-k)
    tlab = Variable(torch.from_numpy(np.eye(10)[target_label]).float()).to(device)
    real = torch.max(output * tlab)
    second = torch.max((1 - tlab) * output)
    # 如果指定了对象，则让这个更接近，否则选择第二个较大的
    if (target):
        return torch.max(second - real, torch.Tensor([-k]).to(device))
    else:
        return torch.max(real - second, torch.Tensor([-k]).to(device))

def CW_attack_L2(img, model, right_label, iteration=1000, lr=1e-3, target=False, target_label=0):
    binary_number = 10           # 二分查找最大次数
    maxc = 1e10                 # 从0.01-100去找c
    minc = 0
    c = 1e-3                    # from c = 0:01 to c = 100 on the MNIST dataset.
    min_loss = 1000000          # 找到最小的loss，即为方程的解
    min_loss_img = img          # 扰动后的图片
    k = 0                       # f函数使用，论文默认为0
    b_min = 0                   # 盒约束，论文中使用了0-1 代码中-0.5 0.5 好像也没用上
    b_max = 1
    if (not target):
        target_label = right_label


    for binary_index in range(binary_number):
        print("------------Start {} search, current c is {}------------".format(binary_index, c))

        # 将img转换为w，w=arctanh(2x-1)，作为原始图片
        w = Variable(torch.from_numpy(np.arctanh((img.numpy() - 0.5) / 0.5 * 0.99999)).float()).to(device)

        w_pert = Variable(torch.zeros_like(w).float()).to(device)
        w_pert.requires_grad = True
        # 最初图像x
        x = Variable(img).to(device)
        optimizer = optim.Adam([w_pert], lr=lr)
        isSuccessfulAttack = False

        for iteration_index in range(1, iteration + 1):
            optimizer.zero_grad()

            # w加入扰动w_pert之后的新图像
            adv_img = 0.5*(torch.tanh(w+w_pert)+1)
            output = model(adv_img)                   # Z(x)
            Distance_loss = L2_norm(w, adv_img)       # ||x,x'||_2
            F_loss= f_loss(output, target_label, target, k)
            CW_loss = Distance_loss + c * F_loss
            CW_loss.backward()
            optimizer.step()

            if iteration_index % 200 == 0:
                print('Iters: [{}/{}]  CW_Loss: {}, L2_distance:{}, F_func_Loss:{}'
                      .format(iteration_index, iteration, CW_loss.item(), Distance_loss.item(), F_loss.item()))

            pred_result = output.argmax(1, keepdim=True).item()
            # 指定目标模式下,此处考虑l2距离最小,即找到最小的loss1
            if (target):
                if (min_loss > Distance_loss and pred_result == target_label):
                    flag = False
                    for i in range(20):
                        output = model(adv_img)
                        pred_result = output.argmax(1, keepdim=True).item()
                        if (pred_result != target_label):
                            flag = True  # 原模型中存在dropout，此处判断连续成功攻击20次，则视为有效
                            break
                    if (flag):
                        continue
                    min_loss = Distance_loss
                    min_loss_img = adv_img
                    print('success when loss: {}, pred: {}'.format(min_loss, pred_result))
                    isSuccessfulAttack = True
            # 非目标模式，找到最接近的一个,连续20次不预测成功
            else:
                if (min_loss > Distance_loss and pred_result != right_label):
                    flag = False
                    for i in range(50):
                        output = model(adv_img)
                        pred_result = output.argmax(1, keepdim=True).item()
                        if (pred_result == right_label):
                            flag = True  # 原模型中存在dropout，此处判断连续成功攻击50次，则视为有效
                            break
                    if (flag):
                        continue
                    min_loss = Distance_loss
                    min_loss_img = adv_img
                    print('success when loss: {}, pred: {}'.format(min_loss, pred_result))
                    isSuccessfulAttack = True
        if (isSuccessfulAttack):
            maxc = min(maxc, c)
            if maxc < 1e9:
                c = (minc + maxc) / 2
        # 攻击失败，尝试放大c
        else:
            minc = max(minc, c)
            if (maxc < 1e9):
                c = (maxc + minc) / 2
            else:
                c = c * 10
    return min_loss_img

if __name__=='__main__':
    target_model = MNIST_Model()
    target_model.load_state_dict(torch.load("AlexNet_model.pth", map_location=torch.device('cpu')))
    target_model.eval()
    test_data = datasets.MNIST(root='./mnist_data/', train=False, download=False, transform=transforms.ToTensor())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_model.eval().to(device)

    # 打印原始模型预测结果
    test_img, test_label = test_data[1234]
    img = transforms.ToPILImage()(test_img).convert('RGB')
    plt.imshow(img)
    plt.show()
    inputimg = Variable(torch.unsqueeze(test_img, 0)).to(device)
    output = target_model(inputimg)
    print("The pred result is ", output.argmax(1, keepdim=True).item())

    # 打印并输出攻击后的图片预测结果
    victim_img=test_img
    victim_label=test_label
    print("Correct Label is ", victim_label)
    victim_img_input = torch.unsqueeze(victim_img, 0).to(device)
    attack_img = CW_attack_L2(victim_img_input, target_model, victim_label, iteration=1000, lr=0.01, target=True, target_label=5)
    attack_imgs = Variable(torch.squeeze(attack_img, 0))
    imgs = transforms.ToPILImage()(attack_imgs).convert('RGB')
    plt.imshow(imgs)
    plt.show()
    adv_inputimg=Variable(torch.unsqueeze(attack_imgs, 0)).to(device)
    output = target_model(adv_inputimg)
    print("The pred result is ", output.argmax(1, keepdim=True).item())

    # 打印对抗样本
    pert = attack_imgs - test_img
    perts = pert / abs(pert).max()/2.0+0.5
    perts = transforms.ToPILImage()(perts).convert('RGB')
    plt.imshow(perts, cmap=plt.gray())
    plt.show()
