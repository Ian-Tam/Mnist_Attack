import torch
from data_loader import test_loader,train_loader
import tqdm
import time
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
from Classfication_model import THNet,MNIST_Model

def FGSM_attack(img, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    attack_img = img + epsilon * sign_data_grad
    # attack_img = torch.clamp(attack_img,0,1)
    # attack_img = img - epsilon * sign_data_grad
    return attack_img

def Attack_test(model, test_loader, epsilon):
    attack_eval_acc = 0.0
    eval_acc = 0.0
    Correct = 0
    adv_examples = []
    for img, label in test_loader:
        img.requires_grad_()
        output = model(img)

        # 查看在干净样本上的测试正确率
        pred = output.max(1)[1]
        correct = (pred == label).sum().item()
        test_acc = correct / img.shape[0]
        eval_acc += test_acc

        loss = criterion(output,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # attacking
        img_grad_data = img.grad.data
        attack_img = FGSM_attack(img, epsilon, img_grad_data)

        attack_output = model(attack_img)
        attack_pred = attack_output.max(1)[1]
        # print("pred:",attack_pred)
        # print("label:",label)
        attack_correct = (attack_pred == label).sum().item()
        attack_acc = attack_correct / img.shape[0]
        attack_eval_acc += attack_acc
        # print(attack_pred.data)
        for i in attack_pred:
            if attack_pred[i].item() == label[i].item():
                Correct += 1
                # Special case for saving 0 epsilon examples
                if (epsilon == 0) and (len(adv_examples) < 5):
                    adv_ex = attack_img[i].squeeze().detach().cpu().numpy()
                    adv_examples.append((pred[i].item(), attack_pred[i].item(), adv_ex))
            else:
                # Save some adv examples for visualization later
                if len(adv_examples) < 5:
                    adv_ex = attack_img[i].squeeze().detach().cpu().numpy()
                    adv_examples.append((pred[i].item(), attack_pred[i].item(), adv_ex))
    Clearn_acc = eval_acc / len(test_loader)
    Attack_acc = attack_eval_acc / len(test_loader)
    # print("/nTest eval acc:",eval_acc / len(test_loader))
    # print("Test Attack eval acc: ",attack_eval_acc / len(test_loader))
    return Clearn_acc, Attack_acc , adv_examples

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("This model running on: ", device)
    # model = THNet().to(device)
    model = MNIST_Model().to(device)
    model.load_state_dict(torch.load("AlexNet_model.pth"))
    print(model)
    epsilons = [0, .05, .1, .15, .2, .25, .3]
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    criterion = nn.CrossEntropyLoss()

    examples = []
    for epsilon in epsilons:
        Clearn_acc, Attack_acc, adv_examples = Attack_test(model,test_loader,epsilon)
        print("epsilon = {} , clearn_ACC = {} , attack_ACC = {} ".format(epsilon,Clearn_acc,Attack_acc))
        examples.append(adv_examples)
    print("All is done!")

    cnt = 0
    plt.figure(figsize=(8, 10))
    for i in range(len(epsilons)):
        for j in range(len(examples[i])):
            cnt += 1
            plt.subplot(len(epsilons), len(examples[0]), cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
            orig, adv, ex = examples[i][j]
            plt.title("{} -> {}".format(orig, adv))
            plt.imshow(ex, cmap="gray")
    plt.tight_layout()
    plt.show()