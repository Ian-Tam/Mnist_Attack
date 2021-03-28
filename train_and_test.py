from data_loader import test_loader,train_loader
import torch
# import tqdm
import time
import torch.nn.functional as F
# from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
from Classfication_model import THNet,MNIST_Model
from torchvision import models

def get_time(start_time = None):
    if start_time == None:
        return time.time()
    else:
        return time.time() - start_time

def train_model(model):
    LOSS = []
    ACC = []

    for epoch in range(50):
        model.train()
        start_time = get_time()
        train_loss = 0.0
        eval_acc = 0
        Correct = 0
        # for img,label in tqdm(train_loader,desc=str("epoch:{}".format(epoch)),ncols=100):
        #     time.sleep(0.05)
        for img, label in train_loader:

            img,label = img.to(device), label.to(device)
            img.requires_grad_()

            output = model(img)
            loss = criterion(output,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print(img.grad.data)
            train_loss += loss.item()
            #output.max() 返回两个值（values,indices)
            _,pred = output.max(1)
            Correct = (pred == label).sum().item()
            acc = Correct / img.shape[0]
            eval_acc += acc
            # print("Epoch:{},step:{},step_loss:{:.4f},step_acc:{:.4f}".format(epoch,i,loss,acc))
        LOSS.append(train_loss / len(train_loader))
        ACC.append((eval_acc / len(train_loader)))
        print("\n Epoch:{}, Train_loss:{:.5f}, Train_ACC: {:.5f},Time_using: ".format(epoch, train_loss/len(train_loader), eval_acc/len(train_loader), get_time(start_time)))
    # torch.save(obj=model.state_dict(),f="model.pth")
    # print("Save model successfully!")
    torch.save(obj=model.state_dict(),f="AlexNet_model.pth")
    print("Save MnistNet_model successfully!")
    return LOSS, ACC

def test_model(model):
    eval_loss = 0.0
    eval_acc = 0.0
    model.eval()
    for i, (img, label) in enumerate(test_loader):
        img, label = img.to(device), label.to(device)
        output = model(img)
        test_loss = criterion(output, label)
        _, pred =output.max(1)
        correct = (pred == label).sum().item()
        test_acc = correct / img.shape[0]
        eval_loss += test_loss
        eval_acc += test_acc
    print("Test Loss: {:.4f},Test ACC:{:.4f}".format(eval_loss/len(test_loader),eval_acc/len(test_loader)))

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("This model running on: ", device)
    device = torch.device("cpu")
    # model = THNet().to(device)
    model = MNIST_Model().to(device)
    print(model)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    criterion = nn.CrossEntropyLoss()
    # print("Model training....")
    # train_model(model)
    print("Model testing...")
    model.load_state_dict(torch.load("AlexNet_model.pth"))
    model.eval()
    print("成功加载测试模型！")
    test_model(model)
