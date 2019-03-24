import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

EPOCH = 2
BATCH_SIZE_ini = 50
LR = 0.001

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1,out_channels=16,kernel_size=5,stride=1,padding=2),nn.ReLU(),nn.MaxPool2d(kernel_size=2))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 5, 1, 2),nn.ReLU(),nn.MaxPool2d(2))
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output

batch_list=[]
correct_list=[]
loss_list=[]
sen_list=[]
tloss_list=[]
tcorrect_list=[]

# models
for b in range(0,1):
    BATCH_SIZE = BATCH_SIZE_ini*(2**b)
    train_data = torchvision.datasets.MNIST(root='./mnist/', train=True, transform=torchvision.transforms.ToTensor()) # download=True
    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_data = torchvision.datasets.MNIST(root='./mnist/', train=False, transform=torchvision.transforms.ToTensor()) # download=True
    test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)
    model = CNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()
    
    print('Model {} Start Training.'.format(b))

    #each epoch
    for i in range(1,EPOCH+1):
        sen=0.0
        for step, (bx, by) in enumerate(train_loader):
            print('batch start')
            b_x = Variable(bx, requires_grad=True)  # batch x
            b_y = Variable(by)  # batch y

            output = model(b_x)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward(retain_graph=False)
            optimizer.step()
            grad_per_batch=torch.norm(b_x.grad.data.view(len(b_x),-1),2,1).mean()
            sen = sen + grad_per_batch.numpy()


        if i ==EPOCH:
            print('Testing set eval start')
            tcorrect=0.0
            tloss=0.0
            Loss=0.0
            correct=0.0

            for step, (tx,ty) in enumerate(test_loader):
                tx = Variable(tx, requires_grad=True)  # batch x
                ty = Variable(ty)  # batch y
                toutput=model(tx)
                loss = loss_func(toutput, ty)
                tloss+=loss.item()
                pred = toutput.data.max(1, keepdim=True)[1]
                tcorrect += pred.eq(ty.data.view_as(pred)).cpu().sum()
                #print('Tloss, Tcor:',tloss, tcorrect)

            print('Training set eval start')
            for step, (bx,by) in enumerate(train_loader):
                bx = Variable(bx, requires_grad=True)  # batch x
                by = Variable(by)  # batch y
                output=model(bx)
                loss = loss_func(output, by)
                Loss+=loss.item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(by.data.view_as(pred)).cpu().sum()
                #print('loss, cor:',Loss, correct)
            
            tcorrect=tcorrect.numpy()
            print("Tloss,Tc:", tloss*BATCH_SIZE/len(test_loader.dataset), tcorrect/len(test_loader.dataset) )
            tcorrect_list.append(tcorrect/len(test_loader.dataset))
            tloss_list.append(tloss*BATCH_SIZE/len(test_loader.dataset))

            correct=correct.numpy()
            print("sen,loss,acc:", sen, Loss*BATCH_SIZE/len(train_loader.dataset), correct/len(train_loader.dataset))   
            correct_list.append(correct/len(train_loader.dataset))
            loss_list.append(Loss*BATCH_SIZE/len(train_loader.dataset))
            sen_list.append(sen)
            batch_list.append(BATCH_SIZE)


tloss_list=np.array(tloss_list)
np.save('tloss_MNIST-s.npy', tloss_list)
tcorrect_list=np.array(tcorrect_list)
np.save('tcorrect_MNIST-s.npy', tcorrect_list)

loss_list=np.array(loss_list)
np.save('loss_MNIST-s.npy', loss_list)
sen_list=np.array(sen_list)
np.save('sen_MNIST-s.npy',sen_list)
correct_list=np.array(correct_list)
np.save('acc_MNIST-s.npy', correct_list)
batch_list=np.array(batch_list)
np.save('batch_MNIST-s.npy', batch_list)