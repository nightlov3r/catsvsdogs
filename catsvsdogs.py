import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

REBUILD_DATA=True
if torch.cuda.is_available():
    device=torch.device("cuda:0")
    #print("Running on the GPU")
else:
    device=torch.device("cpu")
    #print("Running on the CPU")


class DogsVSCats():
    IMG_SIZE=50
    CATS="PetImages/Cat"
    DOGS="PetImages/Dog"
    LABELS={CATS: 0,DOGS: 1}

    training_data=[]
    catcount=0
    dogcount=0

    def make_training_data(self):
        for label in self.LABELS:
            print(label)
            for f in tqdm(os.listdir(label)):
                try:
                    path=os.path.join(label,f)
                    img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
                    img=cv2.resize(img,(self.IMG_SIZE,self.IMG_SIZE))
                    self.training_data.append([np.array(img),np.eye(2)[self.LABELS[label]]])

                    if label==self.CATS:
                        self.catcount+=1
                    else:
                        self.dogcount+=1
                except Exception as e:
                    pass #image isn't good
        np.random.shuffle(self.training_data)
        np.save("training_data.npy",self.training_data)
        print("Cats: ",self.catcount)
        print("Dogs: ",self.dogcount)
#if REBUILD_DATA:
    #dogsvcats=DogsVSCats()
    #dogsvcats.make_training_data()
training_data=np.load("training_data.npy",allow_pickle=True)
#print(len(training_data))
#plt.imshow(training_data[1][0],cmap="gray")
#plt.show()



class Net(nn.Module):
    def __init__(self):
        super().__init__() # just run the init of parent class (nn.Module)
        self.conv1 = nn.Conv2d(1, 32, 5) # input is 1 image, 32 output channels, 5x5 kernel / window
        self.conv2 = nn.Conv2d(32, 64, 5) # input is 32, bc the first layer output 32. Then we say the output will be 64 channels, 5x5 kernel / window
        self.conv3 = nn.Conv2d(64, 128, 5)

        x = torch.randn(50,50).view(-1,1,50,50)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512) #flattening.
        self.fc2 = nn.Linear(512, 2) # 512 in, 2 out bc we're doing 2 classes (dog vs cat).

    def convs(self, x):
        # max pooling over 2x2
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = torch.flatten(x, 1, -1)

        if self._to_linear is None:
            self._to_linear = x.shape[1]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # bc this is our output layer. No activation here.
        return F.softmax(x, dim=1)


X=torch.Tensor([i[0] for i in training_data]).view(-1,50,50)
X=X/255.0
y=torch.Tensor([i[1] for i in training_data])

VAL_PCT=0.1
val_size=int(len(X)*VAL_PCT)
#print(val_size)

train_X=X[:-val_size]
train_y=y[:-val_size]

test_X=X[-val_size:]
test_y=y[-val_size:]

BATCH_SIZE = 100
#EPOCHS = 15


net=Net().to(device)


def train(net,EPOCHS):
    optimizer=optim.Adam(net.parameters(),lr=0.001)
    loss_function=nn.MSELoss()
    accuracy=0
    count=0
    """
    while accuracy<.9:
        for i in tqdm(range(0, len(train_X), BATCH_SIZE)): # from 0, to the len of x, stepping BATCH_SIZE at a time. [:50] ..for now just to dev
            #print(f"{i}:{i+BATCH_SIZE}")
            batch_X = train_X[i:i+BATCH_SIZE].view(-1, 1, 50, 50)
            batch_y = train_y[i:i+BATCH_SIZE]

            batch_X,batch_y=batch_X.to(device),batch_y.to(device)

            net.zero_grad()

            outputs = net(batch_X)
            loss = loss_function(outputs,batch_y)
            loss.backward()
            optimizer.step()    # Does the update
        accuracy=test(net)
        count+=1
        print(f"Epoch: {count}. Loss: {loss}. Accuracy: {accuracy}")
        """
    for epoch in range(EPOCHS):
        for i in tqdm(range(0, len(train_X), BATCH_SIZE)): # from 0, to the len of x, stepping BATCH_SIZE at a time. [:50] ..for now just to dev
            #print(f"{i}:{i+BATCH_SIZE}")
            batch_X = train_X[i:i+BATCH_SIZE].view(-1, 1, 50, 50)
            batch_y = train_y[i:i+BATCH_SIZE]

            batch_X,batch_y=batch_X.to(device),batch_y.to(device)

            net.zero_grad()

            outputs = net(batch_X)
            loss = loss_function(outputs,batch_y)
            loss.backward()
            optimizer.step()    # Does the update
        accuracy=test(net)
        #plt.plot(epoch,accuracy)
        #plt.pause(0.05)
        print(f"Epoch: {epoch}. Loss: {loss}. Accuracy: {accuracy*100}%")
    #plt.show()             plotting doesn't work for some reason? it just stops the code from running


def test(net):
    correct = 0
    total = 0

    with torch.no_grad():
        for i in range(len(test_X)):
            real_class = torch.argmax(test_y[i]).to(device)
            net_out = net(test_X[i].view(-1, 1, 50, 50).to(device))[0]  # returns a list, 
            predicted_class = torch.argmax(net_out)

            if predicted_class == real_class:
                correct += 1
            total += 1
    return(round(correct/total, 3))


#train(net)
def saveModel(net,path):
    torch.save(net.state_dict(),path)

def loadModel(path):
    net=Net().to(device)
    torch.load(path)
    net.load_state_dict(torch.load(path))
    print(f"Model {path} has {test(net)*100}% accuracy.")
    return net

net=loadModel('30epochs.pt')
train(net,15)
saveModel(net,'45epochs.pt')

def fwd_pass(X,y,train=False):
    if train:
        net.zero_grad()
    outputs=net(X)
    matches=[torch.argmax(i)==torch.argmax(j)for i,j in zip(outputs,y)]
    acc=matches.count(True)/len(matches)
    loss=loss_function(outputs,y)

    if train:
        loss.backward()
        optimizer.step()
    return acc,loss

def test2(size=32):

    random_start=np.random.randint(len(test_X)-size)
    X,y=test_X[random_start:random_start+size],test_y[random_start:random_start+size]
    with torch.no_grad():
        val_acc,val_loss=fwd_pass(X.view(-1,1,50,50).to(device),y.to(device))
    return val_acc,val_loss
#val_acc,val_loss=test2(size=32)
#print(val_acc,val_loss)