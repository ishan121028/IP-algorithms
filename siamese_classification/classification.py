import torch
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
from custom_dataset import SiameseDataset
from torch.utils.data import DataLoader
import sys
# sys.stdout = open("output.txt", 'w')
# sys.stderr = open("err.txt", "w")

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),

            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3)
        )
        # Defining the fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(4608, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),

            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),

            nn.Linear(128, 2))

    def forward_once(self, x):
        # Forward pass
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        # forward pass of input 1
        output1 = self.forward_once(input1)
        # forward pass of input 2
        output2 = self.forward_once(input2)
        return output1, output2


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on:
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x0, x1, y):
        # euclidian distance
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)

        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss

def evaluate(model, testloader):
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            img0, img1, label = data
            img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
            optimizer.zero_grad()
            output1, output2 = net(img0, img1)
            diff = output1 - output2
            dist_sq = torch.sum(torch.pow(diff, 2), 1)
            print(label)
            print(dist_sq)

if __name__ == "__main__":
    TRAIN_CSV = "./datasets/dataset.csv"
    TRAIN_DIR1 = "./datasets/OLI/"
    TRAIN_DIR2 = "./datasets/OLIVINE/OLIVINE/"
    config = {
        "epochs": 20,
        "lr": 1e-3,
        "weight_decay": 0.0005,
        "batch_size": 32
    }
    siamese_dataset = SiameseDataset(TRAIN_CSV, TRAIN_DIR1, TRAIN_DIR2,
                                     transform=transforms.Compose([transforms.Resize((128, 128)),
                                                                   transforms.ToTensor()
                                                                   ])
                                     )
    train_set, test_set = torch.utils.data.random_split(siamese_dataset,
                                                        [int(siamese_dataset.__len__() * 0.8), siamese_dataset.__len__() -
                                                         int(siamese_dataset.__len__() * 0.8)])

    TrainLoader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
    TestLoader = DataLoader(test_set, batch_size=config["batch_size"], shuffle=True)

    # Declare Siamese Network
    net = SiameseNetwork().cuda()
    # Decalre Loss Function
    criterion = ContrastiveLoss()
    # Declare Optimizer

    optimizer = torch.optim.Adam(net.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    # train the model
    def train():
        loss = []
        for epoch in range(config["epochs"]):
            for i, data in enumerate(TrainLoader, 0):
                img0, img1, label = data
                img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
                optimizer.zero_grad()
                output1, output2 = net(img0, img1)
                loss_contrastive = criterion(output1, output2, label)
                loss_contrastive.backward()
                optimizer.step()
            print("Epoch {}\n Current loss {}\n".format(epoch, loss_contrastive.item()))
            loss.append(loss_contrastive.item())

        return net, loss


    # set the device to cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, loss = train()
    plt.plot(loss, linestyle='--', marker='o', color='b')
    evaluate(model=model, testloader=TestLoader)
    torch.save(model.state_dict(), "model.pt")
    print("Model Saved Successfully")
    plt.xlabel("Epochs")
    plt.ylabel("Contrastive Loss")
    plt.show()
