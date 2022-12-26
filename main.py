"""Landscape categorization DNN"""
import os
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE is", DEVICE)

def load_all(directory, label, skip_after = None):
    """Create a list of all sample files for delayed loading"""
    samples = []
    print("Reading", directory)

    for file in os.listdir(os.fsencode(directory)):
        filename = os.fsdecode(file)
        if filename.endswith(".jpeg"):
            label = torch.Tensor([label]).view(1).type(torch.long).to(DEVICE)
            samples.append((directory+"\\"+filename, label))

            if skip_after is not None:
                skip_after -= 1
                if skip_after <= 0:
                    break
        else:
            continue

    return samples

class ImageDataset(Dataset):
    """Dataset including data augmentation pipeline"""
    def __init__(self, samples, augment):
        self.samples = samples
        if augment:
            self.transform = torchvision.transforms.Compose([
              torchvision.transforms.RandomRotation(
                degrees=(-10,10),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
              torchvision.transforms.CenterCrop((282,282)),
              torchvision.transforms.RandomCrop((256,256)),
              torchvision.transforms.RandomHorizontalFlip(),
              torchvision.transforms.ColorJitter(brightness=0.05, hue=0.05),
            ])
        else:
            self.transform = torchvision.transforms.Compose([
              torchvision.transforms.CenterCrop((256,256)),
            ])
            self.resize = torchvision.transforms.Resize(
              (312,312),
              interpolation=torchvision.transforms.InterpolationMode.BILINEAR)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = self.resize(torchvision.io.read_image(self.samples[idx][0]).to(DEVICE))
        img = img.type(torch.float) / 255

        return self.transform(img), self.samples[idx][1]

class Network(torch.nn.Module):
    """The DNN for categorization"""
    def __init__(self):
        super(Network, self).__init__()

        self.relu = torch.nn.GELU()

        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=(5,5), padding="same").to(DEVICE)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)).to(DEVICE) # 8x128x128
        self.bn1 = torch.nn.BatchNorm2d(num_features=16)

        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=(5,5), padding="same").to(DEVICE)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)).to(DEVICE) # 16x64x64
        self.bn2 = torch.nn.BatchNorm2d(num_features=32)

        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=(5,5), padding="same").to(DEVICE)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)).to(DEVICE) # 32x32x32
        self.bn3 = torch.nn.BatchNorm2d(num_features=64)

        self.conv4 = torch.nn.Conv2d(64, 128, kernel_size=(5,5), padding="same").to(DEVICE)
        self.pool4 = torch.nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)).to(DEVICE) # 64x16x16
        self.bn4 = torch.nn.BatchNorm2d(num_features=128)

        self.conv5 = torch.nn.Conv2d(128, 256, kernel_size=(5,5), padding="same").to(DEVICE)
        self.pool5 = torch.nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)).to(DEVICE) # 128x8x8
        self.bn5 = torch.nn.BatchNorm2d(num_features=256)

        self.conv6 = torch.nn.Conv2d(256, 256, kernel_size=(5,5), padding="same").to(DEVICE)
        self.pool6 = torch.nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)).to(DEVICE) # 256x4x4
        self.bn6 = torch.nn.BatchNorm2d(num_features=256)

        self.linear1 = torch.nn.Linear(4096, 2048).to(DEVICE)
        self.dropout1 = torch.nn.Dropout(p=0.5)
        self.linear2 = torch.nn.Linear(2048, 1024).to(DEVICE)
        self.dropout2 = torch.nn.Dropout(p=0.5)
        self.linear3 = torch.nn.Linear(1024, 512).to(DEVICE)
        self.dropout3 = torch.nn.Dropout(p=0.5)
        self.linear4 = torch.nn.Linear(512, 5).to(DEVICE)

        self.flatten = torch.nn.Flatten()

    def forward(self, x):
        """Forward pass"""
        x = self.relu(self.bn1(self.pool1(self.conv1(x))))
        x = self.relu(self.bn2(self.pool2(self.conv2(x))))
        x = self.relu(self.bn3(self.pool3(self.conv3(x))))
        x = self.relu(self.bn4(self.pool4(self.conv4(x))))
        x = self.relu(self.bn5(self.pool5(self.conv5(x))))
        x = self.relu(self.bn6(self.pool6(self.conv6(x))))
        x = self.flatten(x)

        x = self.dropout1(self.relu(self.linear1(x)))
        x = self.dropout2(self.relu(self.linear2(x)))
        x = self.dropout3(self.relu(self.linear3(x)))
        x = self.linear4(x)
        x = F.log_softmax(x, dim=1)
        return x

trainset = []
valset = []

skip_after = None
trainset += load_all("Training Data\\Coast", 0, skip_after)
trainset += load_all("Training Data\\Desert", 1, skip_after)
trainset += load_all("Training Data\\Forest", 2, skip_after)
trainset += load_all("Training Data\\Glacier", 3, skip_after)
trainset += load_all("Training Data\\Mountain", 4, skip_after)

valset += load_all("Validation Data\\Coast", 0, skip_after)
valset += load_all("Validation Data\\Desert", 1, skip_after)
valset += load_all("Validation Data\\Forest", 2, skip_after)
valset += load_all("Validation Data\\Glacier", 3, skip_after)
valset += load_all("Validation Data\\Mountain", 4, skip_after)

BATCH_SIZE = 32
trainset = ImageDataset(trainset, True)
valset = ImageDataset(valset, False)
train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=True)

model = Network()
model.to(DEVICE)

s = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Model uses", s, "parameters")

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.975)
loss_function = torch.nn.NLLLoss()

def epoch(data, training):
    """Training/Validation of a single epoch"""
    if training:
        model.train()
        name = "Train:"
    else:
        model.eval()
        name = "Valid:"

    total_loss = 0
    cnt = 0
    correct = 0
    for batch, label in data:
        if training:
            optimizer.zero_grad()

        x = model(batch)
        loss = loss_function(x, label.view(-1))
        total_loss += loss.item()

        if training:
            loss.backward()
            optimizer.step()

        correct += torch.sum((torch.argmax(x, dim=1) == label.view(-1)).float())

        cnt = cnt + 1
        if cnt % 10 == 0:
            print(name,
              "     loss {:.5f}".format(total_loss / cnt),
              "     accuracy ", "{:.2f}%".format(correct / (cnt * BATCH_SIZE) * 100))

    print(name,
      "     loss {:.5f}".format(total_loss / cnt),
      "     accuracy ", "{:.2f}%".format(correct / (cnt * BATCH_SIZE) * 100))
    return correct / (cnt * BATCH_SIZE)

PATH="model.pt"
best_accuracy = 0
for ep in range(50):
    print("Epoch", ep)
    epoch(train_loader, True)
    accuracy = epoch(val_loader, False)
    scheduler.step()
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        print("This is the best model so far, saving it")
        torch.save({
          'epoch': ep,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          }, PATH)
