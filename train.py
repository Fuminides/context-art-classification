import torch
import torch.nn as nn
from model_grad_cam import LeNet
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
from parser import gen_parser
from blob_dataloader import BlobDataset

# Parse the arguments
parser = gen_parser()
args = parser.parse_args()

# Load the data
if args.dataset == 'mnist':
    train_dataset = datasets.MNIST(root='.', train=True, transform=transforms.ToTensor(), download=True)
elif args.dataset == 'cifar10':
    train_dataset = datasets.CIFAR10('data', train=True, download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))
                         ]))
elif args.dataset == 'blob':
    train_dataset = BlobDataset('../trials/', train=True, transform=transforms.ToTensor())

    

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

# Define the model
model = LeNet([args.image_size, args.image_size, 3], [4, 2])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# Train the model
def train(model, criterion, optimizer, train_loader, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target_list = []
        for target_x in target:
            target_x = target_x.to(device)
            target_list.append(target_x)

        
        optimizer.zero_grad()
        outputs = model(data)
        total_loss = None
        for i in range(len(target_list)):
            if total_loss is None:
                total_loss = criterion(outputs[i], target_list[i]) / len(target_list)
            else:
                total_loss += criterion(outputs[i], target_list[i]) / len(target_list)

        total_loss.backward()
        optimizer.step()


epochs = args.epochs
for epoch in range(1, epochs + 1):
    print('Epoch ', epoch)
    train(model, criterion, optimizer, train_loader, epoch)

# Save the model
torch.save(model.state_dict(), args.model_destination + 'model.pt')