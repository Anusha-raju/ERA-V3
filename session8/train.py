from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
from torchsummary import summary
from utils import FitEvaluate, device
from data import trainloader, testloader
from model import Net

model = Net().to(device)
summary(model, input_size=(3, 32, 32))


optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
scheduler = StepLR(optimizer, step_size=5, gamma=0.6)

model_fiteval = FitEvaluate(model, device,trainloader,testloader)
model_fiteval.epoch_training(optimizer, scheduler = scheduler,EPOCHS=20)