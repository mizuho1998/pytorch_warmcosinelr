import matplotlib.pyplot as plt
import torch
import torch.optim as optim

from WarmCosineAnnealingLR import WarmCosineAnnealingLR

model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18')
optimizer = optim.SGD(model.parameters(), lr=0.1)

epoch = 100
scheduler = WarmCosineAnnealingLR(optimizer, epoch, warmup=10, eta_min=1e-5)

l = []
for epoch in range(epoch):
    l.append(optimizer.param_groups[0]['lr'])
    optimizer.step()
    scheduler.step()

plt.plot(l)
plt.savefig('./image.png')
