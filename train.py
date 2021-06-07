import torch
from numpy import mod
from Generators.CellGenerator import Fake_cell_generator
from Generators.ModuleGenerator import Module_generator
from Generators.DistortedImageGenerator import DistortedImageGenerator

import cv2

if __name__ == "__main__":
    device = torch.device("cuda")

    cell_generator = Fake_cell_generator()
    module_generator = Module_generator(cell_generator)
    distorted_img_generator = DistortedImageGenerator(module_generator)

    model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=False)

    net = torch.nn.Sequential(
            torch.nn.Linear(512, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 11),
        )

    model.fc = net
    model.to(device)
    print(model)
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.1)

    for epoch in range(200):
        running_loss = 0.0
        for i, data in enumerate(distorted_img_generator):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data[0].to(device)
            labels = data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs.float())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        print('[%d] loss: %.3f' %(epoch + 1, running_loss))
        if epoch % 10 == 9:
            torch.save(model.state_dict(), 'c')

    print('Finished Training')