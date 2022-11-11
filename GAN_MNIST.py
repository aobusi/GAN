import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.utils import save_image
import os
import torch.nn.functional as F

# Hyper Parameters
batch_size = 100
epochs = 300
latent_size = 100
hidden_size = 256
image_size = 784

RealImage = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,
    transform=torchvision.transforms.ToTensor(),  # 转换PIL.Image成Tensor
    download=True,
)

RealLoader = DataLoader(dataset=RealImage, batch_size=batch_size, shuffle=True)

# 判别器: 输入原始图片，输出判别的结果
class Discriminator(nn.Module):
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        return torch.sigmoid(self.fc4(x))


# 生成器: 根据给定的分布，来生成一张图片
class Generator(nn.Module):
    def __init__(self, g_input_dim, g_output_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(g_input_dim, 256)  # 100 -> 256
        self.fc2 = nn.Linear(256, 512)  # 256 -> 512
        self.fc3 = nn.Linear(512, 1024)  # 512 -> 1024
        self.fc4 = nn.Linear(1024, g_output_dim)  # 1024 -> 784

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))


G = Generator(g_input_dim=latent_size, g_output_dim=image_size)
D = Discriminator(image_size)
loss = nn.BCELoss()
optimizer1 = optim.Adam(D.parameters(), lr=0.0003)
optimizer2 = optim.Adam(G.parameters(), lr=0.0003)

for epoch in range(epochs):
    for step, (x, y) in enumerate(RealLoader):
        images = x.reshape(-1,image_size)  # 真图像
        real_labels = torch.ones(batch_size, 1).reshape(x.size(0)).type(torch.FloatTensor)
        fake_labels = torch.zeros(batch_size, 1).reshape(x.size(0)).type(torch.FloatTensor)

        # ================================================================== #
        #                      训练判别器                                      #
        # ================================================================== #

        # 定义判别器的损失函数
        outputs = D(images)
        real_loss = loss(outputs, real_labels)
        real_score = outputs

        # 定义判别器对假图像的损失函数
        fack_digit = torch.randn(batch_size, latent_size)
        fake_images = G(fack_digit)

        outputs = D(fake_images)
        fake_loss = loss(outputs, fake_labels)
        fake_score = outputs

        # 得到判别器的总损失
        total_loss = real_loss + fake_loss

        optimizer1.zero_grad()
        total_loss.backward()
        optimizer1.step()

        # ================================================================== #
        #                      训练生成器                                      #
        # ================================================================== #

        z = torch.randn(batch_size, latent_size)
        fake_images = G(z)
        outputs = D(fake_images)

        g_loss = loss(outputs, real_labels)

        optimizer2.zero_grad()
        g_loss.backward()
        optimizer2.step()

        if (step+1) % 200 == 0:
            print(
                'Epoch [{}/{}],  total_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}' .format(
                epoch, epochs, total_loss.item(),
                g_loss.item(), real_score.mean().item(), fake_score.mean().item()))

    # 保存真图像
    if (epoch + 1) == 1:
        images = images.reshape(images.size(0), 1, 28, 28)
        save_image(images, os.path.join('../img/mnist', 'real_images.png'))

    # 保存假图像
    fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
    save_image(fake_images, os.path.join('../img/mnist', 'fake_images-{}.png'.format(epoch+1)))

# 保存模型
torch.save(G.state_dict(), 'G.ckpt')
torch.save(D.state_dict(), 'D.ckpt')