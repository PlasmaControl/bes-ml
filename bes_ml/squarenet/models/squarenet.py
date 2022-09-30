""" Full assembly of the parts to form the complete network """

from .squarenet_utils import *


class SquareNet(nn.Module):
    def __init__(self, bilinear=False):
        super(SquareNet, self).__init__()
        self.bilinear = bilinear

        self.inc = DoubleConv(1, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=5, stride=2),
            nn.Conv2d(64, 32, kernel_size=3),
            nn.Conv2d(32, 16, kernel_size=3),
            nn.Conv2d(16, 8, kernel_size=3)
        )
        self.relu = nn.ReLU()
        self.fc = nn.Sequential(
            nn.Linear(in_features=1152, out_features=512),
            nn.Linear(in_features=512, out_features=256),
            nn.Linear(in_features=256, out_features=128)
        )

    def forward(self, x):

        x1, x2 = x[:, :, 0, :, :], x[:, :, 1, :, :]

        x1a, x2a = self.inc(x1), self.inc(x2)
        x1b, x2b = self.down1(x1a), self.down1(x2a)
        x1c, x2c = self.down2(x1b), self.down2(x2b)
        x1d, x2d = self.down3(x1c), self.down3(x2c)
        x1e, x2e = self.down4(x1d), self.down4(x2d)
        x1, x2 = self.up1(x1e, x2d), self.up1(x2e, x1d)
        x1, x2 = self.up2(x1, x2c), self.up2(x2, x1c)
        x1, x2 = self.up3(x1, x2b), self.up3(x2, x1b)
        x1, x2 = self.up4(x1, x2a), self.up4(x2, x1a)

        x = torch.cat([x2, x1], dim=1)

        x = self.outc(x)
        x = self.fc(torch.flatten(x, start_dim=1))

        return x

if __name__ == '__main__':
    x = torch.rand((1, 1, 2, 40, 40)) * 10

    x1, x2 = x[:, :, 0, :, :], x[:, :, 1, :, :]

    model = SquareNet()
    results = model(x1, x2)
    print(x)