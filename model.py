import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import cv2

import utils


class Model(nn.Module):

    def __init__(self, hidden_size=512, fn_weight=1):
        super(Model, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features=4, out_features=hidden_size),
            nn.Tanh(),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.Tanh(),
        )
        self.output_layer = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=4),
            nn.Sigmoid(),
        )
        self.fn_weight = fn_weight
        self.to(self.device)
        self._x = None
        self._image = None

    def forward(self):
        x = torch.zeros((1, 4), dtype=torch.float32).fill_(0.5).to(self.device)
        x = self.input_layer(x)
        x = self.fc1(x)
        x = self.output_layer(x)
        x = x.view(-1)
        self._x = x
        return x

    def fit(self, image, epochs=10, steps=500, lr0=1e-6, lrf=1e-2, fn_weight=None, visualize=True):
        self.fn_weight = self.fn_weight if fn_weight is None else fn_weight

        # Store image as numpy array and make a torch tensor version
        self._image = image
        image = torch.tensor(image, dtype=torch.float32, device=self.device)

        # Setup optimizer and lr scheduler
        optimizer = torch.optim.RMSprop(self.parameters(), lr=lr0)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda i: (((1 + np.cos(i * np.pi / (epochs - 1))) / 2) ** 1) * (1 - lrf) + lrf,
        )

        self.train()
        for epoch in range(epochs):
            print("Epoch: %s" % str(epoch).ljust(6))
            for step in range(steps):
                self.forward()
                loss, metrics = self.criterion(image)
                loss.backward()
                optimizer.step()
                print("\rloss: %.4e, fn: %.4e, fp: %.4e" % (loss.detach(), metrics["fn"], metrics["fp"]), end="\r")
                if visualize and step % 20 == 0:
                    self.visualize()
            lr_scheduler.step()
            print()

        # Print results
        x = self._x.detach().cpu().numpy() * np.tile(self._image.shape[0:2][::-1], 2)
        print("\nEllipse parameters: x=%.3e, y=%.3e, w=%.3e, h=%.3e" % tuple(x))

    def criterion(self, image):
        e = self.formulate_ellipse()
        fn_loss = torch.mean(image * self.sigmoid(e, c=20))
        fp_loss = torch.mean((1 - image) * F.relu(1 - e))
        e = e.detach()
        fp = len(e[torch.bitwise_and(image == 0, e < 1)])
        fn = len(e[torch.bitwise_and(image == 1, e > 1)])
        return fp_loss + fn_loss * self.fn_weight, {"fn": fn, "fp": fp}

    def formulate_ellipse(self):
        h, w = self._image.shape
        my, mx = torch.meshgrid(torch.arange(h), torch.arange(w))
        my = my.to(device=self.device, dtype=torch.float32)
        mx = mx.to(device=self.device, dtype=torch.float32)
        return (mx - self._x[0] * w) ** 2 / (self._x[2] * w) ** 2 + (my - self._x[1] * h) ** 2 / (self._x[3] * h) ** 2

    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def sigmoid(x, c=1):
        return 1 / (1 + torch.exp(-c * x + c))

    def visualize(
            self,
            window_name="Visualisation",
            fn_col=(38, 38, 210),
            tp_col=(38, 210, 38),
            fp_col=(38, 110, 38)
    ):
        x = self._x.detach().cpu().numpy()  # Detach params
        x *= np.tile(np.array(self._image.shape), 2)  # Scale params by image shape
        x, y, a, b = x.astype(int)  # Unpack ellipse parameters

        # Create the image
        ellipse = cv2.ellipse(np.zeros_like(self._image), (x, y), (a, b), 0, 0, 360, 1, -1)
        image = cv2.cvtColor(self._image.copy().astype(np.uint8), cv2.COLOR_GRAY2BGR)  # Convert to RGBA
        image[np.all(image == 1, axis=2)] = fn_col  # Colour the blob
        image[ellipse * self._image == 1] = tp_col  # Colour the area of blob which is inside the ellipse
        image[ellipse * (1 - self._image) == 1] = fp_col  # Colour area which is outside blob & inside ellipse

        # Display the image
        cv2.imshow(window_name, image)
        cv2.waitKey(1)
        return image


if __name__ == "__main__":
    im = utils.load("examples/01.png", pad=100, invert=True)

    model = Model()
    model.fit(im, epochs=10, steps=1000, fn_weight=15, lrf=1e-1)
