import contextlib
import cv2
from itertools import count
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class PlateauMonitor(object):

    def __init__(self, threshold=1e-2, period=100):
        self.period = period
        self.threshold = threshold
        self.values = np.array([], dtype=np.float32).reshape(0)
        self.average_change = None
        self.plateaued = False

    def step(self, x):
        self.values = self.values if len(self.values) < self.period else self.values[1:]
        self.values = np.append(self.values, np.array(x, dtype=np.float32))
        v0 = self.values[:-1]
        v1 = self.values[1:]
        if len(self.values) == self.period:
            self.average_change = np.abs(np.average((v1 - v0) / v0)) * 100
            self.plateaued = self.average_change < self.threshold

    def reset(self):
        self.values = np.array([], dtype=np.float32)
        self.plateaued = False


class Model(nn.Module):

    def __init__(self, image=None, steps=100,
                 fn_max=5, threshold_1=1e-2, threshold_2=1e-4, period_1=10, period_2=100,
                 lr_steps=25, verbose_steps=10, vis_steps=1, video_filename=None
                 ):
        super(Model, self).__init__()
        self.plateau_monitor_1 = PlateauMonitor()
        self.plateau_monitor_2 = PlateauMonitor()

        # Public variables

        self.image = image
        self.fn_max = fn_max
        self.lr_steps = lr_steps
        self.period_1 = period_1
        self.period_2 = period_2
        self.steps = steps
        self.threshold_1 = threshold_1
        self.threshold_2 = threshold_2
        self.verbose_steps = verbose_steps
        self.vis_window = "Visualisation"
        self.vis_steps = vis_steps
        self.vis_scale = 1.0
        self.video_filename = video_filename

        # Private variables

        self._default_weights = (0.5, 0.5, 0.25, 0.25, 0)
        self._weights = None

        self._blob_size = None  # For storing number of +ve pixels
        self._step = 0  # For tracking the number of optimization steps
        self._e = None  # For storing the current ellipse prediction
        self._loss = None   # For storing the loss
        self._loss_as_float = None  # For storing the loss as a float (without autograd)
        self._metrics = None  # For storing metric values
        self._fn_weight = None  # For applying weight to false negatives
        self._base = None  # For increasing fn_weight incrementally
        self._parameter_names = ("X", "Y", "W", "H", "Theta")
        self._video_writer = None

        # Set internal parameters
        self.set_weights()

        # Optimizer variables
        self._optimizer_cls = torch.optim.Adam   # Default optimzer class
        self._optimizer_args = {"lr": 1e-2}  # Default kwargs for optimizer
        self._optimizer = self._optimizer_cls(self.parameters(), **self._optimizer_args)  # Init optimizer

        # Lr scheduler variables
        self._lr_scheduler_cls = torch.optim.lr_scheduler.ExponentialLR  # Default lr scheduler class
        self._lr_scheduler_args = {"gamma": 0.9}  # Default kwargs for lr_scheduler
        self._lr_scheduler = self._lr_scheduler_cls(self.optimizer, **self._lr_scheduler_args)  # Init lr scheduler

    # Public methods

    def fit(self, steps=None):
        """
        :param steps:
        :return:
        """
        assert self.image is not None
        assert self.optimizer is not None

        self.steps = self.steps if steps is None else steps
        image = torch.tensor(self.image, dtype=torch.float32, device=self.device)

        print("PHASE 1 - WARM UP")
        self.__phase_1(image)
        print("PHASE 2 - EXPAND")
        self.__phase_2(image)
        print("PHASE 3 - ENCOMPASS")
        self.__phase_3(image)
        print("PHASE 4 - FINE TUNE")
        self.__phase_4(image)
        self.print_parameters()
        self.__finish()

    def load_image(self, filename, pad=0, invert=False, threshold=None, read_flag=cv2.IMREAD_UNCHANGED):
        """
        :param filename:
        :param pad:
        :param invert:
        :param threshold:
        :return:
        """
        image = cv2.imread(filename, read_flag)
        if image is None:
            raise FileNotFoundError(filename)
        if threshold is not None:
            image[image <= threshold] = 0
        image[image > 0] = 1
        if invert:
            image = 1 - image  # Invert image from 0-hot to 1-hot
        if pad > 0:
            h, w = image.shape
            im_ = np.zeros((h + pad * 2, w + pad * 2), dtype=np.uint8)
            im_[pad:pad + h, pad:pad + w] = image
            image = im_
        self.image = image

    def get_lr(self):
        """
        :return:
        """
        assert len(self.optimizer.param_groups) == 1
        return self.optimizer.param_groups[0]["lr"]

    def print_parameters(self):
        print(", ".join(["%s: %f" % (k, v) for k, v in zip(self._parameter_names, self.weights)]))

    def print_status(self, check_step=False):
        """
        :param check_step:
        :return:
        """
        if not check_step or (self.step + 1) % self.verbose_steps == 0:
            msg = "STEP: %s | Loss: %.3e, FN: %.3e, FP: %.3e"
            print(msg % (self.step_as_str, self.loss_as_float, self._metrics["fn"], self._metrics["fp"]))

    def reset(self):
        """
        :return:
        """
        self.set_weights()
        self.reset_optimizer()
        if self.lr_scheduler is not None:
            self.reset_lr_scheduler()
        self.plateau_monitor_1.reset()
        self.plateau_monitor_2.reset()

        self._blob_size = None
        self._step = 0
        self._e = None
        self._loss = None
        self._loss_as_float = None
        self._metrics = None
        self._fn_weight = None
        self._base = None

    def reset_lr_scheduler(self):
        """
        :return:
        """
        assert self.lr_scheduler is not None, "lr_scheduler not set"
        self._lr_scheduler = self._lr_scheduler_cls(self.optimizer, **self._lr_scheduler_args)
        print(
            "Reset LR scheduler: %s(%s)" % (
                self._lr_scheduler_cls.__name__,
                ", ".join(["%s=%s" % (k, v) for k, v in self._lr_scheduler_args.items()])
            )
        )

    def reset_optimizer(self):
        """
        :return:
        """
        self._optimizer = self._optimizer_cls(self.parameters(), **self._optimizer_args)
        print(
            "Reset optimizer: %s(%s)" % (
                self._optimizer_cls.__name__,
                ", ".join(["%s=%s" % (k, v) for k, v in self._optimizer_args.items()])
            )
        )

    def set_lr_scheduler(self, scheduler, **kwargs):
        """
        :param scheduler:
        :param kwargs:
        :return:
        """
        assert self.optimizer is not None, "Please use set_ptimizer before using set_lr_scheduler"
        self._lr_scheduler_cls = scheduler
        self._lr_scheduler_args = kwargs
        self._lr_scheduler = self._lr_scheduler_cls(self.optimizer, **self._lr_scheduler_args)
        print(
            "Set LR scheduler: %s(%s)" % (
                self._lr_scheduler_cls.__name__,
                ", ".join(["%s=%s" % (k, v) for k, v in self._lr_scheduler_args.items()])
            )
        )

    def set_optimizer(self, optimizer, **kwargs):
        """
        :param optimizer:
        :param kwargs:
        :return:
        """
        self._optimizer_cls = optimizer
        self._optimizer_args = kwargs
        self._optimizer = self._optimizer_cls(self.parameters(), **self._optimizer_args)
        print(
            "Set optimizer: %s(%s)" % (
                self._optimizer_cls.__name__,
                ", ".join(["%s=%s" % (k, v) for k, v in self._optimizer_args.items()])
            )
        )

    def set_weights(self, weights=None):
        """
        :return:
        """
        if weights is not None:
            self._default_weights = (0.5, 0.5, 0.25, 0.25, 0)
        self._weights = nn.Parameter(torch.tensor(self._default_weights, dtype=torch.float32))
        print("Set internal weights: %s" % str(self._default_weights))

    def show_image(self):
        """
        :return:
        """
        window_name = "Image"
        cv2.imshow(window_name, self.image)
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)

    # Private methods

    def __criterion(self, image):
        """
        :param image:
        :return:
        """
        self.__formulate_ellipse()

        # Calculate loss
        fn_loss = torch.sum(image * F.relu(self._e))  # relu(e): 0 inside elipse and 1 -> inf outside.
        fn_loss = fn_loss / (self.blob_size * 1 / (self.fn_weight * self.blob_size))  # weighted scale by blob size
        fp_loss = torch.sum((1 - image) * F.relu(-self._e))  # relu(-e) gives 1 -> 0 inside elipse and 0 outside.
        fp_loss = fp_loss / self.blob_size  # Scale by blob size
        self.loss = fn_loss + fp_loss

        # Calculate metrics
        fn = len(self._e[torch.bitwise_and(image == 1, self._e > 0)])
        fp = len(self._e[torch.bitwise_and(image == 0, self._e <= 0)])
        self._metrics = {"fn": fn, "fp": fp}

    def __finish(self):
        with contextlib.suppress(AttributeError):
            self._video_writer.release()
        self._video_writer = None
        with contextlib.suppress(cv2.error):
            cv2.destroyWindow(self.vis_window)

    def __formulate_ellipse(self):
        """
        - Draws an elipse as described by the parameters in self.weights.
        - Values of the result conform to -1 < I < 0 inside the elipse and 0 < I < inf outside the elipse.
        - Expects self.image, self.device, and self.weights to exist.
        :return: torch.tensor with the same shape as self.image.
        """
        h, w = self.image.shape
        my, mx = torch.meshgrid(torch.arange(h), torch.arange(w))
        my = my.to(device=self.device, dtype=torch.float32)
        mx = mx.to(device=self.device, dtype=torch.float32)
        px, py, pw, ph, t = self.weights   # Predicted x, a, y, b, theta
        a = pw * w
        b = ph * h
        self._e = (
            (mx - px * w)**2 * (a**2 * torch.sin(t)**2 + b**2 * torch.cos(t)**2)
            + (my - py * h)**2 * (a**2 * torch.cos(t)**2 + b**2 * torch.sin(t)**2)
            + 2 * (mx - px * w) * (my - py * h) * (b**2 - a**2) * torch.sin(t) * torch.cos(t)
        ) / (a**2 * b**2) - 1

    def __phase_1(self, image):
        """
        :param image:
        :return:
        """
        for self.step in count(0):
            self.__criterion(image)
            self.loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.plateau_monitor_1.step(self.loss_as_float)
            self.print_status(check_step=True)
            self.__update_lr(check_step=True)
            self.__visualize(wait=1, check_step=True)
            if self.plateau_monitor_1.plateaued:
                msg = "STEP: %s | Plateauted: Loss decreased by an average of %f%% in the last %i steps"
                print(msg % (self.step_as_str, self.plateau_monitor_1.average_change, self.plateau_monitor_1.period))
                break

    def __phase_2(self, image):
        """
        :param image:
        :return:
        """
        for self.step in range(self.steps):
            self.fn_weight = (1 / (self.blob_size * self.base ** (self.step + 1)))
            self.__criterion(image)
            self.loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.print_status(check_step=True)
            self.__update_lr(check_step=True)
            self.__visualize(wait=1, check_step=True)

    def __phase_3(self, image):
        """
        :param image:
        :return:
        """
        for self.step in range(self.steps):
            self.fn_weight = (self.fn_max - 1) * ((self.step + 1) / self.steps) + 1
            self.__criterion(image)
            self.loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.print_status(check_step=True)
            self.__update_lr(check_step=True)
            self.__visualize(wait=1, check_step=True)

    def __phase_4(self, image):
        """
        :param image:
        :return:
        """
        for self.step in count(0):
            self.__criterion(image)
            self.loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.plateau_monitor_2.step(self.loss_as_float)
            self.print_status(check_step=True)
            self.__update_lr(check_step=True)
            self.__visualize(wait=1, check_step=True)
            if self.plateau_monitor_2.plateaued:
                msg = "STEP: %s | Plateauted: Loss decreased by an average of %f%% in the last %i steps"
                print(msg % (self.step_as_str, self.plateau_monitor_2.average_change, self.plateau_monitor_2.period))
                break

    def __update_lr(self, check_step=False):
        """
        :param check_step:
        :return:
        """
        if (not check_step or (self.step + 1) % self.lr_steps == 0) and self.lr_scheduler is not None:
            lr0 = self.get_lr()
            self.lr_scheduler.step()
            msg = "STEP: %s | Learning rate changed from %.3e to %.3e"
            print(msg % (self.step_as_str, lr0, self.get_lr()))

    def __visualize(self, wait=0, fn_col=(38, 38, 210), tp_col=(38, 210, 38), fp_col=(38, 110, 38), check_step=False):
        """
        :param wait:
        :param fn_col:
        :param tp_col:
        :param fp_col:
        :param check_step:
        :return:
        """
        if not check_step or (self.step + 1) % self.vis_steps == 0:

            # Get ellipse as binary image
            e = self._e.detach().cpu().numpy()
            e[e <= 0] = -1
            e[e > 0] = 0
            e = -e

            # Create image
            image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)  # Convert to RGBA
            image[np.all(image == 1, axis=2)] = fn_col  # Colour the blob
            image[e * self.image == 1] = tp_col  # Colour the area of blob which is inside the ellipse
            image[e * (1 - self.image) == 1] = fp_col  # Colour area which is outside blob & inside ellipse

            # Write to video (if video_filename is not None)
            if self.video_filename is not None:
                if self._video_writer is None:
                    h, w = self.image.shape[0:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    self._video_writer = cv2.VideoWriter(self.video_filename, fourcc, 60, (w, h))
                if self._video_writer.isOpened() is False:
                    raise Warning("Error opening video stream or file")
                self._video_writer.write(image)

            # Display the image
            cv2.imshow(self.vis_window, image)
            cv2.waitKey(wait)

    # Properties

    @property
    def base(self):
        if self._base is None:
            self._base = (1 / self.blob_size) ** (1 / self.steps)
        return self._base

    @property
    def blob_size(self):
        assert self.image is not None
        if self._blob_size is None:
            self._blob_size = np.sum(self.image)
        return self._blob_size

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def fn_weight(self):
        if self._fn_weight is None:
            self._fn_weight = 1 / self.blob_size
        return self._fn_weight

    @property
    def image(self):
        return self._image

    @property
    def metrics(self):
        return self._metrics

    @property
    def loss(self):
        return self._loss

    @property
    def loss_as_float(self):
        return self._loss_as_float

    @property
    def lr_scheduler(self):
        return self._lr_scheduler

    @property
    def lr_scheduler_args(self):
        return self._lr_scheduler_args

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def optimizer_args(self):
        return self._optimizer_args

    @property
    def period_1(self):
        return self._period_1

    @property
    def period_2(self):
        return self._period_2

    @property
    def step_as_str(self):
        return str(self.step + 1).ljust(6)

    @property
    def threshold_1(self):
        return self._threshold_1

    @property
    def threshold_2(self):
        return self._threshold_2

    @property
    def weights(self):
        return self._weights

    # Property setters

    @image.setter
    def image(self, new_image):
        """
        :param new_image:
        :return:
        """
        if new_image is not None:
            assert isinstance(new_image, np.ndarray), "Image must be a NumPy array"
            assert new_image.ndim == 2, "Image must have exactly 2 dimensions"
            assert np.min(new_image) == 0 and np.max(new_image == 1), \
                "Image must be binary (0 and 1), but has min and max of %i and %i" % (np.min(new_image), np.max(new_image))
        self._image = new_image

    @loss.setter
    def loss(self, new_value):
        """
        Store given value as loss and store a detached version as loss_as_float
        :param new_value: Current loss as torch tensor with auto_grad
        :return: None
        """
        assert isinstance(new_value, torch.Tensor)
        assert new_value.requires_grad
        self._loss = new_value
        self._loss_as_float = new_value.detach().cpu().item()

    @lr_scheduler_args.setter
    def lr_scheduler_args(self, new_value):
        assert isinstance(new_value, dict)
        self._lr_scheduler_args = new_value
        self._lr_scheduler = self._lr_scheduler_cls(self.optimizer, **self._lr_scheduler_args)
        print(
            "Set LR scheduler: %s(%s)" % (
                self._lr_scheduler_cls.__name__,
                ", ".join(["%s=%s" % (k, v) for k, v in self._lr_scheduler_args.items()])
            )
        )

    @optimizer_args.setter
    def optimizer_args(self, new_value):
        assert isinstance(new_value, dict)
        self._optimizer_args = new_value
        self._optimizer = self._optimizer_cls(self.parameters(), **self._optimizer_args)
        print(
            "Set optimizer: %s(%s)" % (
                self._optimizer_cls.__name__,
                ", ".join(["%s=%s" % (k, v) for k, v in self._optimizer_args.items()])
            )
        )

    @period_1.setter
    def period_1(self, new_value):
        """
        :param new_value:
        :return:
        """
        self._period_1 = new_value
        self.plateau_monitor_1.period = self.period_1

    @period_2.setter
    def period_2(self, new_value):
        """
        :param new_value:
        :return:
        """
        self._period_2 = new_value
        self.plateau_monitor_2.period = self.period_2

    @threshold_1.setter
    def threshold_1(self, new_value):
        """
        :param new_value:
        :return:
        """
        self._threshold_1 = new_value
        self.plateau_monitor_1.threshold = self.threshold_1

    @threshold_2.setter
    def threshold_2(self, new_value):
        """
        :param new_value:
        :return:
        """
        self._threshold_2 = new_value
        self.plateau_monitor_2.threshold = self.threshold_2

    @fn_weight.setter
    def fn_weight(self, new_value):
        """
        :param new_value:
        :return:
        """
        self._fn_weight = new_value


if __name__ == "__main__":

    # Initialise model
    model = Model()
    #model.to(torch.device("cuda:0"))
    model.lr_scheduler_args = {"gamma": 0.75}

    model.load_image("examples/01.png", pad=100)
    #model.video_filename = "blob-1.mp4"
    model.fit()

    model.reset()
    model.load_image("examples/02.png", pad=100)
    #model.video_filename = "blob-2.mp4"
    model.fit()

    model.reset()
    model.load_image("examples/03.png", pad=100)
    #model.video_filename = "blob-3.mp4"
    model.fit()
