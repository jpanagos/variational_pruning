import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchNorm1dPruning(nn.BatchNorm1d):

    def __init__(self, num_features, eps=1e-05, momentum=0.1,
                 affine = True, track_running_stats = True):
        super(BatchNorm1dPruning, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        # initialize saliency distribution parameters
        self.saliency_m = torch.nn.Parameter(torch.abs(torch.randn_like(self.weight)))
        self.saliency_logvar = torch.nn.Parameter(torch.abs(torch.randn_like(self.weight)))

        # initialize pruning mask
        self.register_buffer('mask', torch.ones_like(self.weight))

    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # sample
        saliency_s = torch.exp(0.5 * self.saliency_logvar)
        eps = torch.randn_like(saliency_s)
        channel_saliency = self.saliency_m + eps * saliency_s
        # perform pruning
        channel_saliency *= self.mask

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 2])
            # use biased var in train
            var = input.var([0, 2], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        output = (input - mean[None, :, None]) / (torch.sqrt(var[None, :, None] + self.eps))
        output = channel_saliency[None, :, None] * (output + self.bias[None, :, None])
        #if self.affine:
        #    output = output * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return output

    def prune_channels(self):
        m_check = torch.gt(self.saliency_m.abs(), 0.02)
        saliency_s = torch.exp(0.5 * self.saliency_logvar)
        s_check = torch.gt(saliency_s, 0.01)
        t_mask = (m_check | s_check).float()
        self.mask *= t_mask # channel pruning is persistent through epochs
        return

    def get_kl(self, reduction='sum', k=0.5):
        saliency_s = torch.exp(0.5 * self.saliency_logvar)
        if reduction == 'sum':
            return (-k)*torch.sum(torch.square(self.saliency_m / saliency_s))
        else:
            return (-k)*torch.mean(torch.square(self.saliency_m / saliency_s))


class BatchNorm2dPruning(nn.BatchNorm2d):

    def __init__(self, num_features, eps=1e-05, momentum=0.1,
                 affine = True, track_running_stats = True):
        super(BatchNorm2dPruning, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        # initialize saliency distribution parameters
        self.saliency_m = torch.nn.Parameter(torch.abs(torch.randn_like(self.weight)))
        self.saliency_logvar = torch.nn.Parameter(torch.abs(torch.randn_like(self.weight)))

        # initialize pruning mask
        self.register_buffer('mask', torch.ones_like(self.weight))

    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # sample
        saliency_s = torch.exp(0.5 * self.saliency_logvar)
        eps = torch.randn_like(saliency_s)
        channel_saliency = self.saliency_m + eps * saliency_s
        # perform pruning
        channel_saliency *= self.mask

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 2, 3])
            # use biased var in train
            var = input.var([0, 2, 3], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        output = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
        output = channel_saliency[None, :, None, None] * (output + self.bias[None, :, None, None])
        #if self.affine:
        #    output = output * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return output

    def prune_channels(self):
        m_check = torch.gt(self.saliency_m.abs(), 0.02)
        saliency_s = torch.exp(0.5 * self.saliency_logvar)
        s_check = torch.gt(saliency_s, 0.01)
        t_mask = (m_check | s_check).float()
        self.mask *= t_mask # channel pruning is persistent through epochs
        return

    def get_kl(self, reduction='sum', k=0.5):
        saliency_s = torch.exp(0.5 * self.saliency_logvar)
        if reduction == 'sum':
            return (-k)*torch.sum(torch.square(self.saliency_m / saliency_s))
        else:
            return (-k)*torch.mean(torch.square(self.saliency_m / saliency_s))


class BatchNorm3dPruning(nn.BatchNorm3d):

    def __init__(self, num_features, eps=1e-05, momentum=0.1,
                 affine = True, track_running_stats = True):
        super(BatchNorm3dPruning, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        # initialize saliency distribution parameters
        self.saliency_m = torch.nn.Parameter(torch.abs(torch.randn_like(self.weight)))
        self.saliency_logvar = torch.nn.Parameter(torch.abs(torch.randn_like(self.weight)))

        # initialize pruning mask
        self.register_buffer('mask', torch.ones_like(self.weight))

    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # sample
        saliency_s = torch.exp(0.5 * self.saliency_logvar)
        eps = torch.randn_like(saliency_s)
        channel_saliency = self.saliency_m + eps * saliency_s
        # perform pruning
        channel_saliency *= self.mask

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 2, 3, 4])
            # use biased var in train
            var = input.var([0, 2, 3, 4], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        output = (input - mean[None, :, None, None, None]) / (torch.sqrt(var[None, :, None, None, None] + self.eps))
        output = channel_saliency[None, :, None, None, None] * (output + self.bias[None, :, None, None, None])
        #if self.affine:
        #    output = output * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return output

    def prune_channels(self):
        m_check = torch.gt(self.saliency_m.abs(), 0.02)
        saliency_s = torch.exp(0.5 * self.saliency_logvar)
        s_check = torch.gt(saliency_s, 0.01)
        t_mask = (m_check | s_check).float()
        self.mask *= t_mask # channel pruning is persistent through epochs
        return

    def get_kl(self, reduction='sum', k=0.5):
        saliency_s = torch.exp(0.5 * self.saliency_logvar)
        if reduction == 'sum':
            return (-k)*torch.sum(torch.square(self.saliency_m / saliency_s))
        else:
            return (-k)*torch.mean(torch.square(self.saliency_m / saliency_s))


### Alternative implementation ###

class BatchNorm1dPruningAbs(nn.BatchNorm1d):

    def __init__(self, num_features, eps=1e-05, momentum=0.1,
                 affine = True, track_running_stats = True):
        super(BatchNorm1dPruningAbs, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        # initialize saliency distribution parameters
        self.saliency_m = torch.nn.Parameter(torch.empty_like(self.weight))
        self.saliency_logvar = torch.nn.Parameter(torch.empty_like(self.weight))

        self.saliency_m.data.normal_(0, math.sqrt(2.0/float(num_features))).abs_()
        self.saliency_logvar.data.normal_(0, math.sqrt(2.0/float(num_features))).abs_()

        # initialize pruning mask
        self.register_buffer('mask', torch.ones_like(self.weight))

    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # sample
        saliency_s = torch.exp(0.5 * self.saliency_logvar)
        eps = torch.randn_like(saliency_s)
        channel_saliency = self.saliency_m + eps * saliency_s
        # perform pruning
        channel_saliency *= self.mask

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 2])
            # use biased var in train
            var = input.var([0, 2], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        output = (input - mean[None, :, None]) / (torch.sqrt(var[None, :, None] + self.eps))
        output = channel_saliency[None, :, None] * (output + self.bias[None, :, None])
        #if self.affine:
        #    output = output * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return output

    def prune_channels(self):
        m_check = torch.gt(self.saliency_m.abs(), 0.02)
        #saliency_s = torch.exp(0.5 * self.saliency_logvar)
        s_check = torch.gt(self.saliency_logvar.abs(), 0.01)
        t_mask = (m_check | s_check).float()
        self.mask *= t_mask # channel pruning is persistent through epochs
        return

    def get_kl(self):
        saliency_s = torch.exp(0.5 * self.saliency_logvar)
        return (-0.05)*torch.mean(torch.square(self.saliency_m / saliency_s))


class BatchNorm2dPruningAbs(nn.BatchNorm2d):

    def __init__(self, num_features, eps=1e-05, momentum=0.1,
                 affine = True, track_running_stats = True):
        super(BatchNorm2dPruningAbs, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        # initialize saliency distribution parameters
        self.saliency_m = torch.nn.Parameter(torch.empty_like(self.weight))
        self.saliency_logvar = torch.nn.Parameter(torch.empty_like(self.weight))

        self.saliency_m.data.normal_(0, math.sqrt(2.0/float(num_features))).abs_()
        self.saliency_logvar.data.normal_(0, math.sqrt(2.0/float(num_features))).abs_()

        # initialize pruning mask
        self.register_buffer('mask', torch.ones_like(self.weight))

    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # sample
        saliency_s = torch.exp(0.5 * self.saliency_logvar)
        eps = torch.randn_like(saliency_s)
        channel_saliency = self.saliency_m + eps * saliency_s
        # perform pruning
        channel_saliency *= self.mask

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 2, 3])
            # use biased var in train
            var = input.var([0, 2, 3], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        output = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
        output = channel_saliency[None, :, None, None] * (output + self.bias[None, :, None, None])
        #if self.affine:
        #    output = output * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return output

    def prune_channels(self):
        m_check = torch.gt(self.saliency_m.abs(), 0.02)
        #saliency_s = torch.exp(0.5 * self.saliency_logvar)
        s_check = torch.gt(self.saliency_logvar.abs(), 0.01)
        t_mask = (m_check | s_check).float()
        self.mask *= t_mask # channel pruning is persistent through epochs
        return

    def get_kl(self):
        saliency_s = torch.exp(0.5 * self.saliency_logvar)
        return (-0.05)*torch.mean(torch.square(self.saliency_m / saliency_s))


class BatchNorm3dPruningAbs(nn.BatchNorm3d):

    def __init__(self, num_features, eps=1e-05, momentum=0.1,
                 affine = True, track_running_stats = True):
        super(BatchNorm3dPruningAbs, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        # initialize saliency distribution parameters
        self.saliency_m = torch.nn.Parameter(torch.empty_like(self.weight))
        self.saliency_logvar = torch.nn.Parameter(torch.empty_like(self.weight))

        self.saliency_m.data.normal_(0, math.sqrt(2.0/float(num_features))).abs_()
        self.saliency_logvar.data.normal_(0, math.sqrt(2.0/float(num_features))).abs_()

        # initialize pruning mask
        self.register_buffer('mask', torch.ones_like(self.weight))

    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # sample
        saliency_s = torch.exp(0.5 * self.saliency_logvar)
        eps = torch.randn_like(saliency_s)
        channel_saliency = self.saliency_m + eps * saliency_s
        # perform pruning
        channel_saliency *= self.mask

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 2, 3, 4])
            # use biased var in train
            var = input.var([0, 2, 3, 4], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        output = (input - mean[None, :, None, None, None]) / (torch.sqrt(var[None, :, None, None, None] + self.eps))
        output = channel_saliency[None, :, None, None, None] * (output + self.bias[None, :, None, None, None])
        #if self.affine:
        #    output = output * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return output

    def prune_channels(self):
        m_check = torch.gt(self.saliency_m.abs(), 0.02)
        #saliency_s = torch.exp(0.5 * self.saliency_logvar)
        s_check = torch.gt(self.saliency_logvar.abs(), 0.01)
        t_mask = (m_check | s_check).float()
        self.mask *= t_mask # channel pruning is persistent through epochs
        return

    def get_kl(self):
        saliency_s = torch.exp(0.5 * self.saliency_logvar)
        return (-0.05)*torch.mean(torch.square(self.saliency_m / saliency_s))
