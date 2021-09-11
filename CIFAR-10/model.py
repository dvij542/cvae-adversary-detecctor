import torch.nn as nn
import torch 
from torch.autograd import Variable

def idx2onehot(idx, n):
    assert torch.max(idx).item() < n

    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n).to(idx.device)
    onehot.scatter_(1, idx, 1)
    
    return onehot

class VAE_conv(nn.Module):
    def __init__(self, label, image_size, channel_num, kernel_num, z_size, conditional=False,n_labels=0):
        # configurations
        super().__init__()
        self.label = label
        self.image_size = image_size
        self.channel_num = channel_num
        self.kernel_num = kernel_num
        self.z_size = z_size

        # encoder
        self.encoder = nn.Sequential(
            self._conv(channel_num, kernel_num // 4),
            self._conv(kernel_num // 4, kernel_num // 2),
            self._conv(kernel_num // 2, kernel_num, last=True),
        )

        # encoded feature's size and volume
        self.feature_size = image_size // 8
        self.feature_volume = kernel_num * (self.feature_size ** 2)
        self.conditional = conditional
        if self.conditional:
            # q
            self.q_mean = self._linear(self.feature_volume+n_labels, z_size, relu=False)
            self.q_logvar = self._linear(self.feature_volume+n_labels, z_size, relu=False)
            print(z_size+n_labels)
            self.project = self._linear(z_size+n_labels, self.feature_volume, relu=False)
        else :
            # q
            self.q_mean = self._linear(self.feature_volume, z_size, relu=False)
            self.q_logvar = self._linear(self.feature_volume, z_size, relu=False)
            self.project = self._linear(z_size, self.feature_volume, relu=False)

        # decoder
        self.decoder = nn.Sequential(
            self._deconv(kernel_num, kernel_num // 2),
            self._deconv(kernel_num // 2, kernel_num // 4),
            self._deconv(kernel_num // 4, channel_num, last=True),
            nn.Sigmoid()
        )

    def forward(self, x, c=None):
        # encode x
        encoded = self.encoder(x)
        if self.conditional:
            c = idx2onehot(c, n=n_labels)
            # encoded = torch.cat((encoded, c), dim=-1)
        # sample latent code z from q given x.
        mean, logvar = self.q(encoded,c)
        z = self.z(mean, logvar)
        if self.conditional:
            # c = idx2onehot(c, n=n_labels)
            z = torch.cat((z, c), dim=-1)
        
        z_projected = self.project(z).view(
            -1, self.kernel_num,
            self.feature_size,
            self.feature_size,
        )

        # reconstruct x from z
        x_reconstructed = self.decoder(z_projected)

        # return the parameters of distribution of q given x and the
        # reconstructed image.
        return x_reconstructed, mean, logvar, z
    
    def inference(self, z, c=None):
        # c = idx2onehot(c, n=n_labels)
        # z = torch.cat((z, c), dim=-1)
        # print(z.shape)
        # print(self.conditional)
        z_projected = self.project(z)
        z_projected = z_projected.view(
            -1, self.kernel_num,
            self.feature_size,
            self.feature_size,
        )
        recon_x = self.decoder(z_projected)
        return recon_x
    # ==============
    # VAE components
    # ==============

    def q(self, encoded,c=None):
        if c is None :
          unrolled = encoded.view(-1, self.feature_volume)
        else :
          unrolled = encoded.view(-1, self.feature_volume)
          unrolled = torch.cat((unrolled, c), dim=-1)

        return self.q_mean(unrolled), self.q_logvar(unrolled)

    def z(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        eps = (
            Variable(torch.randn(std.size())).cuda() if self._is_on_cuda else
            Variable(torch.randn(std.size()))
        )
        return eps.mul(std).add_(mean)

    def reconstruction_loss(self, x_reconstructed, x):
        return nn.BCELoss(size_average=False)(x_reconstructed, x) / x.size(0)

    def kl_divergence_loss(self, mean, logvar):
        return ((mean**2 + logvar.exp() - 1 - logvar) / 2).sum() / mean.size(0)

    # =====
    # Utils
    # =====

    @property
    def name(self):
        return (
            'VAE'
            '-{kernel_num}k'
            '-{label}'
            '-{channel_num}x{image_size}x{image_size}'
        ).format(
            label=self.label,
            kernel_num=self.kernel_num,
            image_size=self.image_size,
            channel_num=self.channel_num,
        )

    def sample(self, size):
        z = Variable(
            torch.randn(size, self.z_size).cuda() if self._is_on_cuda() else
            torch.randn(size, self.z_size)
        )
        z_projected = self.project(z).view(
            -1, self.kernel_num,
            self.feature_size,
            self.feature_size,
        )
        return self.decoder(z_projected).data

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda

    # ======
    # Layers
    # ======

    def _conv(self, channel_size, kernel_num, last=False):
        conv = nn.Conv2d(
                channel_size, kernel_num,
                kernel_size=3, stride=2, padding=1,
        )
        return conv if last else nn.Sequential(
            conv,
            nn.BatchNorm2d(kernel_num),
            nn.ReLU(),
        )

    def _deconv(self, channel_num, kernel_num, last=False):
        deconv = nn.ConvTranspose2d(
            channel_num, kernel_num,
            kernel_size=4, stride=2, padding=1,
        )
        return deconv if last else nn.Sequential(
            deconv,
            nn.BatchNorm2d(kernel_num),
            nn.ReLU(),
        )

    def _linear(self, in_size, out_size, relu=True):
        return nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.ReLU(),
        ) if relu else nn.Linear(in_size, out_size)