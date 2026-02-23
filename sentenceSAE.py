import torch
import torch.nn as nn
import torch.nn.functional as F


class TiedTranspose(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.linear = linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.linear.bias is None
        return F.linear(x, self.linear.weight.t(), None)

    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight.t()

    @property
    def bias(self) -> torch.Tensor:
        return self.linear.bias


class Autoencoder(nn.Module):
    def __init__(self, n_latents, n_inputs, sparsity_factor, activation=nn.ReLU()):
        super().__init__()
        self.sparsity_factor = sparsity_factor
        self.n_inputs = n_inputs
        self.n_latents = n_latents
        self.encoder = nn.Linear(n_inputs, n_latents, bias=False)
        self.latent_bias = nn.Parameter(torch.zeros(n_latents))
        self.activation = activation
        
            
    def forward(self, x):
        """
        :param x: last token embedding (shape: [batch, 1, n_inputs])
        :return latents: (batch, 1, n_latents)
        """
        latents_pre_act = self.encoder(x) + self.latent_bias # latents shape: [batch, 1, n_latents]
        latents = self.activation(latents_pre_act) # latents shape: [batch, 1, n_latents]
        return latents