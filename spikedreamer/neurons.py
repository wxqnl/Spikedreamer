"""Functional spiking neurons: recurrent state belongs to each trajectory."""

import torch
from torch import nn
from .vendor.normalization import PopNorm
from .vendor.surrogate import QGateGrad
from .vendor.tools import weight_init


class LIFStack(nn.Module):
    def __init__(self, inputs, hidden, layers, c):
        super().__init__()
        self.hidden, self.depth = hidden, layers
        self.tau, self.threshold = c.tau, c.threshold
        self.projections = nn.ModuleList(
            nn.Linear(inputs if i == 0 else hidden, hidden, bias=False)
            for i in range(layers))
        self.norms = nn.ModuleList(
            PopNorm(hidden, threshold=c.threshold * c.core_norm_gain, v_reset=0.0, eps=1e-3)
            for _ in range(layers))
        self.fire = QGateGrad(alpha=2.0, requires_grad=False)
        self.apply(weight_init)

    def initial(self, batch):
        return self.projections[0].weight.new_zeros(batch, self.depth, self.hidden)

    def forward(self, x, state):
        membranes = []
        for i, (linear, norm) in enumerate(zip(self.projections, self.norms)):
            voltage = state[:, i] + (norm(linear(x)) - state[:, i]) / self.tau
            x = self.fire(voltage - self.threshold)
            membranes.append(voltage * (1 - x.detach()))
        return x, torch.stack(membranes, 1)


class MCNCell(nn.Module):
    """Same nonlinear integration as the released MCNode, with explicit state."""
    def __init__(self, inputs, size, c):
        super().__init__()
        self.basal_w = nn.Linear(inputs + size, size)
        self.apical_w = nn.Linear(inputs + size, size)
        self.soma_w = nn.Linear(inputs, size)
        self.norms = nn.ModuleList(
            PopNorm(size, threshold=c.threshold * c.core_norm_gain, v_reset=0.0, eps=1e-3)
            for _ in range(3))
        self.tau, self.tau_b, self.tau_a = c.tau, c.tau_basal, c.tau_apical
        self.threshold = c.threshold
        self.reset_dendrites = c.reset_dendrites
        self.memoryless = c.method == "binary"
        self.fire = QGateGrad(alpha=c.tau, requires_grad=False)
        self.apply(weight_init)

    def forward(self, x, state):
        combined = torch.cat([x, state["deter"]], -1)
        basal_input = self.norms[0](self.basal_w(combined))
        apical_input = self.norms[1](self.apical_w(combined))
        soma_input = self.norms[2](self.soma_w(x))
        if self.memoryless:
            voltage = torch.sigmoid(apical_input) * (basal_input + soma_input) / self.tau
            spike = self.fire(voltage - self.threshold)
            zeros = torch.zeros_like(spike)
            return dict(deter=spike, soma=zeros, basal=zeros, apical=zeros)
        basal = state["basal"] + (basal_input - state["basal"]) / self.tau_b
        apical = state["apical"] + (apical_input - state["apical"]) / self.tau_a
        soma = state["soma"]
        # Keep the released code's update, including its two soma terms.
        voltage = soma + torch.sigmoid(apical) * (basal - 2 * soma + soma_input) / self.tau
        spike = self.fire(voltage - self.threshold)
        reset = 1 - spike.detach()
        return dict(deter=spike, soma=voltage * reset,
                    basal=basal * reset if self.reset_dendrites else basal,
                    apical=apical * reset if self.reset_dendrites else apical)
