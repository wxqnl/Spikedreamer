"""RSSMs expose batch-first state tensors with no hidden module state."""

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Independent, kl_divergence

from .config import upstream_config
from .neurons import LIFStack, MCNCell
from .vendor import networks, tools


def detach_state(state):
    return {k: v.detach() for k, v in state.items()}


def flatten_state(state):
    return {k: v.flatten(0, 1) for k, v in state.items()}


def stack_states(states, dim=1):
    return {k: torch.stack([s[k] for s in states], dim) for k in states[0]}


class RSSMBase(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.feature_size = c.deter + c.stoch * c.classes

    def get_dist(self, state):
        return Independent(tools.OneHotDist(
            state["logit"].float(), unimix_ratio=self.c.unimix), 1)

    def draw(self, logits, sample):
        distribution = tools.OneHotDist(logits.float(), unimix_ratio=self.c.unimix)
        if sample:
            return distribution.sample()
        hard = F.one_hot(distribution.probs.argmax(-1), self.c.classes).float()
        return hard + (distribution.probs - distribution.probs.detach())

    def reset_state(self, state, is_first):
        initial = self.initial(is_first.shape[0])
        return {k: torch.where(
            is_first.bool().reshape((-1,) + (1,) * (v.ndim - 1)), initial[k], v)
            for k, v in state.items()}

    def observe(self, embed, actions, is_first, state=None, sample=True):
        state = self.initial(actions.shape[0]) if state is None else state
        posts, priors = [], []
        for h in range(actions.shape[1]):
            state, prior = self.obs_step(state, actions[:, h], embed[:, :, h],
                                         is_first[:, h], sample=sample)
            posts.append(state)
            priors.append(prior)
        return stack_states(posts), stack_states(priors)

    def imagine(self, actions, state, sample=True):
        states = []
        for h in range(actions.shape[1]):
            state = self.img_step(state, actions[:, h], sample=sample)
            states.append(state)
        return stack_states(states)

    def kl_loss(self, post, prior):
        dyn = kl_divergence(self.get_dist(detach_state(post)), self.get_dist(prior))
        rep = kl_divergence(self.get_dist(post), self.get_dist(detach_state(prior)))
        return (self.c.dyn_scale * dyn.clamp_min(self.c.kl_free) +
                self.c.rep_scale * rep.clamp_min(self.c.kl_free)).mean(), dyn, rep


class LegacyRSSM(RSSMBase):
    """Wrap the released MCN RSSM; only move its spike axis to a trailing axis."""
    def __init__(self, c, actions, embed):
        super().__init__(c)
        self.raw = networks.RSSM(
            upstream_config(c), stoch=c.stoch, deter=c.deter, hidden=c.hidden,
            layers_input=c.input_layers, layers_output=c.output_layers,
            discrete=c.classes, act="LIFNode", norm="PopNorm", cell="MCRNN",
            unimix_ratio=c.unimix, initial="learned", num_actions=actions,
            embed=embed, device=c.device)

    @staticmethod
    def external(raw):
        return {k: v.movedim(0, -2) if k == "deter" else v
                for k, v in raw.items()}

    @staticmethod
    def internal(state):
        return {k: v.movedim(-2, 0) if k == "deter" else v
                for k, v in state.items()}

    def initial(self, batch):
        return self.external(self.raw.initial(batch))

    def obs_step(self, state, action, embed, is_first, sample=True):
        # Upstream mutates action and state dictionaries; never mutate caller data.
        post, prior = self.raw.obs_step(self.internal(state), action.clone(),
                                         embed, is_first.to(action.dtype).clone(), sample=sample)
        return self.external(post), self.external(prior)

    def img_step(self, state, action, sample=True):
        return self.external(self.raw.img_step(
            self.internal(state), action.clone(), sample=sample))

    def get_feat(self, state):
        return self.raw.get_feat(self.internal(state))


class TimeAlignedRSSM(RSSMBase):
    def __init__(self, c, actions, embed):
        super().__init__(c)
        self.input = LIFStack(c.stoch * c.classes + actions, c.hidden, c.input_layers, c)
        self.cell = MCNCell(c.hidden, c.deter, c)
        self.prior_hidden = LIFStack(c.deter, c.hidden, c.output_layers, c)
        self.posterior_hidden = LIFStack(c.deter + embed, c.hidden, c.output_layers, c)
        self.prior_logits = nn.Linear(c.hidden, c.stoch * c.classes)
        self.posterior_logits = nn.Linear(c.hidden, c.stoch * c.classes)
        self.prior_logits.apply(tools.weight_init)
        self.posterior_logits.apply(tools.weight_init)
        self.initial_membrane = nn.Parameter(torch.zeros(3, c.deter))
        if c.method == "gru":
            # Same single-step spike encoder/heads, with a GRU dynamics control.
            self.gru = nn.GRUCell(c.hidden, c.deter)
            del self.cell

    def initial(self, batch):
        zeros = self.initial_membrane.new_zeros
        logit = zeros(batch, self.c.stoch, self.c.classes)
        return dict(logit=logit, stoch=self.draw(logit, False),
                    deter=zeros(batch, self.c.deter),
                    soma=self.initial_membrane[0].expand(batch, -1),
                    basal=self.initial_membrane[1].expand(batch, -1),
                    apical=self.initial_membrane[2].expand(batch, -1),
                    input_mem=self.input.initial(batch),
                    prior_mem=self.prior_hidden.initial(batch),
                    steps=zeros(batch, 1))

    def readout(self, state):
        return state["soma"] if self.c.readout == "membrane" else state["deter"]

    def _core(self, x, state):
        if self.c.method == "gru":
            deter = self.gru(x, state["deter"])
            return dict(deter=deter, soma=deter, basal=state["basal"],
                        apical=state["apical"])
        return self.cell(x, state)

    def _advance(self, x, state):
        core = self._core(x, state)
        s, membrane = self.prior_hidden(self.readout(core), state["prior_mem"])
        logits = self.prior_logits(s).reshape(-1, self.c.stoch, self.c.classes)
        return {**state, **core, "prior_mem": membrane, "logit": logits}

    def img_step(self, state, action, sample=True):
        if not self.c.persistent:
            reset = self.initial(action.shape[0])
            state = {k: reset[k] if k in ("soma", "basal", "apical", "input_mem", "prior_mem")
                     else v for k, v in state.items()}
        action = action / action.abs().clamp_min(1).detach()
        inputs = torch.cat([state["stoch"].flatten(-2), action], -1)
        x, input_mem = self.input(inputs, state["input_mem"])
        result = self._advance(x, state)
        result["input_mem"] = input_mem
        steps = torch.ones_like(state["steps"])
        # Gate is an explicit, non-differentiable inference rule; no fictitious
        # gradient through integer compute counts. Entropy uses the actual prior.
        for k in range(1, self.c.adaptive_max_steps):
            with torch.no_grad():
                p = self.get_dist(result).base_dist.probs
                entropy = (-(p * p.clamp_min(1e-8).log()).sum(-1).mean(-1) /
                           math.log(self.c.classes)).clamp(0, 1)
                active = ((entropy > self.c.entropy_threshold) &
                          (steps[:, 0] == k)).nonzero().flatten()
            if active.numel() == 0:
                break
            small = {name: value.index_select(0, active) for name, value in result.items()}
            # External action/latent evidence is consumed once. Additional updates
            # receive zero external spikes and the most recent recurrent spike.
            refined = self._advance(torch.zeros_like(x.index_select(0, active)), small)
            result = {name: value.index_copy(0, active, refined[name])
                      for name, value in result.items()}
            steps = steps.index_add(0, active, torch.ones_like(steps.index_select(0, active)))
        result["steps"] = steps
        result["stoch"] = self.draw(result["logit"], sample)
        return result

    def obs_step(self, state, action, embed, is_first, sample=True):
        state = self.reset_state(state, is_first)
        action = action * (~is_first.bool()).unsqueeze(-1)
        prior = self.img_step(state, action, sample=sample)
        # Encoder is a per-observation SNN readout. With io_steps>1 the baseline
        # isolation configuration uses its temporal mean as the boundary input.
        obs = embed.mean(0)
        x, _ = self.posterior_hidden(
            torch.cat([self.readout(prior), obs], -1),
            self.posterior_hidden.initial(action.shape[0]))
        logits = self.posterior_logits(x).reshape(-1, self.c.stoch, self.c.classes)
        post = {**prior, "logit": logits, "stoch": self.draw(logits, sample)}
        return post, prior

    def get_feat(self, state):
        feat = torch.cat([state["stoch"].flatten(-2), self.readout(state)], -1)
        # A singleton interface is the main setting; io_steps=8 isolates only the
        # RSSM change while keeping the original encoder/decoder/behavior cost.
        return feat.unsqueeze(0).expand(self.c.io_steps, *feat.shape)


def make_rssm(c, actions, embed):
    return LegacyRSSM(c, actions, embed) if c.method == "legacy" else TimeAlignedRSSM(c, actions, embed)
