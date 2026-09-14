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
        # One host transfer per sequence, not a CUDA-dependent Python branch
        # at each of the 64 observations. These flags never carry gradients.
        resets = is_first.bool().any(0).cpu().tolist()
        for h in range(actions.shape[1]):
            state, prior = self.obs_step(state, actions[:, h], embed[:, :, h],
                                         is_first[:, h], sample=sample, reset=resets[h])
            posts.append(state)
            priors.append(prior)
        return stack_states(posts), stack_states(priors)

    @torch.no_grad()
    def warmup(self, embed, actions, is_first, valid, sample=True):
        """Recover state from real preceding observations without BPTT storage."""
        if valid.shape != actions.shape[:2]:
            raise ValueError("Warm-up validity must match [batch, prefix length]")
        valid = valid.bool()
        first = is_first.bool() & valid
        state = self.initial(actions.shape[0])
        active = valid.any(0).cpu().tolist()
        resets = first.any(0).cpu().tolist()
        for h, present in enumerate(active):
            if not present:
                continue
            post, _ = self.obs_step(state, actions[:, h], embed[:, :, h],
                                    first[:, h], sample=sample, reset=resets[h])
            state = {k: torch.where(
                valid[:, h].reshape((-1,) + (1,) * (value.ndim - 1)),
                value, state[k]) for k, value in post.items()}
        return detach_state(state)

    def imagine(self, actions, state, sample=True):
        states = []
        for h in range(actions.shape[1]):
            state = self.img_step(state, actions[:, h], sample=sample)
            states.append(state)
        return stack_states(states)

    def kl_loss(self, post, prior):
        rep = kl_divergence(self.get_dist(post), self.get_dist(detach_state(prior)))
        dyn = kl_divergence(self.get_dist(detach_state(post)), self.get_dist(prior))
        # Reduce each KL before scaling, matching the released FP32 operation order.
        rep_loss = rep.clamp_min(self.c.kl_free).mean()
        dyn_loss = dyn.clamp_min(self.c.kl_free).mean()
        return self.c.dyn_scale * dyn_loss + self.c.rep_scale * rep_loss, dyn, rep


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

    def obs_step(self, state, action, embed, is_first, sample=True, reset=None):
        # Upstream mutates action and state dictionaries; never mutate caller data.
        post, prior = self.raw.obs_step(self.internal(state), action.clone(),
                                         embed, is_first.to(action.dtype).clone(),
                                         sample=sample, reset=reset)
        return self.external(post), self.external(prior)

    def img_step(self, state, action, sample=True):
        return self.external(self.raw.img_step(
            self.internal(state), action.clone(), sample=sample))

    def get_feat(self, state):
        return self.raw.get_feat(self.internal(state))


class StatefulFixedRSSM(LegacyRSSM):
    """Released fixed-T computation with trajectory-owned neuron membranes.

    Parameter names, initialization, spike-indexed recurrence, temporal readouts,
    and peripheral networks match LegacyRSSM. Only the input LIF, MCN and prior
    LIF boundary states can persist. The observation-only posterior LIF resets
    per observation; it is not part of the imagined transition.
    """
    memory_keys = ("input_mem", "soma", "basal", "apical", "prior_mem")

    def neuron_initial(self, batch):
        zeros = self.raw.W.new_zeros
        return dict(input_mem=zeros(batch, self.c.input_layers, self.c.hidden),
                    soma=zeros(batch, self.c.deter),
                    basal=zeros(batch, self.c.deter),
                    apical=zeros(batch, self.c.deter),
                    prior_mem=zeros(batch, self.c.output_layers, self.c.hidden))

    @staticmethod
    def lif_step(layers, x, state):
        membranes = []
        for i in range(0, len(layers), 3):
            current = layers[i + 1](layers[i](x))
            neuron = layers[i + 2]
            previous = state[:, i // 3]
            voltage = previous + (current - previous) / neuron.tau
            x = neuron.act_fun(voltage - neuron.threshold)
            membranes.append(voltage * (1 - x.detach()))
        return x, torch.stack(membranes, 1)

    def prior_readout(self, spikes, membrane):
        outputs = []
        for spike in spikes:
            x, membrane = self.lif_step(self.raw._img_out_layers, spike, membrane)
            outputs.append(x)
        # Preserve the released sequential sum, not a different reduction.
        stats = self.raw._suff_stats_layer("ims", sum(outputs) / self.c.core_steps)
        return stats, membrane

    def initial(self, batch):
        memory = self.neuron_initial(batch)
        deter = torch.tanh(self.raw.W).repeat(self.c.core_steps, batch, 1)
        stats, _ = self.prior_readout(deter, memory["prior_mem"])
        # Legacy initializes the categorical state from the learned deter, but
        # resets the readout membranes before the first actual transition.
        return dict(logit=self.raw.W.new_zeros(batch, self.c.stoch, self.c.classes),
                    stoch=self.raw.get_dist(stats).mode(),
                    deter=deter.movedim(0, -2), **memory,
                    steps=self.raw.W.new_zeros(batch, 1))

    def core_step(self, x, previous_spike, soma, basal, apical):
        cell = self.raw._cell
        neuron = cell.neuron
        combined = torch.cat((x, previous_spike), -1)
        apical_input = cell.apcial_w(combined)
        basal_input = cell.basal_w(combined)
        soma_input = cell.soma_w(x)
        apical_input = cell.apical_norm(apical_input)
        basal_input = cell.basal_norm(basal_input)
        soma_input = cell.soma_norm(soma_input)
        basal = basal + (basal_input - basal) / neuron.tau_basal
        apical = apical + (apical_input - apical) / neuron.tau_apical
        gate = F.sigmoid(apical)
        drive = basal - soma + soma_input
        soma = soma + gate * (drive - soma) / neuron.tau
        spike = neuron.act_fun(soma - neuron.threshold)
        reset = 1.0 - spike.detach()
        return spike, soma * reset, basal * reset, apical * reset

    def img_step(self, state, action, sample=True):
        memory = state if self.c.persistent else self.neuron_initial(action.shape[0])
        action = action * (1.0 / action.abs().clamp_min(1.0)).detach()
        inputs = torch.cat((state["stoch"].flatten(-2), action), -1)
        input_mem = memory["input_mem"]
        input_spikes = []
        # Repeat the same external latent/action evidence at every internal step,
        # exactly as the released model; this is not adaptive zero-input refinement.
        for _ in range(self.c.core_steps):
            x, input_mem = self.lif_step(self.raw._inp_layers, inputs, input_mem)
            input_spikes.append(x)
        soma, basal, apical = (memory[k] for k in ("soma", "basal", "apical"))
        spikes = []
        for step, x in enumerate(input_spikes):
            # The released recurrence pairs internal index t with h-1's index t.
            # Keep that skeleton rather than feeding the latest within-h spike.
            spike, soma, basal, apical = self.core_step(
                x, state["deter"][:, step], soma, basal, apical)
            spikes.append(spike)
        stats, prior_mem = self.prior_readout(spikes, memory["prior_mem"])
        dist = self.raw.get_dist(stats)
        return dict(**stats, stoch=dist.sample() if sample else dist.mode(),
                    deter=torch.stack(spikes, 0).movedim(0, -2),
                    input_mem=input_mem, soma=soma, basal=basal, apical=apical,
                    prior_mem=prior_mem,
                    steps=action.new_full((action.shape[0], 1), self.c.core_steps))

    def obs_step(self, state, action, embed, is_first, sample=True, reset=None):
        action = action * (1.0 / action.abs().clamp_min(1.0)).detach()
        if reset is None:
            reset = bool(torch.sum(is_first) > 0)
        if reset:
            initial = self.initial(action.shape[0])
            first = is_first.to(action.dtype)
            action = action * (1.0 - first[:, None])
            state = {k: v * (1.0 - first.reshape(
                         (-1,) + (1,) * (v.ndim - 1))) +
                         initial[k] * first.reshape((-1,) + (1,) * (v.ndim - 1))
                     for k, v in state.items()}
        prior = self.img_step(state, action, sample)
        inputs = torch.cat((prior["deter"].movedim(-2, 0), embed), -1)
        membrane = action.new_zeros(action.shape[0], self.c.output_layers, self.c.hidden)
        outputs = []
        for step in range(self.c.core_steps):
            x, membrane = self.lif_step(self.raw._obs_out_layers, inputs[step], membrane)
            outputs.append(x)
        stats = self.raw._suff_stats_layer("obs", sum(outputs) / self.c.core_steps)
        dist = self.raw.get_dist(stats)
        post = {**prior, **stats, "stoch": dist.sample() if sample else dist.mode()}
        return post, prior


class StatefulSlowMemoryRSSM(StatefulFixedRSSM):
    """Fast spiking soma with independently retained, learnable slow dendrites.

    The inherited trajectory state, fixed-T spike-index recurrence, posterior,
    prior and peripheral readouts are unchanged. Only the dendritic integration
    and spike-triggered reset rule differ from StatefulFixedRSSM.
    """
    def __init__(self, c, actions, embed):
        super().__init__(c, actions, embed)
        keep = math.exp(-math.log(2.0) / c.dendrite_half_life_init)
        logit = math.log(keep) - math.log1p(-keep)
        # Constant initialization consumes no RNG, preserving peripheral init.
        self.basal_retention_logit = nn.Parameter(torch.full((c.deter,), logit))
        self.apical_retention_logit = nn.Parameter(torch.full((c.deter,), logit))

    def dendrite_write_fraction(self):
        # Learn retention per environment decision, then distribute it over T.
        # expm1 avoids cancellation for near-one retention. Each update remains
        # a convex interpolation; neither dendrite is cleared by soma firing.
        return tuple(-torch.expm1(F.logsigmoid(p) / self.c.core_steps)
                     for p in (self.basal_retention_logit, self.apical_retention_logit))

    def retention_metrics(self):
        basal = self.basal_retention_logit.detach().sigmoid()
        apical = self.apical_retention_logit.detach().sigmoid()
        return dict(basal_keep_env_mean=basal.mean(), basal_keep_env_min=basal.min(),
                    basal_keep_env_max=basal.max(), apical_keep_env_mean=apical.mean(),
                    apical_keep_env_min=apical.min(), apical_keep_env_max=apical.max())

    def core_step(self, x, previous_spike, soma, basal, apical):
        cell = self.raw._cell
        neuron = cell.neuron
        combined = torch.cat((x, previous_spike), -1)
        apical_input = cell.apical_norm(cell.apcial_w(combined))
        basal_input = cell.basal_norm(cell.basal_w(combined))
        soma_input = cell.soma_norm(cell.soma_w(x))
        write_basal, write_apical = self.dendrite_write_fraction()
        basal = basal + write_basal * (basal_input - basal)
        apical = apical + write_apical * (apical_input - apical)
        # Retain the released soma equation, apical gate and surrogate function.
        gate = F.sigmoid(apical)
        drive = basal - soma + soma_input
        soma = soma + gate * (drive - soma) / neuron.tau
        spike = neuron.act_fun(soma - neuron.threshold)
        soma = soma * (1.0 - spike.detach())
        # Episode boundaries still reset all state via the inherited obs_step.
        return spike, soma, basal, apical



class StatefulGatedMemoryRSSM(StatefulFixedRSSM):
    """Input-dependent dendritic retention with the original spiking readout."""

    def __init__(self, c, actions, embed):
        super().__init__(c, actions, embed)
        keep = math.exp(-math.log(2.0) / c.dendrite_half_life_init)
        logit = math.log(keep) - math.log1p(-keep)
        # Start from the previous slow-memory dynamics without shifting the RNG
        # used by peripheral networks. The weights learn input selectivity.
        with torch.random.fork_rng(devices=[]):
            self.memory_gates = nn.Linear(c.hidden + c.deter, 2 * c.deter)
            nn.init.zeros_(self.memory_gates.weight)
            nn.init.constant_(self.memory_gates.bias, logit)

    def initial(self, batch):
        state = super().initial(batch)
        # Detached telemetry only; never consumed by dynamics or feature heads.
        state["gate_stats"] = self.raw.W.new_zeros(batch, 6)
        return state

    def dendrite_write_fraction(self, combined):
        # Gate logits describe per-decision retention. Distribute each gate
        # over T microsteps; a changing gate has no single fixed half-life.
        log_keep = F.logsigmoid(self.memory_gates(combined)) / self.c.core_steps
        return -torch.expm1(log_keep), log_keep

    def core_step(self, x, previous_spike, soma, basal, apical, write_fraction=None):
        cell = self.raw._cell
        neuron = cell.neuron
        combined = torch.cat((x, previous_spike), -1)
        apical_input = cell.apical_norm(cell.apcial_w(combined))
        basal_input = cell.basal_norm(cell.basal_w(combined))
        soma_input = cell.soma_norm(cell.soma_w(x))
        if write_fraction is None:
            write_fraction, _ = self.dendrite_write_fraction(combined)
        write_basal, write_apical = write_fraction.chunk(2, -1)
        basal = basal + write_basal * (basal_input - basal)
        apical = apical + write_apical * (apical_input - apical)
        gate = F.sigmoid(apical)
        drive = basal - soma + soma_input
        soma = soma + gate * (drive - soma) / neuron.tau
        spike = neuron.act_fun(soma - neuron.threshold)
        soma = soma * (1.0 - spike.detach())
        return spike, soma, basal, apical

    def img_step(self, state, action, sample=True):
        action = action * (1.0 / action.abs().clamp_min(1.0)).detach()
        inputs = torch.cat((state["stoch"].flatten(-2), action), -1)
        input_mem = state["input_mem"]
        input_spikes = []
        for _ in range(self.c.core_steps):
            x, input_mem = self.lif_step(self.raw._inp_layers, inputs, input_mem)
            input_spikes.append(x)
        # These gates depend only on precomputed input spikes and the previous
        # environment step, so evaluate all T gates in one batched projection.
        combined = torch.cat((torch.stack(input_spikes, 1), state["deter"]), -1)
        writes, log_keep = self.dendrite_write_fraction(combined)
        soma, basal, apical = (state[k] for k in ("soma", "basal", "apical"))
        spikes = []
        for step, x in enumerate(input_spikes):
            spike, soma, basal, apical = self.core_step(
                x, state["deter"][:, step], soma, basal, apical, writes[:, step])
            spikes.append(spike)
        stats, prior_mem = self.prior_readout(spikes, state["prior_mem"])
        dist = self.raw.get_dist(stats)
        # Product of the T direct carry factors, not a full-network memory
        # horizon. Keep only six detached summaries per trajectory step.
        keep = log_keep.detach().sum(1).exp().reshape(action.shape[0], 2, self.c.deter)
        gate_stats = torch.stack((keep.mean(-1), keep.amin(-1), keep.amax(-1)), -1).flatten(-2)
        return dict(**stats, stoch=dist.sample() if sample else dist.mode(),
                    deter=torch.stack(spikes, 0).movedim(0, -2),
                    input_mem=input_mem, soma=soma, basal=basal, apical=apical,
                    prior_mem=prior_mem, gate_stats=gate_stats,
                    steps=action.new_full((action.shape[0], 1), self.c.core_steps))

    def retention_metrics(self, post):
        stats = post["gate_stats"].detach()
        basal_weights, apical_weights = self.memory_gates.weight.detach().chunk(2, 0)
        return dict(basal_keep_env_mean=stats[..., 0].mean(),
                    basal_keep_env_min=stats[..., 1].amin(),
                    basal_keep_env_max=stats[..., 2].amax(),
                    apical_keep_env_mean=stats[..., 3].mean(),
                    apical_keep_env_min=stats[..., 4].amin(),
                    apical_keep_env_max=stats[..., 5].amax(),
                    basal_gate_weight_rms=basal_weights.square().mean().sqrt(),
                    apical_gate_weight_rms=apical_weights.square().mean().sqrt())


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

    def obs_step(self, state, action, embed, is_first, sample=True, reset=None):
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



class ANNGRURSSM(RSSMBase):
    """Layer-normalized Dreamer GRU with ANN input/prior/posterior layers."""
    def __init__(self, c, actions, embed):
        super().__init__(c)
        from .vendor.ann_networks import RSSM
        self.raw = RSSM(stoch=c.stoch, deter=c.deter, hidden=c.hidden,
                        rec_depth=1, discrete=c.classes, act="SiLU", norm=True,
                        unimix_ratio=c.unimix, initial="learned",
                        num_actions=actions, embed=embed, device=c.device)

    def initial(self, batch):
        return self.raw.initial(batch)

    def get_feat(self, state):
        return self.raw.get_feat(state).unsqueeze(0)

    def img_step(self, state, action, sample=True):
        action = action * (1.0 / action.abs().clamp_min(1.0)).detach()
        return self.raw.img_step(state, action, sample=sample)

    def obs_step(self, state, action, embed, is_first, sample=True, reset=None):
        if reset is None:
            reset = bool(is_first.any())
        if reset:
            state = self.reset_state(state, is_first)
            action = action * (1 - is_first.to(action.dtype)[:, None])
        prior = self.img_step(state, action, sample)
        x = self.raw._obs_out_layers(torch.cat((prior["deter"], embed[0]), -1))
        stats = self.raw._suff_stats_layer("obs", x)
        return dict(**stats, stoch=self.draw(stats["logit"], sample),
                    deter=prior["deter"]), prior


class LIFRecurrentCell(nn.Module):
    """One-compartment current-driven LIF; no MCN gates or dendrites."""
    def __init__(self, c):
        super().__init__()
        from .vendor.normalization import PopNorm
        from .vendor.surrogate import QGateGrad
        self.projection = nn.Linear(c.hidden + c.deter, c.deter)
        self.norm = PopNorm(c.deter, threshold=c.threshold, v_reset=0.0, eps=1e-3)
        self.threshold = nn.Parameter(torch.tensor(float(c.threshold)))
        self.tau = c.tau
        self.fire = QGateGrad(alpha=2.0, requires_grad=False)
        self.apply(tools.weight_init)

    def forward(self, x, previous_spike, membrane):
        current = self.norm(self.projection(torch.cat((x, previous_spike), -1)))
        voltage = membrane + (current - membrane) / self.tau
        spike = self.fire(voltage - self.threshold)
        return spike, voltage * (1.0 - spike.detach())


class LIFFixedRSSM(StatefulFixedRSSM):
    """Fixed-T8 LIF RSSM with trajectory-owned single-compartment memory.

    Input/prior LIF states and recurrent voltage carry across environment
    transitions and imagination. Posterior and peripheral LIFs reset per call.
    Internal index recurrence follows the frozen fixed-T MCN comparison.
    """
    memory_keys = ("input_mem", "membrane", "prior_mem")

    def __init__(self, c, actions, embed):
        super().__init__(c, actions, embed)
        # Replace the complete MCN module, so no unused MCN parameters survive.
        self.raw._cell = LIFRecurrentCell(c)

    def neuron_initial(self, batch):
        zeros = self.raw.W.new_zeros
        return dict(input_mem=zeros(batch, self.c.input_layers, self.c.hidden),
                    membrane=zeros(batch, self.c.deter),
                    prior_mem=zeros(batch, self.c.output_layers, self.c.hidden))

    def initial(self, batch):
        memory = self.neuron_initial(batch)
        deter = torch.tanh(self.raw.W).repeat(self.c.core_steps, batch, 1)
        stats, _ = self.prior_readout(deter, memory["prior_mem"])
        return dict(logit=self.raw.W.new_zeros(batch, self.c.stoch, self.c.classes),
                    stoch=self.raw.get_dist(stats).mode(),
                    deter=deter.movedim(0, -2), **memory,
                    steps=self.raw.W.new_zeros(batch, 1))

    def img_step(self, state, action, sample=True):
        memory = state if self.c.persistent else self.neuron_initial(action.shape[0])
        action = action * (1.0 / action.abs().clamp_min(1.0)).detach()
        inputs = torch.cat((state["stoch"].flatten(-2), action), -1)
        input_mem = memory["input_mem"]
        input_spikes = []
        for _ in range(self.c.core_steps):
            x, input_mem = self.lif_step(self.raw._inp_layers, inputs, input_mem)
            input_spikes.append(x)
        membrane = memory["membrane"]
        spikes = []
        for step, x in enumerate(input_spikes):
            spike, membrane = self.raw._cell(x, state["deter"][:, step], membrane)
            spikes.append(spike)
        stats, prior_mem = self.prior_readout(spikes, memory["prior_mem"])
        return dict(**stats, stoch=self.draw(stats["logit"], sample),
                    deter=torch.stack(spikes, 0).movedim(0, -2),
                    input_mem=input_mem, membrane=membrane, prior_mem=prior_mem,
                    steps=action.new_full((action.shape[0], 1), self.c.core_steps))


def make_rssm(c, actions, embed):
    if c.method == "ann_gru":
        return ANNGRURSSM(c, actions, embed)
    if c.method == "lif":
        return LIFFixedRSSM(c, actions, embed)
    if c.method == "legacy":
        return LegacyRSSM(c, actions, embed)
    if c.method == "stateful":
        return StatefulFixedRSSM(c, actions, embed)
    if c.method == "stateful_slowmem":
        return StatefulSlowMemoryRSSM(c, actions, embed)
    if c.method == "stateful_gatedmem":
        return StatefulGatedMemoryRSSM(c, actions, embed)
    return TimeAlignedRSSM(c, actions, embed)
