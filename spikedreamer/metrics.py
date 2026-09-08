"""Instrumentation runs separately from latency measurements."""

import threading
import time
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from .vendor.surrogate import QGateGrad


class OperationCounter:
    """Forward operator ledger; dense GPU work never receives a sparsity discount."""
    def __init__(self, module):
        self.module, self.handles = module, []
        self.rows = defaultdict(lambda: defaultdict(float))

    def __enter__(self):
        for name, layer in self.module.named_modules():
            if isinstance(layer, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d, nn.GRUCell, QGateGrad)):
                self.handles.append(layer.register_forward_hook(
                    lambda m, args, output, name=name: self.record(name, m, args, output)))
        return self

    def record(self, name, layer, args, output):
        row = self.rows[name]
        row["calls"] += 1
        x = args[0].detach()
        if isinstance(layer, QGateGrad):
            row["neuron_updates"] += output.numel()
            row["spikes"] += float(output.detach().sum())
            return
        if isinstance(layer, nn.GRUCell):
            # The fused cell has no nn.Linear children for hooks to discover.
            macs = x.numel() * 3 * layer.hidden_size + output.numel() * 3 * layer.hidden_size
            row["dense_macs"] += float(macs)
            row["analog_input_dense_macs"] += float(macs)
            return
        if isinstance(layer, nn.Linear):
            macs = x.numel() * layer.out_features
        elif isinstance(layer, nn.ConvTranspose2d):
            # Dense algorithm upper bound; kernel implementations may avoid padded work.
            macs = x.numel() * (layer.out_channels // layer.groups) * np.prod(layer.kernel_size)
        else:
            macs = output.numel() * (layer.in_channels // layer.groups) * np.prod(layer.kernel_size)
        row["dense_macs"] += float(macs)
        binary = bool(((x == 0) | (x == 1)).all())
        if binary:
            row["binary_input_dense_macs"] += float(macs)
            if isinstance(layer, nn.Linear):
                row["linear_binary_synops_proxy"] += float(x.sum()) * layer.out_features
            else:
                row["conv_binary_synops_estimate"] += float(x.mean()) * float(macs)
        else:
            row["analog_input_dense_macs"] += float(macs)

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()

    def result(self):
        total = defaultdict(float)
        for row in self.rows.values():
            for k, v in row.items():
                total[k] += v
        return dict(
            totals=dict(total), layers={k: dict(v) for k, v in self.rows.items()},
            scope="forward Linear/Conv/ConvTranspose/GRU projection MACs and QGate neuron updates; not total FLOPs",
            proxy_note="Binary SynOps assume event-driven execution. Conv counts use mean activity; "
                       "no sparsity discount is applied to dense GPU MACs.")


def energy_proxy(ledger, mac_pj, ac_pj, neuron_pj, source):
    """User-specified arithmetic model, separate from measured GPU device energy."""
    if min(mac_pj, ac_pj, neuron_pj) < 0:
        raise ValueError("Energy coefficients cannot be negative")
    counts = ledger["totals"]
    synops = (counts.get("linear_binary_synops_proxy", 0) +
              counts.get("conv_binary_synops_estimate", 0))
    joules = 1e-12 * (counts.get("analog_input_dense_macs", 0) * mac_pj +
                     synops * ac_pj + counts.get("neuron_updates", 0) * neuron_pj)
    return dict(estimated_arithmetic_joules=joules, coefficient_source=source,
                mac_pj=mac_pj, ac_pj=ac_pj, neuron_update_pj=neuron_pj,
                note="Hypothetical event-driven arithmetic only. Excludes memory traffic, "
                     "normalization, probability operations and other uncounted operators; "
                     "not GPU or neuromorphic-device measured energy.")


class EnergyMeter:
    def __init__(self, device):
        self.error, self.handle, self.nvml = None, None, None
        self.samples, self.stop = [], threading.Event()
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            uuid = str(torch.cuda.get_device_properties(device).uuid)
            # PyTorch exposes the bare UUID; NVML requires the GPU- prefix.
            # Resolve by identity, not index, because CUDA_VISIBLE_DEVICES remaps it.
            if not uuid.startswith(("GPU-", "MIG-")):
                uuid = "GPU-" + uuid
            self.handle = pynvml.nvmlDeviceGetHandleByUUID(uuid)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def counter(self):
        try:
            return self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle) / 1000
        except Exception:
            return None

    def sample_power(self):
        while not self.stop.is_set():
            try:
                self.samples.append((time.perf_counter(),
                                     self.nvml.nvmlDeviceGetPowerUsage(self.handle) / 1000))
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                return
            self.stop.wait(0.02)

    def __enter__(self):
        self.start_time = time.perf_counter()
        self.start_energy = self.counter() if self.handle else None
        self.thread = None
        if self.handle:
            self.thread = threading.Thread(target=self.sample_power, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, *args):
        self.end_energy = self.counter() if self.handle else None
        self.end_time = time.perf_counter()
        self.stop.set()
        if self.thread:
            self.thread.join()
        if self.nvml:
            self.nvml.nvmlShutdown()

    def result(self, repeats):
        if self.start_energy is not None and self.end_energy is not None:
            joules, source = self.end_energy - self.start_energy, "nvml_total_energy_counter"
        elif len(self.samples) >= 2:
            stamps, powers = np.asarray(self.samples).T
            # Extend edge samples to the measurement boundaries.
            joules = float(np.trapz(
                np.r_[powers[0], powers, powers[-1]],
                np.r_[self.start_time, stamps, self.end_time]))
            source = "nvml_sampled_power_integral"
        else:
            joules, source = None, "unavailable"
        return dict(raw_joules=joules,
                    raw_joules_per_call=None if joules is None else joules / repeats,
                    source=source, sampling_error=self.error,
                    note="GPU device energy, including idle draw; requires an otherwise idle GPU.")


def measure(fn, device, seconds=10.0, minimum=5):
    for _ in range(3):
        fn()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    wall, gpu = [], []
    start = time.perf_counter()
    with EnergyMeter(device) as energy:
        while len(wall) < minimum or time.perf_counter() - start < seconds:
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t = time.perf_counter()
            a.record()
            fn()
            b.record()
            torch.cuda.synchronize(device)
            wall.append((time.perf_counter() - t) * 1000)
            gpu.append(a.elapsed_time(b))
    return dict(iterations=len(wall), wall_p50_ms=float(np.median(wall)),
                wall_p95_ms=float(np.percentile(wall, 95)), wall_mean_ms=float(np.mean(wall)),
                gpu_p50_ms=float(np.median(gpu)),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                **energy.result(len(wall)))
