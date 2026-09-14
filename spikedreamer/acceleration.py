"""Regional compilation of the unchanged fixed-T network computations."""

import torch


def accelerate(agent):
    """Compile the methods actually called, leaving optimizer/guards in eager mode."""
    if agent.c.method not in ("legacy", "stateful", "stateful_slowmem", "stateful_gatedmem", "ann_gru", "lif"):
        raise ValueError("Compiled execution requires a supported fixed-step RSSM")
    # Distribution parameters are constructed by softmax/sigmoid/positive-scale
    # transforms. Keep the trainer's explicit finite loss and gradient guards,
    # but avoid distribution constructors synchronizing on every latent draw.
    torch.distributions.Distribution.set_default_validate_args(False)
    # Shared head code sees training, imagined batches, and online/eval batches.
    torch._dynamo.config.cache_size_limit = 64
    # PyTorch 2.6's far-apart recomputation heuristic enumerates repeated
    # paths through this recurrent graph. Other partitioning guards remain on.
    # This changes compiler activation planning, not the model or gradients.
    from torch._functorch import config as aot_config
    aot_config.ban_recompute_used_far_apart = False
    options = {"triton.cudagraphs": False, "fallback_random": True,
               "compile_threads": 4}
    # Reuse one transition graph across the real sequence and imagination;
    # compiling the entire BPTT loop would instead unroll 64 large copies.
    dynamics = agent.wm.dynamics
    dynamics.obs_step = torch.compile(dynamics.obs_step, fullgraph=True,
                                      dynamic=False, options=options)
    dynamics.img_step = torch.compile(dynamics.img_step, fullgraph=True,
                                      dynamic=False, options=options)
    for module in (agent.wm.encoder, agent.wm.decoder._cnn):
        # Replace the callable, not the module: checkpoint keys stay unchanged.
        module.forward = torch.compile(module.forward, fullgraph=True,
                                       dynamic=False, options=options)
    for module in (agent.wm.reward, agent.wm.cont, agent.actor,
                   agent.value, agent.slow_value):
        # Keep Python distribution objects outside the tensor-only graph.
        if agent.c.method == "ann_gru":
            module.layers.forward = torch.compile(module.layers.forward,
                                                  fullgraph=True, dynamic=False,
                                                  options=options)
        else:
            module._spike_features = torch.compile(module._spike_features,
                                                   fullgraph=True, dynamic=False,
                                                   options=options)
