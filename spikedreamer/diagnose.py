"""Quantitative open-loop predictions on real replay observations/actions."""

import argparse
import json
from pathlib import Path
import torch
import imageio.v2 as imageio
import numpy as np
from .evaluate import load_agent
from .profile import checkpoint_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--context", type=int, default=5)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 15, 50])
    parser.add_argument("--output", required=True)
    parser.add_argument("--video")
    args = parser.parse_args()
    agent, c, saved = load_agent(args.checkpoint, args.device)
    if args.context < 1 or min(args.horizons) < 1:
        parser.error("Context and horizons must be positive")
    if args.context + max(args.horizons) > c.batch_length:
        parser.error("Requested context+horizon exceeds the saved sequence length")
    batch = checkpoint_batch(args.checkpoint, c, saved)
    results = {}
    with torch.no_grad(), agent.autocast():
        data = agent.wm.preprocess(batch)
        embed = agent.wm.encoder(data)
        state, _ = agent.wm.dynamics.observe(
            embed[:, :, :args.context], data["action"][:, :args.context],
            data["is_first"][:, :args.context], sample=False)
        start = {k: v[:, -1] for k, v in state.items()}
        priors = agent.wm.dynamics.imagine(
            data["action"][:, args.context:args.context + max(args.horizons)], start, sample=False)
        feat = agent.wm.dynamics.get_feat(priors)
        predicted_image = agent.wm.decoder(feat)["image"].mode()
        reward = agent.wm.reward(feat).mode().squeeze(-1)
        cont = agent.wm.cont(feat).mean.squeeze(-1)
        # Never score open-loop predictions across a recorded episode reset.
        valid = ~data["is_first"][:, args.context:args.context + max(args.horizons)].bool()
        valid = valid.cumprod(1).bool()
        for horizon in args.horizons:
            index = args.context + horizon - 1
            keep = valid[:, horizon - 1]
            if not keep.any():
                results[str(horizon)] = dict(valid_sequences=0)
                continue
            results[str(horizon)] = dict(
                valid_sequences=int(keep.sum()),
                image_mse=float((predicted_image[keep, horizon-1] - data["image"][keep, index]).square().mean()),
                reward_mse=float((reward[keep, horizon-1] - data["reward"][keep, index]).square().mean()),
                continue_mse=float((cont[keep, horizon-1] - (1-data["is_terminal"][keep, index])).square().mean()))
        if args.video:
            # Show one contiguous sequence; the montage is prediction vs truth.
            row = int(valid.sum(1).argmax())
            length = int(valid[row].sum())
            if length:
                truth = data["image"][row, args.context:args.context+length]
                prediction = predicted_image[row, :length]
                video = torch.cat([truth, prediction], 2).add(0.5).clamp(0, 1)
                imageio.mimsave(args.video, (video.cpu().numpy() * 255).astype(np.uint8), fps=15)
    payload = dict(checkpoint_frames=saved["frames"], horizons=results,
                   note="In-replay diagnostics, not held-out generalization results.")
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
