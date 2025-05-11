#!/usr/bin/env python
"""Re‑worked SpLIP training script.

Key changes vs. the original `train.py`
--------------------------------------
* **Jigsaw consistency loss** – adds the same patch‑level jigsaw loss used in
  `debug_gradient_flow.py`, which was proven to back‑propagate clean gradients
  to the prompt‑mappers.
* **Gradient flow fixes** – we always request patch tensors from
  `encode_branch(..., return_patches=True)` and cast them to `float()` (this
  mirrors the unit‑test behaviour that kept `requires_grad=True`).
* **Single optimizer** – now optimises *both* SpLIP and the `JigsawSolver`
  parameters, ensuring joint training.
* **Checkpointing & resume** – solver state is persisted / reloaded.
* **Config** – new hyper‑param `--lambda-jig` to weight the jigsaw loss; default
  `0.5` works well in practice.
"""

import os
import argparse
from multiprocessing import freeze_support

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from PIL import Image

import clip
from model_wrapper import SpLIP_CLIPRepo
from dataset import SketchPhotoDataset
from torch.nn.parallel import DistributedDataParallel as DDP

# NEW: jigsaw‑loss bits -------------------------------------------------------
from jigsaw import JigsawSolver, compute_jigsaw_loss
# -----------------------------------------------------------------------------


def adaptive_triplet_loss(s_feats, p_feats, labels, class_feats):
    """Adaptive triplet loss exactly as before (unchanged)."""
    neg_feats = p_feats.roll(1, dims=0)
    mu = F.cosine_similarity(class_feats[labels],
                             class_feats[labels.roll(1)],
                             dim=-1)
    d_pos = (s_feats - p_feats).pow(2).sum(1)
    d_neg = (s_feats - neg_feats).pow(2).sum(1)
    loss = F.relu(d_pos - d_neg + mu).mean()

    # NaN/Inf guard (leave debug prints – they help!)
    if torch.isnan(loss) or torch.isinf(loss):
        print("WARNING: Triplet loss is NaN or Inf")
        print(f"d_pos: {d_pos}")
        print(f"d_neg: {d_neg}")
        print(f"mu   : {mu}")
        return torch.tensor(0.0, device=loss.device, requires_grad=True)

    return loss


# ----------------------------------------------------------------------------
# Training script
# ----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser("Continue SpLIP‑CLIPRepo Training (with Jigsaw)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint (raw or wrapped) to resume from")
    parser.add_argument("--add-epochs", type=int, default=2,
                        help="Number of *additional* epochs to train")
    parser.add_argument("--lambda-jig", type=float, default=0.5,
                        help="Weight for jigsaw loss term")
    parser.add_argument("--debug", action="store_true",
                        help="Disable AMP + print every gradient for sanity")
    args = parser.parse_args()

    # ---------------------------------------------------------------------
    # Device & misc
    # ---------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    # ---------------------------------------------------------------------
    # Model, Jigsaw solver, optimiser, AMP scaler
    # ---------------------------------------------------------------------
    model = SpLIP_CLIPRepo(base_model="ViT-B/32", m=4, n=2, device=device).to(device)
    model.train()

    # Build a JigsawSolver **whose dimensions exactly match the model**
    dv = model.clip_model.visual.conv1.weight.shape[0]  # visual patch‑dim
    n_patches = model.Bvt.n                            #  number of patches
    solver = JigsawSolver(dv=dv, num_patches=n_patches).to(device)
    solver.train()

    # AMP scaler (still useful even with joint params)
    scaler = torch.cuda.amp.GradScaler(enabled=not args.debug)

    # Optimiser – joint params (model + solver)
    optim_params = (list(filter(lambda p: p.requires_grad, model.parameters())) +
                    list(solver.parameters()))
    optimizer = torch.optim.AdamW(optim_params, lr=1e-3)

    # ---------------------------------------------------------------------
    # Optional DDP (multi‑GPU)
    # ---------------------------------------------------------------------
    if torch.cuda.device_count() > 1:
        torch.distributed.init_process_group(backend="nccl")
        model  = DDP(model,  device_ids=[torch.cuda.current_device()])
        solver = DDP(solver, device_ids=[torch.cuda.current_device()])

    # Convenience to get underlying nn.Module when DDP‑wrapped
    def unwrap(m):
        return m.module if hasattr(m, "module") else m

    model_core  = unwrap(model)
    solver_core = unwrap(solver)

    # ---------------------------------------------------------------------
    # Data
    # ---------------------------------------------------------------------
    DATA_ROOT = r"C:\1WorkHere------------\CS747\ZSE-SBIR\datasets\Sketchy\256x256"
    _, preprocess = clip.load("ViT-B/32", device="cpu", jit=False)
    ds = SketchPhotoDataset(DATA_ROOT, transform=preprocess)
    loader = DataLoader(ds, batch_size=64, shuffle=True,
                        num_workers=8, pin_memory=True, persistent_workers=True)

    # ---------------------------------------------------------------------
    # Resume (if any)
    # ---------------------------------------------------------------------
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        print(f"=> loading checkpoint '{args.resume}'")
        ckpt = torch.load(args.resume, map_location=device)

        # Distinguish between wrapped / raw
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model_core.load_state_dict(ckpt["model_state_dict"])
            solver_core.load_state_dict(ckpt.get("solver_state_dict", {}))
            optimizer.load_state_dict(ckpt.get("optimizer_state_dict", {}))
            scaler.load_state_dict(ckpt.get("scaler_state_dict", {}))
            start_epoch = ckpt.get("epoch", -1) + 1
            print(f"=> resuming from epoch {start_epoch}")
        else:
            model_core.load_state_dict(ckpt)
            print("=> loaded raw model weights (solver fresh) – starting from epoch 0")

    # ---------------------------------------------------------------------
    # Hyper‑parameters for the combined loss
    # ---------------------------------------------------------------------
    tau   = 0.07   # temperature for CLS‑like loss
    alpha = 1.0    # weight for classification loss
    beta  = args.lambda_jig  # weight for jigsaw loss (cmd‑line param)

    # ---------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------
    for epoch in trange(start_epoch, start_epoch + args.add_epochs, desc="Epochs"):
        running_loss = 0.0

        for step, (sketches, images, labels) in enumerate(tqdm(loader, leave=False), 1):
            optimizer.zero_grad(set_to_none=True)

            # ***** Forward pass (always request patches) *****
            with torch.cuda.amp.autocast(enabled=not args.debug):
                # Encode branches *with* patch tensors (for jigsaw)
                img_feats, photo_patches   = model.encode_branch(images.to(device, non_blocking=True),
                                                                  prompt_type="photo",
                                                                  return_patches=True)
                skt_feats, sketch_patches  = model.encode_branch(sketches.to(device, non_blocking=True),
                                                                  prompt_type="sketch",
                                                                  return_patches=True)

                # Cast to float32 to avoid AMP oddities (mirrors unit‑test)
                sketch_patches = sketch_patches.float()
                photo_patches  = photo_patches.float()

                # Class‑prompt embedding (re‑built every mini‑batch)
                class_feats = model_core.encode_class_prompts(ds.classes)

                # *Losses*
                l_trip = adaptive_triplet_loss(skt_feats, img_feats, labels.to(device), class_feats)
                logits = (img_feats @ class_feats.T) / tau
                l_cls  = F.cross_entropy(logits, labels.to(device))
                l_jig  = compute_jigsaw_loss(sketch_patches, photo_patches, solver)

                loss = l_trip + alpha * l_cls + beta * l_jig

            # ***** Back‑prop *****
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)  # for gradient norms / clipping
            else:
                loss.backward()

            # (Optional) print a few gradient norms for sanity in debug mode
            if args.debug and step % 10 == 0:
                print(f"\n[Epoch {epoch:02d} | Step {step}] gradient norms (prompt mappers):")
                for n, p in model_core.named_parameters():
                    if p.requires_grad and any(k in n for k in ("Bt.mapper", "Bv.mapper", "Bvt.mapper")):
                        g = p.grad
                        print(f"  {n:55s} : {'None' if g is None else f'{g.norm().item():.6f}' }")

            # Optimiser step + scaler update
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1:02d} — Avg Loss: {running_loss / len(loader):.4f}")

        # -----------------------------------------------------------------
        # Checkpoint – include *everything* (model, solver, optimiser, scaler)
        # -----------------------------------------------------------------
        os.makedirs("checkpoints", exist_ok=True)
        ckpt_path = f"checkpoints/splip_cliprepo_epoch{epoch}.pth"
        torch.save({
            "epoch"              : epoch,
            "model_state_dict"   : model_core.state_dict(),
            "solver_state_dict"  : solver_core.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict"  : scaler.state_dict(),
        }, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

    # ---------------------------------------------------------------------
    # Final evaluation (mAP@200) – identical to the old logic
    # ---------------------------------------------------------------------
    model_core.eval()
    map200 = compute_map_at_k(model_core, ds, preprocess, device,
                              k=200, batch_size=64)
    print(f"mAP@200 on Sketchy: {map200:.4f}")


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    freeze_support()
    main()
