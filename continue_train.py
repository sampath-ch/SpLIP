#!/usr/bin/env python
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


def adaptive_triplet_loss(s_feats, p_feats, labels, class_feats):
    neg_feats = p_feats.roll(1, dims=0)
    mu = F.cosine_similarity(
        class_feats[labels],
        class_feats[labels.roll(1)],
        dim=-1
    )
    d_pos = (s_feats - p_feats).pow(2).sum(1)
    d_neg = (s_feats - neg_feats).pow(2).sum(1)
    return F.relu(d_pos - d_neg + mu).mean()


def compute_map_at_k(model, ds, preprocess, device, k=200, batch_size=64):
    gallery_paths, gallery_labels = [], []
    for cls in ds.classes:
        idx = ds.class_to_idx[cls]
        for p in ds.images[cls]:
            gallery_paths.append(p)
            gallery_labels.append(idx)

    sketch_paths, sketch_labels = [], []
    for cls in ds.classes:
        idx = ds.class_to_idx[cls]
        for p in ds.sketches[cls]:
            sketch_paths.append(p)
            sketch_labels.append(idx)

    model.eval()
    gallery_feats = []
    with torch.no_grad():
        for i in range(0, len(gallery_paths), batch_size):
            batch = gallery_paths[i : i + batch_size]
            imgs = [preprocess(Image.open(p).convert("RGB")) for p in batch]
            x = torch.stack(imgs).to(device)
            feats = model.encode_branch(x, prompt_type="photo")
            gallery_feats.append(F.normalize(feats, dim=1))
    gallery_feats = torch.cat(gallery_feats, dim=0)
    gallery_labels = torch.tensor(gallery_labels, device=device)

    sketch_feats = []
    with torch.no_grad():
        for i in range(0, len(sketch_paths), batch_size):
            batch = sketch_paths[i : i + batch_size]
            sks = [preprocess(Image.open(p).convert("RGB")) for p in batch]
            x = torch.stack(sks).to(device)
            feats = model.encode_branch(x, prompt_type="sketch")
            sketch_feats.append(F.normalize(feats, dim=1))
    sketch_feats = torch.cat(sketch_feats, dim=0)
    sketch_labels = torch.tensor(sketch_labels, device=device)

    aps = []
    for q_feat, q_label in zip(sketch_feats, sketch_labels):
        sims = gallery_feats @ q_feat
        topk = sims.topk(k).indices
        hits = (gallery_labels[topk] == q_label).float()
        if hits.sum() == 0:
            aps.append(0.0)
            continue
        cum_hits = hits.cumsum(dim=0)
        precision_at_i = cum_hits / torch.arange(1, k+1, device=device).float()
        ap = (precision_at_i * hits).sum() / hits.sum()
        aps.append(ap.item())
    return float(sum(aps) / len(aps))


def main():
    parser = argparse.ArgumentParser("Continue SpLIP-CLIPRepo Training")
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint (raw or wrapped) to resume from')
    parser.add_argument('--add-epochs', type=int, default=2,
                        help='Number of additional epochs to train')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    # Model setup
    model = SpLIP_CLIPRepo(base_model="ViT-B/32", m=4, n=2, device=device).to(device)
    scaler = torch.cuda.amp.GradScaler()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3
    )

    # Optional DDP
    if torch.cuda.device_count() > 1:
        torch.distributed.init_process_group(backend="nccl")
        model = DDP(model, device_ids=[torch.cuda.current_device()])

    # DataLoader
    DATA_ROOT = r"C:\1WorkHere------------\CS747\ZSE-SBIR\datasets\Sketchy\256x256"
    _, preprocess = clip.load("ViT-B/32", device="cpu", jit=False)
    ds = SketchPhotoDataset(DATA_ROOT, transform=preprocess)
    loader = DataLoader(
        ds, batch_size=128, shuffle=True,
        num_workers=8, pin_memory=False, persistent_workers=True
    )

    # Class prompts
    if hasattr(model, "module"):
        class_feats = model.module.encode_class_prompts(ds.classes)
    else:
        class_feats = model.encode_class_prompts(ds.classes)
    tau, alpha = 0.07, 1.0

    # Resume logic
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        print(f"=> loading checkpoint '{args.resume}'")
        ckpt = torch.load(args.resume, map_location=device)
        model_to_load = model.module if hasattr(model, "module") else model
        # Wrapped checkpoint
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model_to_load.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt.get('optimizer_state_dict', {}))
            scaler.load_state_dict(ckpt.get('scaler_state_dict', {}))
            start_epoch = ckpt.get('epoch', -1) + 1
            print(f"=> resuming from epoch {start_epoch}")
        else:
            # Raw state_dict
            model_to_load.load_state_dict(ckpt)
            start_epoch = 0
            print("=> loaded raw model weights, starting from epoch 0")
    else:
        print("=> no checkpoint provided, training from scratch")

    # Training loop
    for epoch in trange(start_epoch, start_epoch + args.add_epochs, desc="Epochs"):
        running_loss = 0.0
        optimizer.zero_grad()
        for step, (sketches, images, labels) in enumerate(tqdm(loader, leave=False), start=1):
            sketches = sketches.to(device, non_blocking=True)
            images   = images.to(device, non_blocking=True)
            labels   = labels.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                img_feats, skt_feats = model(images, sketches)
                l_trip = adaptive_triplet_loss(skt_feats, img_feats, labels, class_feats)
                logits = (img_feats @ class_feats.t()) / tau
                l_cls  = F.cross_entropy(logits, labels)
                loss   = l_trip + alpha * l_cls

            scaler.scale(loss).backward()
            if step % 2 == 0 or step == len(loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            running_loss += loss.item()

        print(f"Epoch {epoch+1:02d} — Avg Loss: {running_loss/len(loader):.4f}")

        # Save checkpoint
        os.makedirs("checkpoints", exist_ok=True)
        ckpt_path = f"checkpoints/splip_cliprepo_epoch{epoch}.pth"
        model_to_save = model.module if hasattr(model, "module") else model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
        }, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

    # Final evaluation
    model_to_eval = model.module if hasattr(model, "module") else model
    map200 = compute_map_at_k(model_to_eval, ds, preprocess, device, k=200, batch_size=64)
    print(f"mAP@200 on Sketchy: {map200:.4f}")


if __name__ == "__main__":
    freeze_support()
    main()
