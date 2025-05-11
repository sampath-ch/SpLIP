#!/usr/bin/env python
"""
Comprehensive gradient debugging for SpLIP-CLIPRepo model
This script isolates and tests each component to identify gradient flow issues
"""
import torch
import torch.nn.functional as F
from model_wrapper import SpLIP_CLIPRepo


def debug_gradient_flow():
    """Step by step debug of gradient flow through the model"""
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create model with smaller dimensions for faster testing
    model = SpLIP_CLIPRepo(base_model="ViT-B/32", m=4, n=2, device=device).to(device)
    model.train()  # Ensure training mode is on

    # Register hooks to track gradients at critical points
    activation_gradients = {}

    def hook_fn(name):
        def hook(grad):
            activation_gradients[name] = grad.detach().clone()
            # Print stats immediately for debugging
            print(f"HOOK: {name} - grad shape: {grad.shape}, norm: {grad.norm().item():.6f}")
            if torch.isnan(grad).any():
                print(f"WARNING: NaN detected in {name} gradients!")
            if torch.isinf(grad).any():
                print(f"WARNING: Inf detected in {name} gradients!")

        return hook

    # Test 1: Class prompts encoding and gradient flow
    print("\n=== TEST 1: CLASS PROMPTS AND GRADIENTS ===")
    class_names = ["dog", "cat", "horse", "bird", "car"]

    # Track input tensors that will need gradients
    inputs_requiring_grad = []

    # Generate class features
    print("Encoding class prompts...")
    class_feats = model.encode_class_prompts(class_names)
    print(f"Class features shape: {class_feats.shape}")
    print(f"Class features requires_grad: {class_feats.requires_grad}")

    # Set requires_grad explicitly if needed
    if not class_feats.requires_grad:
        print("WARNING: Class features don't have requires_grad=True!")
        print("Explicitly setting requires_grad=True on class features.")
        class_feats.requires_grad_(True)

    # Register hook on class features
    class_feats.register_hook(hook_fn("class_feats"))
    inputs_requiring_grad.append(class_feats)

    # Optimizer for all trainable parameters
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3
    )

    # Create dummy batch
    batch_size = 4
    sketches = torch.randn(batch_size, 3, 224, 224, device=device)
    images = torch.randn(batch_size, 3, 224, 224, device=device)
    labels = torch.randint(0, len(class_names), (batch_size,), device=device)

    # Forward pass with tracking enabled
    print("\nRunning forward pass...")
    with torch.set_grad_enabled(True):
        img_feats, skt_feats = model(images, sketches)
        print(f"Image features shape: {img_feats.shape}, requires_grad: {img_feats.requires_grad}")
        print(f"Sketch features shape: {skt_feats.shape}, requires_grad: {skt_feats.requires_grad}")

        # Register hooks
        img_feats.register_hook(hook_fn("img_feats"))
        skt_feats.register_hook(hook_fn("skt_feats"))

        # Verify we have no detached tensors
        if not img_feats.requires_grad:
            print("ERROR: Image features don't require grad!")
        if not skt_feats.requires_grad:
            print("ERROR: Sketch features don't require grad!")

        # Compute losses with detailed tracking
        print("\nComputing losses...")

        # 1. Triplet loss
        neg_feats = img_feats.roll(1, dims=0)

        # Class features similarity
        mu = F.cosine_similarity(
            class_feats[labels],
            class_feats[labels.roll(1)],
            dim=-1
        )
        mu.register_hook(hook_fn("mu"))

        # Distance computations
        d_pos = (skt_feats - img_feats).pow(2).sum(1)
        d_neg = (skt_feats - neg_feats).pow(2).sum(1)

        d_pos.register_hook(hook_fn("d_pos"))
        d_neg.register_hook(hook_fn("d_neg"))

        l_trip = F.relu(d_pos - d_neg + mu).mean()
        l_trip.register_hook(hook_fn("l_trip"))

        print(f"Triplet loss: {l_trip.item():.6f}")
        print(f"mu values: {mu}")
        print(f"d_pos values: {d_pos}")
        print(f"d_neg values: {d_neg}")

        # 2. Classification loss
        tau = 0.07
        logits = (img_feats @ class_feats.t()) / tau
        logits.register_hook(hook_fn("logits"))

        l_cls = F.cross_entropy(logits, labels)
        l_cls.register_hook(hook_fn("l_cls"))

        print(f"Classification loss: {l_cls.item():.6f}")

        # Combined loss
        alpha = 1.0
        loss = l_trip + alpha * l_cls

        print(f"Combined loss: {loss.item():.6f}")

    # Check parameters before backward
    print("\nParameters before backward:")
    param_before = {}
    for name, param in model.named_parameters():
        if param.requires_grad and any(n in name for n in ("Bt", "Bv", "Bvt")):
            param_before[name] = param.detach().clone()

    # Backward pass
    print("\nRunning backward pass...")
    optimizer.zero_grad()
    loss.backward()

    # Check if any parameter updated
    print("\nChecking parameter updates:")
    for name, param in model.named_parameters():
        if param.requires_grad and any(n in name for n in ("Bt", "Bv", "Bvt")):
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                print(f"  {name:40s} grad norm = {grad_norm:.6f}")

                # Check for NaN or Inf
                if torch.isnan(param.grad).any():
                    print(f"  WARNING: NaN detected in {name} gradients!")
                if torch.isinf(param.grad).any():
                    print(f"  WARNING: Inf detected in {name} gradients!")
            else:
                print(f"  {name:40s} grad = None")

    # Step the optimizer
    optimizer.step()

    # Verify parameters changed
    print("\nVerifying parameter updates:")
    params_updated = False
    for name, param in model.named_parameters():
        if name in param_before:
            diff = (param - param_before[name]).norm().item()
            params_updated = params_updated or diff > 0
            print(f"  {name:40s} changed by {diff:.8f}")

    print(f"\nParameters updated: {params_updated}")

    # Test with a different batch size
    print("\n=== TEST 2: DIFFERENT BATCH SIZE ===")
    batch_size = 8
    sketches = torch.randn(batch_size, 3, 224, 224, device=device)
    images = torch.randn(batch_size, 3, 224, 224, device=device)
    labels = torch.randint(0, len(class_names), (batch_size,), device=device)

    # --- rebuild class_feats on a fresh graph ---
    class_feats = model.encode_class_prompts(class_names)
    class_feats.requires_grad_(True)

    # Reset gradients
    optimizer.zero_grad()

    # Forward pass
    img_feats, skt_feats = model(images, sketches)

    # Loss computation
    neg_feats = img_feats.roll(1, dims=0)
    mu = F.cosine_similarity(
        class_feats[labels],
        class_feats[labels.roll(1)],
        dim=-1
    )
    d_pos = (skt_feats - img_feats).pow(2).sum(1)
    d_neg = (skt_feats - neg_feats).pow(2).sum(1)
    l_trip = F.relu(d_pos - d_neg + mu).mean()

    logits = (img_feats @ class_feats.t()) / tau
    l_cls = F.cross_entropy(logits, labels)

    loss = l_trip + alpha * l_cls
    loss.backward()

    # Check gradients again
    print("\nGradients after second backward pass:")
    has_gradients = False
    for name, param in model.named_parameters():
        if param.requires_grad and any(n in name for n in ("Bt", "Bv", "Bvt")):
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                has_gradients = has_gradients or grad_norm > 0
                print(f"  {name:40s} grad norm = {grad_norm:.6f}")
            else:
                print(f"  {name:40s} grad = None")

    print(f"Has non-zero gradients: {has_gradients}")

    # Test 3: Check model with detach explicitly disabled
    print("\n=== TEST 3: CHECKING CLIP INTERNALS ===")
    # Check if any operation inside model code is detaching tensors

    # Clear gradients
    optimizer.zero_grad()

    # Forward pass with mini batch
    mini_batch = 2
    sketches = torch.randn(mini_batch, 3, 224, 224, device=device)
    images = torch.randn(mini_batch, 3, 224, 224, device=device)
    labels = torch.randint(0, len(class_names), (mini_batch,), device=device)

    # Test each component separately
    img_feats = model.encode_branch(images, prompt_type="photo")
    print(f"Image features requires_grad: {img_feats.requires_grad}")

    skt_feats = model.encode_branch(sketches, prompt_type="sketch")
    print(f"Sketch features requires_grad: {skt_feats.requires_grad}")

    # Loss computation (simplified)
    loss = F.mse_loss(img_feats, skt_feats)
    print(f"Simple MSE loss: {loss.item()}")

    # Backward pass
    loss.backward()

    # Check gradients
    print("\nGradients with simplified loss:")
    has_gradients = False
    for name, param in model.named_parameters():
        if param.requires_grad and any(n in name for n in ("Bt", "Bv", "Bvt")):
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                has_gradients = has_gradients or grad_norm > 0
                print(f"  {name:40s} grad norm = {grad_norm:.6f}")
            else:
                print(f"  {name:40s} grad = None")

    print(f"Has gradients with simplified loss: {has_gradients}")

    return has_gradients


if __name__ == "__main__":
    success = debug_gradient_flow()
    if success:
        print("\n✅ GRADIENT FLOW WORKING: Gradients are flowing correctly!")
    else:
        print("\n❌ GRADIENT FLOW ERROR: Gradients not flowing correctly.")