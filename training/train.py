"""Train the palmprint encoder (Python) and export it for the Rust inference crate.

    python training/train.py data/rois.npz --out rust/models --epochs 40

Pipeline: identity-disjoint split (train / cal / test palms) → ResNet-18 + projection head trained with
supervised-contrastive loss (or the prototype's self-supervised NT-Xent, --loss simclr) under SAM →
model selected by calibration-set EER → verification threshold calibrated on the calibration palms at a
target FAR → test-palm metrics (EER, TAR@FAR, MCC at the threshold, cross-session rank-1) →
ONNX export of the 512-d backbone + model card + parity fixtures that `palmprint verify-model` checks.
"""
import argparse, copy, hashlib, json, math, os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

from common import EMBED_DIM, SIDE, log, split_ids
from metrics import evaluate, far_threshold


# ── model ───────────────────────────────────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    """ResNet-18 backbone (what inference uses: 512-d pooled features) + MLP projection head (training only)."""

    def __init__(self, proj_dim=128, pretrained=True):
        super().__init__()
        r = None
        if pretrained:
            try:
                r = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
            except Exception as e:  # offline: train from scratch
                log("warning", msg=f"ImageNet weights unavailable ({type(e).__name__}); training from scratch")
        r = r or torchvision.models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(r.children())[:-1])
        self.head = nn.Sequential(nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Linear(512, proj_dim))

    def features(self, x):
        return self.backbone(x).flatten(1)

    def forward(self, x):
        return self.head(self.features(x))


class Backbone(nn.Module):
    """Exported graph: input [b,3,138,138] → embedding [b,512]."""

    def __init__(self, enc):
        super().__init__()
        self.backbone = enc.backbone

    def forward(self, x):
        return self.backbone(x).flatten(1)


# ── losses ──────────────────────────────────────────────────────────────────────────────────────────
def supcon_loss(z, labels, t=0.1):
    """Supervised contrastive loss (Khosla et al. 2020): every same-palm view in the batch is a positive."""
    z = F.normalize(z, dim=1)
    sim = z @ z.T / t
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e9)
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos = (labels[:, None] == labels[None, :]) & ~eye
    npos = pos.sum(1)
    keep = npos > 0
    return -((logp * pos).sum(1)[keep] / npos[keep]).mean()


def nt_xent_loss(z, t=0.07):
    """The prototype's self-supervised loss (two views of each image are the only positives)."""
    n = len(z) // 2
    labels = torch.cat([torch.arange(n), torch.arange(n)]).to(z.device)
    return supcon_loss(z, labels, t)


# ── SAM ─────────────────────────────────────────────────────────────────────────────────────────────
class SAM:
    """Sharpness-Aware Minimization (Foret et al. 2021): step to the worst point within an L2 ball of
    radius rho, take the gradient there, return, and let the base optimiser apply it."""

    def __init__(self, params, base, rho=0.05):
        self.params, self.base, self.rho, self.e = [p for p in params if p.requires_grad], base, rho, {}

    @torch.no_grad()
    def first_step(self):
        g = torch.norm(torch.stack([p.grad.norm(2) for p in self.params if p.grad is not None]), 2)
        s = self.rho / (g + 1e-12)
        for p in self.params:
            if p.grad is None:
                continue
            e = p.grad * s
            p.add_(e)
            self.e[p] = e
        self.base.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self):
        for p, e in self.e.items():
            p.sub_(e)
        self.e = {}
        self.base.step()
        self.base.zero_grad(set_to_none=True)


def bn_momentum(model, m):
    """Freeze BatchNorm running statistics during SAM's second (perturbed) forward pass."""
    for mod in model.modules():
        if isinstance(mod, nn.modules.batchnorm._BatchNorm):
            if m is None:
                mod.momentum = getattr(mod, "_m", mod.momentum)
            else:
                mod._m, mod.momentum = mod.momentum, m


# ── data ────────────────────────────────────────────────────────────────────────────────────────────
AUG = transforms.Compose([  # the prototype's augmentations (self_supervised/augmentations.py)
    transforms.RandomResizedCrop(SIDE, scale=(0.8, 1.0), antialias=True),
    transforms.RandomRotation(degrees=15),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.8),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 0.5))], p=0.5),
])


def to_input(u8):
    """uint8 ROIs [n,138,138] → model input [n,3,138,138] exactly as rust/src/encoder.rs builds it."""
    x = torch.from_numpy(u8).float().div_(255.0)[:, None]
    return x.expand(-1, 3, -1, -1).contiguous()


def pk_batches(ids, p, k, rng):
    """Batches of p palms × k images (sampled with replacement when a palm has fewer than k)."""
    by = {}
    for i, v in enumerate(ids):
        by.setdefault(int(v), []).append(i)
    keys = np.array(list(by))
    rng.shuffle(keys)
    for s in range(0, len(keys) - p + 1, p):
        yield np.concatenate([rng.choice(by[c], k, replace=len(by[c]) < k) for c in keys[s:s + p]])


@torch.no_grad()
def embed(model, u8, device, bs=256):
    model.eval()
    out = [model.features(to_input(u8[i:i + bs]).to(device)).cpu().numpy() for i in range(0, len(u8), bs)]
    return np.concatenate(out).astype(np.float32)


# ── main ────────────────────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", help="archive from prepare.py")
    ap.add_argument("--out", default="rust/models")
    ap.add_argument("--name", default="palmprint_encoder_512")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--loss", choices=["supcon", "simclr"], default="supcon")
    ap.add_argument("--p", type=int, default=16, help="palms per batch")
    ap.add_argument("--k", type=int, default=4, help="images per palm per batch")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--rho", type=float, default=0.05, help="SAM radius")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--target-far", type=float, default=1e-3)
    ap.add_argument("--block", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--max-steps", type=int, default=0, help="cap steps per epoch (smoke tests)")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load(a.data)
    rois, names, ids, sessions = d["rois"], d["names"], d["ids"], d["sessions"]
    split = split_ids(ids, a.seed)
    idx = {s: np.flatnonzero(np.isin(ids, split[s])) for s in split}
    log("data", images=len(rois), palms=len(set(ids.tolist())), **{f"{s}_palms": len(split[s]) for s in split},
        **{f"{s}_images": len(idx[s]) for s in split}, device=str(device))

    model = Encoder(pretrained=not a.no_pretrained).to(device)
    base = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sam = SAM(model.parameters(), base, rho=a.rho)
    tr_ids = ids[idx["train"]]
    steps_per_epoch = max(1, len(set(tr_ids.tolist())) // a.p)
    if a.max_steps:
        steps_per_epoch = min(steps_per_epoch, a.max_steps)
    total = steps_per_epoch * a.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(base, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))

    def loss_on(xb, yb):
        v1 = torch.stack([AUG(x) for x in xb])
        v2 = torch.stack([AUG(x) for x in xb])
        z = model(torch.cat([v1, v2]).to(device))
        if a.loss == "supcon":
            return supcon_loss(z, torch.cat([yb, yb]).to(device), a.temperature)
        return nt_xent_loss(z, a.temperature)

    best, best_state, history = None, None, []
    for ep in range(a.epochs):
        model.train()
        tot, n = 0.0, 0
        for step, b in enumerate(pk_batches(tr_ids, a.p, a.k, rng)):
            if step >= steps_per_epoch:
                break
            sel = idx["train"][b]
            xb, yb = to_input(rois[sel]), torch.from_numpy(ids[sel])
            loss = loss_on(xb, yb)
            loss.backward()
            sam.first_step()
            bn_momentum(model, 0.0)
            loss_on(xb, yb).backward()
            bn_momentum(model, None)
            sam.second_step()
            sched.step()
            tot, n = tot + loss.item(), n + 1
        cal = evaluate(embed(model, rois[idx["cal"]], device), ids[idx["cal"]], sessions[idx["cal"]])
        history.append({"epoch": ep + 1, "loss": tot / max(n, 1), "cal_eer": cal["eer"]})
        log("epoch", epoch=ep + 1, of=a.epochs, loss=round(tot / max(n, 1), 4), cal_eer=round(cal["eer"], 4),
            cal_rank1=cal["rank1"])
        if best is None or cal["eer"] < best:
            best, best_state = cal["eer"], copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    e_cal = embed(model, rois[idx["cal"]], device)
    cal = evaluate(e_cal, ids[idx["cal"]], sessions[idx["cal"]])
    thr = far_threshold(cal["impostor"], a.target_far)
    test = evaluate(embed(model, rois[idx["test"]], device), ids[idx["test"]], sessions[idx["test"]], threshold=thr)
    for r in (cal, test):
        r.pop("impostor", None), r.pop("genuine", None)
    log("calibrated", threshold=thr, target_far=a.target_far, test_far=test["at_threshold"]["far"],
        test_tar=test["at_threshold"]["tar"], test_mcc=test["at_threshold"]["mcc"], test_eer=test["eer"], test_rank1=test["rank1"])

    # ── export (the Rust crate loads exactly this) ─────────────────────────────────────────────────
    os.makedirs(a.out, exist_ok=True)
    onnx_path = os.path.join(a.out, f"{a.name}.onnx")
    m = Backbone(model).cpu().eval()
    x = to_input(rois[idx["test"][:8]])
    torch.onnx.export(m, x[:1], onnx_path, input_names=["input"], output_names=["embedding"], opset_version=17,
                      dynamic_axes={"input": {0: "b"}, "embedding": {0: "b"}}, dynamo=False)
    import onnxruntime as ort
    with torch.no_grad():
        ref = m(x).numpy()
    got = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"]).run(None, {"input": x.numpy()})[0]
    np.save(os.path.join(a.out, f"{a.name}.parity_input.npy"), (rois[idx["test"][:8]].astype(np.float32) / 255.0).reshape(8, -1))
    np.save(os.path.join(a.out, f"{a.name}.parity_embedding.npy"), got.astype(np.float32))
    card = {
        "model": f"{a.name}.onnx",
        "onnx_sha256": hashlib.sha256(open(onnx_path, "rb").read()).hexdigest(),
        "input": {"name": "input", "shape": [None, 3, SIDE, SIDE], "values": "preprocessed gray ROI / 255, replicated to 3 channels, no mean/std"},
        "output": {"name": "embedding", "dim": EMBED_DIM, "compare": "cosine"},
        "preprocessing": "preprocess/preprocessor.py (Rust: rust/src/preprocess.rs, bit-exact)",
        "threshold": thr,
        "threshold_rule": f"smallest float32 above the ⌊{a.target_far}·N⌋-th largest calibration impostor score (lean/Match.lean: far_threshold_bound)",
        "target_far": a.target_far,
        "identity_block": a.block,
        "split": split,
        "metrics": {"calibration": cal, "test": test},
        "training": {"loss": a.loss, "optimizer": "AdamW + SAM", "rho": a.rho, "lr": a.lr, "epochs": a.epochs,
                     "p": a.p, "k": a.k, "temperature": a.temperature, "pretrained": not a.no_pretrained,
                     "seed": a.seed, "best_cal_eer": best, "history": history},
        "onnx_vs_pytorch_max_abs": float(np.abs(got - ref).max()),
    }
    with open(os.path.join(a.out, f"{a.name}.card.json"), "w") as f:
        json.dump(card, f, indent=1)
    log("exported", onnx=onnx_path, max_abs_vs_pytorch=card["onnx_vs_pytorch_max_abs"], card=f"{a.name}.card.json")


if __name__ == "__main__":
    main()
