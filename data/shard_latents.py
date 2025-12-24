import io, json
from pathlib import Path

import cv2
import numpy as np
import torch
import webdataset as wds
from tqdm import tqdm
from diffusers import AutoencoderKL


ROOT = Path("./clips")
JSONL = ROOT / "train.jsonl"     # or clips.jsonl
LABELS = json.load(open(ROOT / "labels.json"))

OUTDIR = Path("./shards")
OUTDIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# ---- settings ----
T_TRAIN = 72        # number of frames in one training sample
CROP = 320           # no crop full 320x320 clips
SAMPLES_PER_SHARD = 128
VAE_NAME = "ema"     # "ema" or "mse"  -> stabilityai/sd-vae-ft-ema / sd-vae-ft-mse

# ------------------
vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{VAE_NAME}").to(DEVICE, dtype=DTYPE)
vae.eval()

def read_all_frames(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)  # BGR uint8
    cap.release()
    return frames

def random_window(frames, T, rng):
    n = len(frames)
    if n < T:
        frames = frames + [frames[-1]] * (T - n)
        n = len(frames)
    s = int(rng.integers(0, n - T + 1))
    return frames[s:s+T]

def random_crop(frames, crop, rng):
    h, w = frames[0].shape[:2]
    y0 = int(rng.integers(0, h - crop + 1))
    x0 = int(rng.integers(0, w - crop + 1))
    return [fr[y0:y0+crop, x0:x0+crop] for fr in frames]

def frames_to_vae_input(frames_bgr):
    # -> torch float [T,3,H,W] in [-1,1]
    rgb = [cv2.cvtColor(fr, cv2.COLOR_BGR2RGB) for fr in frames_bgr]
    arr = np.stack(rgb, 0)  # [T,H,W,3] uint8
    x = torch.from_numpy(arr).permute(0,3,1,2).contiguous().float() / 255.0
    x = x * 2.0 - 1.0
    return x

@torch.no_grad()
def encode_latents(x):
    # x: [T,3,256,256] in [-1,1]
    x = x.to(DEVICE, dtype=DTYPE)
    dist = vae.encode(x).latent_dist
    z = dist.sample()  # [T,4,H/8,W/8]
    z = z * vae.config.scaling_factor
    return z.to(torch.float16).cpu()  # store as fp16

rows = [json.loads(l) for l in open(JSONL) if l.strip()]
rng = np.random.default_rng(0)

shard_idx = 0
count_in_shard = 0
writer = wds.TarWriter(str(OUTDIR / f"{shard_idx:06d}.tar"))

for r in tqdm(rows):
    clip_path = (ROOT / r["path"]).as_posix()
    label = r["label"]
    label_id = int(LABELS[label])

    frames = read_all_frames(clip_path)
    frames = random_window(frames, T_TRAIN, rng)
    frames = random_crop(frames, CROP, rng)

    x = frames_to_vae_input(frames)      # [T,3,256,256]
    z = encode_latents(x)                # [T,4,32,32] fp16

    buf = io.BytesIO()
    torch.save({"z": z, "label_id": label_id, "clip_id": r["clip_id"], "label": label}, buf)

    sample = {
        "__key__": r["clip_id"],
        "pt": buf.getvalue(),
    }
    writer.write(sample)

    count_in_shard += 1
    if count_in_shard >= SAMPLES_PER_SHARD:
        writer.close()
        shard_idx += 1
        count_in_shard = 0
        writer = wds.TarWriter(str(OUTDIR / f"{shard_idx:06d}.tar"))

writer.close()
print("done:", shard_idx + 1, "shards in", OUTDIR)
