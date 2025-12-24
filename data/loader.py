import io
import torch
import webdataset as wds
from torch.utils.data import DataLoader

def _decode_pt(sample):
    return torch.load(io.BytesIO(sample["pt"]))

def make_wds(
        shards: str,
        batch_size: int,
        num_workers: int,
        T_train: int = 48, # number of frames from an example, NOTE: maximum frames per sample in current sharding setup is 72
        stride: int = 1, # temporal stride
        hflip_p: float = 0.5,
        time_reverse_p: float = 0.0,
        shuffle_buf: int = 256,
        shard_shuffle: bool = False,
):
    dataset = wds.WebDataset(shards, shardshuffle=shard_shuffle)

    # compat split (optional)
    if hasattr(dataset, "split_by_node"):
        dataset = dataset.split_by_node()
    elif hasattr(wds, "split_by_node"):
        dataset = dataset.compose(wds.split_by_node)
    

    if hasattr(dataset, "split_by_worker"):
        dataset = dataset.split_by_worker()
    elif hasattr(wds, "split_by_worker"):
        dataset = dataset.compose(wds.split_by_worker)

    
    dataset = dataset.shuffle(shuffle_buf).decode().map(_decode_pt)

    def augment(rec):
        z = rec["z"] # (T_full, 4, 40, 40)
        T_full = z.shape[0]

        # temporal stride (optional)
        if stride > 1:
            z = z[::stride]
            T_full = z.shape[0]
        
        # select window
        if T_train < T_full:
            s = torch.randint(0, T_full - T_train +1, (1,)).item() # random over interval [,) hence +1
            z = z[s:s+T_train]

        # flip width
        if hflip_p > 0 and torch.rand(1).item() < hflip_p:
            z = torch.flip(z, dims=[-1]) # (time, c, H, W)
        
        # time reverse
        if time_reverse_p > 0 and torch.rand(1).item() < time_reverse_p:
            z = torch.flip(z, dims=(0))
        
        rec["z"] = z

        return rec
    
    dataset = dataset.map(augment)


    def collate(batch):
        z = torch.stack([b["z"] for b in batch], dim=0) # (B, time, C, H, W)
        y = torch.tensor([int(b["label_id"]) for b in batch], dtype=torch.long)
        return {"z": z, "label_id": y}
    
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                      collate_fn=collate, pin_memory=True, persistent_workers=(num_workers > 0))


        
