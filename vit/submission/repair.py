import sys
import torch


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def repair_checkpoint(path):
    ckpt = torch.load(path)
    in_state_dict = ckpt
    pairings = [
        (src_key, remove_prefix(src_key, "_orig_mod."))
        for src_key in in_state_dict.keys()
    ]
    if all(src_key == dest_key for src_key, dest_key in pairings):
        return  # Do not write checkpoint if no need to repair!
    out_state_dict = {}
    for src_key, dest_key in pairings:
        print(f"{src_key}  ==>  {dest_key}")
        out_state_dict[dest_key] = in_state_dict[src_key]
    ckpt = out_state_dict
    torch.save(ckpt, path)


if __name__ == "__main__":
    paths = sys.argv[1:]
    for path in paths:
        print(path)
        repair_checkpoint(path)
        print("========")