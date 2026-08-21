import os
import random
from collections import OrderedDict

import numpy as np
import torch


_COMPILED_PREFIX = "_orig_mod."


def _strip_compiled_prefix(key):
    while key.startswith(_COMPILED_PREFIX):
        key = key[len(_COMPILED_PREFIX):]
    return key


def _portable_state_dict(module):
    return OrderedDict(
        (_strip_compiled_prefix(key), value)
        for key, value in module.state_dict().items()
    )


def _map_checkpoint_key_to_current(key, current_state):
    if key in current_state:
        return key
    stripped = _strip_compiled_prefix(key)
    if stripped in current_state:
        return stripped
    prefixed = f"{_COMPILED_PREFIX}{stripped}"
    if prefixed in current_state:
        return prefixed
    return key


class MultiModuleEMA:
    def __init__(self, module_map, decay=0.9999):
        self.decay = float(decay)
        self.state = OrderedDict()
        self.set(module_map)

    def _iter_module_tensors(self, module_map):
        for module_name, module in module_map.items():
            if module is None:
                continue
            for key, value in _portable_state_dict(module).items():
                yield f"{module_name}.{key}", value.detach()

    def set(self, module_map):
        self.state = OrderedDict(
            (key, value.detach().cpu().clone())
            for key, value in self._iter_module_tensors(module_map)
        )

    @torch.no_grad()
    def update(self, module_map):
        if not self.state:
            self.set(module_map)
            return
        for key, value in self._iter_module_tensors(module_map):
            src = value.detach().cpu()
            if key not in self.state:
                self.state[key] = src.clone()
                continue
            self.state[key].mul_(self.decay).add_(src, alpha=1.0 - self.decay)

    def state_dict(self):
        return OrderedDict((key, value.clone()) for key, value in self.state.items())

    def load_state_dict(self, state_dict):
        self.state = OrderedDict((key, value.clone()) for key, value in state_dict.items())


def module_map_state_dict(module_map):
    return {
        module_name: (_portable_state_dict(module) if module is not None else None)
        for module_name, module in module_map.items()
    }


def load_module_map_state(module_map, state_dict):
    for module_name, module in module_map.items():
        if module is None:
            continue
        module_state = state_dict.get(module_name)
        if module_state is None:
            continue
        current_state = module.state_dict()
        compatible_state = OrderedDict()
        skipped_shape = []
        for key, value in module_state.items():
            load_key = _map_checkpoint_key_to_current(key, current_state)
            if load_key in current_state and current_state[load_key].shape != value.shape:
                skipped_shape.append(load_key)
                continue
            compatible_state[load_key] = value
        incompatible = module.load_state_dict(compatible_state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys or skipped_shape:
            print(
                f"[CKPT] relaxed load for {module_name}: "
                f"missing={len(incompatible.missing_keys)} "
                f"unexpected={len(incompatible.unexpected_keys)} "
                f"shape_skipped={len(skipped_shape)}"
            )


def move_optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def capture_rng_state(device):
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if device == "cuda" and torch.cuda.is_available():
        state["cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state, device):
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if device == "cuda" and torch.cuda.is_available() and "cuda_all" in state:
        torch.cuda.set_rng_state_all(state["cuda_all"])


def extract_inference_state(module_map, ema=None, use_ema=False):
    if use_ema and ema is not None:
        state = {}
        prefix = ""
        for module_name in module_map:
            if module_map[module_name] is None:
                state[module_name] = None
                continue
            prefix = f"{module_name}."
            state[module_name] = OrderedDict(
                (key[len(prefix):], value.clone())
                for key, value in ema.state.items()
                if key.startswith(prefix)
            )
        return state
    return module_map_state_dict(module_map)


def save_training_checkpoint(
    ckpt_dir,
    tag,
    step,
    module_map,
    optimizer,
    scaler,
    config_snapshot,
    extra_state=None,
    ema=None,
    use_ema=False,
    keep_last_k=3,
    device="cpu",
):
    os.makedirs(ckpt_dir, exist_ok=True)
    state = {
        "step": int(step),
        "config": config_snapshot,
        "modules": module_map_state_dict(module_map),
        "inference_modules": extract_inference_state(module_map, ema=ema, use_ema=use_ema),
        "optimizer": optimizer.state_dict(),
        "scaler": (scaler.state_dict() if scaler is not None else None),
        "ema": (ema.state_dict() if (use_ema and ema is not None) else None),
        "rng_state": capture_rng_state(device),
        "extra_state": extra_state or {},
    }

    ckpt_path = os.path.join(ckpt_dir, f"{tag}.pt")
    latest_path = os.path.join(ckpt_dir, "latest.pt")
    torch.save(state, ckpt_path)
    torch.save(state, latest_path)

    if tag.startswith("step"):
        prune_step_checkpoints(ckpt_dir, keep_last_k=keep_last_k)

    return ckpt_path, latest_path


def prune_step_checkpoints(ckpt_dir, keep_last_k=3):
    if keep_last_k is None or int(keep_last_k) <= 0:
        return
    keep_last_k = int(keep_last_k)
    ckpt_names = []
    for name in os.listdir(ckpt_dir):
        if not (name.startswith("step") and name.endswith(".pt")):
            continue
        try:
            step_num = int(name[len("step"):-3])
        except ValueError:
            continue
        ckpt_names.append((step_num, name))
    ckpt_names.sort()
    for _, name in ckpt_names[:-keep_last_k]:
        path = os.path.join(ckpt_dir, name)
        if os.path.isfile(path):
            os.remove(path)


def resolve_resume_path(resume_from, ckpt_dir):
    if not resume_from:
        return None
    if resume_from == "latest":
        candidate = os.path.join(ckpt_dir, "latest.pt")
        return candidate if os.path.isfile(candidate) else None
    return resume_from if os.path.isfile(resume_from) else None


def load_training_checkpoint(
    checkpoint_path,
    module_map,
    optimizer=None,
    scaler=None,
    ema=None,
    device="cpu",
    restore_rng=True,
):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    load_module_map_state(module_map, checkpoint["modules"])
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
            move_optimizer_state_to_device(optimizer, device)
        except ValueError as exc:
            print(f"[CKPT] skipped optimizer state restore: {exc}")
    if scaler is not None and checkpoint.get("scaler") is not None:
        try:
            scaler.load_state_dict(checkpoint["scaler"])
        except Exception as exc:
            print(f"[CKPT] skipped scaler state restore: {exc}")
    if ema is not None and checkpoint.get("ema") is not None:
        try:
            ema.load_state_dict(checkpoint["ema"])
        except Exception as exc:
            print(f"[CKPT] skipped EMA state restore: {exc}")
    if restore_rng:
        restore_rng_state(checkpoint.get("rng_state"), device=device)
    return checkpoint
