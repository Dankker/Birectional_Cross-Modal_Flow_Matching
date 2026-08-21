"""Shared data, text, metric, and distributed helpers."""

import collections
import importlib.util
import json
import math
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import Dataset


EOT_ID = 49407
SIGMA_COEF = 1e-5 - 1.0


def load_config(path):
    spec = importlib.util.spec_from_file_location("bvfm_image_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_config()


def setup_dist():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank, world, local_rank = 0, 1, 0
    torch.cuda.set_device(local_rank)
    return rank, world, local_rank


def is_dist():
    return dist.is_available() and dist.is_initialized()


def barrier():
    if is_dist():
        dist.barrier()


class CocoImageCaptionDataset(Dataset):
    """One training item for every COCO image-caption annotation."""

    def __init__(self, images_dir, captions_json, crop_size):
        from torchvision import transforms

        with open(captions_json) as handle:
            annotations = json.load(handle)["annotations"]
        self.images_dir = images_dir
        self.items = [
            (int(item["image_id"]), item["caption"].strip())
            for item in annotations]
        self.transform = transforms.Compose([
            transforms.Resize(
                crop_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        from PIL import Image

        image_id, caption = self.items[index]
        path = os.path.join(self.images_dir, f"{image_id:012d}.jpg")
        return self.transform(Image.open(path).convert("RGB")), caption


class ImagePathDataset(Dataset):
    def __init__(self, paths, crop_size):
        from torchvision import transforms

        self.paths = list(paths)
        self.transform = transforms.Compose([
            transforms.Resize(
                crop_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        from PIL import Image

        path = self.paths[index]
        return self.transform(Image.open(path).convert("RGB")), path


def load_val_refs(annotations_json):
    with open(annotations_json) as handle:
        annotations = json.load(handle)["annotations"]
    references = collections.defaultdict(list)
    for item in annotations:
        references[int(item["image_id"])].append(item["caption"].strip())
    return references


def load_val_images(config, _tokenizer=None):
    """Preload the small validation bank used by training diagnostics."""
    from PIL import Image
    from torchvision import transforms

    references = load_val_refs(config.data.val_captions)
    transform = transforms.Compose([
        transforms.Resize(
            config.dataset.crop_size,
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True),
        transforms.CenterCrop(config.dataset.crop_size),
        transforms.ToTensor(),
    ])
    output = []
    for image_id in sorted(references):
        path = os.path.join(
            config.data.val_images_dir, f"{image_id:012d}.jpg")
        if not os.path.isfile(path):
            continue
        output.append((
            transform(Image.open(path).convert("RGB")),
            references[image_id], image_id))
        if len(output) >= int(config.diag.n_val_images):
            break
    return output


@torch.no_grad()
def clip_text_features(clip_encoder, text_ids):
    cast_dtype = clip_encoder.transformer.get_cast_dtype()
    x = clip_encoder.token_embedding(text_ids).to(cast_dtype)
    x = x + clip_encoder.positional_embedding.to(cast_dtype)
    x = x.permute(1, 0, 2)
    x = clip_encoder.transformer(x, attn_mask=clip_encoder.attn_mask)
    x = x.permute(1, 0, 2)
    return clip_encoder.ln_final(x)


def ids_to_text(ids, clip_tokenizer):
    ids = ids.tolist()
    if EOT_ID in ids:
        ids = ids[:ids.index(EOT_ID)]
    ids = [token for token in ids if token not in (0, 49406)]
    try:
        text = clip_tokenizer.decode(ids)
    except AttributeError:
        from open_clip.tokenizer import _tokenizer
        text = _tokenizer.decode(ids)
    return text.strip()


def bleu4(hypotheses, references_list):
    def ngrams(tokens, order):
        return collections.Counter(
            tuple(tokens[index:index + order])
            for index in range(len(tokens) - order + 1))

    clipped = [0] * 4
    totals = [0] * 4
    hypothesis_length = 0
    reference_length = 0
    for hypothesis, references in zip(hypotheses, references_list):
        hypothesis_tokens = hypothesis.lower().split()
        reference_tokens = [reference.lower().split()
                            for reference in references]
        hypothesis_length += len(hypothesis_tokens)
        reference_length += min(
            (abs(len(reference) - len(hypothesis_tokens)), len(reference))
            for reference in reference_tokens)[1]
        for order in range(1, 5):
            hypothesis_ngrams = ngrams(hypothesis_tokens, order)
            max_reference = collections.Counter()
            for reference in reference_tokens:
                max_reference |= ngrams(reference, order)
            clipped[order - 1] += sum(
                min(count, max_reference[gram])
                for gram, count in hypothesis_ngrams.items())
            totals[order - 1] += max(
                0, len(hypothesis_tokens) - order + 1)
    if min(totals) == 0 or min(clipped) == 0:
        return 0.0
    log_precision = sum(
        math.log(count / total)
        for count, total in zip(clipped, totals)) / 4
    brevity = (
        1.0 if hypothesis_length > reference_length
        else math.exp(
            1.0 - reference_length / max(hypothesis_length, 1)))
    return brevity * math.exp(log_precision)


def ar_caption_loss(logits, text_ids):
    target = text_ids[:, 1:]
    eot_cumulative = (target == EOT_ID).cumsum(dim=-1)
    mask = (eot_cumulative == 0) | (target == EOT_ID)
    cross_entropy = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        target.reshape(-1), reduction="none").reshape(target.shape)
    loss = (cross_entropy * mask).sum() / mask.sum()
    accuracy = ((logits.argmax(-1) == target) * mask).sum() / mask.sum()
    return loss, accuracy


def flow_interp(time, source, target):
    view = time.view(-1, *([1] * (source.ndim - 1)))
    state = (1.0 + view * SIGMA_COEF) * source + view * target
    velocity = SIGMA_COEF * source + target
    return state, velocity


def logit_normal_time(batch, device):
    return torch.sigmoid(torch.randn(batch, device=device))


@torch.no_grad()
def encode_image_latents(autoencoder, images, scale_factor):
    sampled, posterior = autoencoder.encode(images)
    mean = posterior.mode()
    sampled = sampled.mul(scale_factor).squeeze(2).permute(0, 2, 1)
    mean = mean.mul(scale_factor).squeeze(2).permute(0, 2, 1)
    return sampled.float(), mean.float()
