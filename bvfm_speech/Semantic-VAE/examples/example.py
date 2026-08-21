import warnings
from pathlib import Path
import sys

import torch
import torchaudio

torch.backends.mkldnn.enabled = False

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dac.model.dac import DAC
from dac.model.utils import read_json_file

warnings.filterwarnings("ignore")


def load_model(save_path: str) -> DAC:
    metainfo = read_json_file(Path(save_path) / "metainfo.json")
    bigvgan_conf = Path(metainfo["DAC"]["bigvgan_conf"])
    if not bigvgan_conf.is_absolute():
        metainfo["DAC"]["bigvgan_conf"] = str(REPO_ROOT / bigvgan_conf)
    ckpt = torch.load(
        Path(save_path) / "dac" / "ema_state_dict.pth", map_location="cpu"
    )

    ckpt = {k.replace("ema_model.", ""): v for k, v in ckpt.items()}
    ckpt = {k: v for k, v in ckpt.items() if not k.startswith("projectors")}

    model = DAC(**metainfo["DAC"])
    del model.projectors
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model


# load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_path = REPO_ROOT / "ckpts" / "semantic_vae_1000k"
model = load_model(save_path=save_path).to(device)

# load audio
audio_path = Path(r"/work/dankker0900/dataset/libritts_r/LibriTTS_R/train-clean-100/103/1241/103_1241_000052_000006.wav")
wav, sr = torchaudio.load(audio_path)
if wav.shape[0] > 1:
    wav = wav.mean(dim=0, keepdim=True)

# resample
if sr != model.sample_rate:
    wav = torchaudio.functional.resample(wav, sr, model.sample_rate)
wav = model.preprocess(wav, model.sample_rate)  # 1, T
wav = wav.to(device)

# encode
with torch.no_grad():
    z_hat, _, _, _ = model.encode(wav.unsqueeze(0))
    x_hat = model.decode(z_hat)

# decode
x_hat = model.decode(z_hat)

out_path = Path(__file__).resolve().parent / f"{Path(audio_path).stem}_recon.wav"
torchaudio.save(
    out_path,
    x_hat.squeeze(0).detach().cpu(),
    sample_rate=model.sample_rate,
)
