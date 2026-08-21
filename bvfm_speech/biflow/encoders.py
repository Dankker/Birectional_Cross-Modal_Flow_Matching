import torch
import torch.nn as nn


class FrozenSpeechT5TextEncoder(nn.Module):
    def __init__(self, model_name="microsoft/speecht5_tts", device="cuda", layer_idx=-1):
        super().__init__()
        print(f"Loading Frozen {model_name} ...")
        from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech
        self.processor = SpeechT5Processor.from_pretrained(model_name)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(model_name).to(device).eval()
        self.device = device
        self.layer_idx = int(layer_idx)
        for p in self.model.parameters():
            p.requires_grad = False

        H = None
        for attr in ["hidden_size", "d_model", "encoder_hidden_size"]:
            if hasattr(self.model.config, attr):
                H = int(getattr(self.model.config, attr))
                break
        self.hidden_size = H if H is not None else 768
        try:
            self.pad_id = int(self.processor.tokenizer.pad_token_id)
        except Exception:
            self.pad_id = 1
        print(f"SpeechT5 hidden: {self.hidden_size} pad: {self.pad_id} layer_idx={self.layer_idx}")

    @torch.no_grad()
    def forward(self, texts):
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(self.device)

        try:
            if hasattr(self.model, "speecht5") and hasattr(self.model.speecht5, "encoder"):
                enc = self.model.speecht5.encoder
                out = enc(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hs = out.hidden_states
                h = hs[self.layer_idx] if hs is not None else out.last_hidden_state
                return h.float(), attention_mask.bool()
        except Exception:
            pass

        try:
            B = input_ids.shape[0]
            mel_bins = int(getattr(self.model.config, "num_mel_bins", 80))
            decoder_input_values = torch.zeros((B, 1, mel_bins), device=self.device)
            spk_dim = int(getattr(self.model.config, "speaker_embedding_dim", 512))
            speaker_embeddings = torch.zeros((B, spk_dim), device=self.device)

            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_values=decoder_input_values,
                speaker_embeddings=speaker_embeddings,
                output_hidden_states=True,
                return_dict=True,
            )
            if hasattr(out, "encoder_outputs") and out.encoder_outputs is not None:
                eo = out.encoder_outputs
                if hasattr(eo, "hidden_states") and eo.hidden_states is not None:
                    h = eo.hidden_states[self.layer_idx]
                    return h.float(), attention_mask.bool()
                if hasattr(eo, "last_hidden_state"):
                    return eo.last_hidden_state.float(), attention_mask.bool()
            if hasattr(out, "encoder_last_hidden_state") and out.encoder_last_hidden_state is not None:
                return out.encoder_last_hidden_state.float(), attention_mask.bool()
        except Exception as e:
            raise RuntimeError(f"Could not extract SpeechT5 text encoder hidden states: {repr(e)}")

        raise RuntimeError("Could not extract SpeechT5 text encoder hidden states (no valid path).")

class FrozenHubertSSLTeacher(nn.Module):
    def __init__(self, model_name="facebook/hubert-base-ls960", device="cuda", layer_idx=-1):
        super().__init__()
        print(f"Loading frozen SSL teacher {model_name} ...")
        from transformers import AutoFeatureExtractor, HubertModel
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = HubertModel.from_pretrained(model_name).to(device).eval()
        self.device = device
        self.layer_idx = int(layer_idx)
        for p in self.model.parameters():
            p.requires_grad = False
        self.hidden_size = int(getattr(self.model.config, "hidden_size", 768))
        print(f"HuBERT hidden: {self.hidden_size} layer_idx={self.layer_idx}")

    @torch.no_grad()
    def forward_wave_list(self, wav_list, sampling_rate=16000):
        inputs = self.feature_extractor(
            wav_list,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs["input_values"].to(self.device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        out = self.model(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hs = out.hidden_states
        h = hs[self.layer_idx] if hs is not None else out.last_hidden_state
        if attention_mask is not None and hasattr(self.model, "_get_feature_vector_attention_mask"):
            feat_mask = self.model._get_feature_vector_attention_mask(h.shape[1], attention_mask)
        else:
            feat_mask = torch.ones(h.shape[:2], device=h.device, dtype=torch.bool)
        return h.float(), feat_mask.bool()
