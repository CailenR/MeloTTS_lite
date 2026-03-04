"""
OpenVoice v2 tone conversion via ONNX. No PyTorch dependency.

- load_se_from_file(): load SE from portable JSON or .bin + .bin.json
- spectrogram_numpy(): match OpenVoice v2 STFT (for converter input)
- OnnxToneConverter: run voice conversion ONNX
- apply_tone_conversion(): apply converter to audio
- synthesize_then_convert(): TTS then convert in one call
"""

import json
import os
from typing import Optional, Tuple, Union

import numpy as np
import onnxruntime as ort
import soundfile as sf


# OpenVoice v2 default (if no config loaded)
DEFAULT_GIN_CHANNELS = 256


def load_se_from_file(path: str) -> np.ndarray:
    """
    Load tone-color SE from portable format. Returns array shape (1, gin_channels, 1).

    Supports:
    - .json: {"se": [f1, f2, ...], "gin_channels": N}
    - .bin: raw little-endian float32; requires path.bin.json with {"gin_channels": N}
    """
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    if path.endswith(".json") and not path.endswith(".bin.json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        se_list = data.get("se", data.get("embedding", []))
        if not se_list:
            raise ValueError(f"No 'se' or 'embedding' key in {path}")
        se = np.array(se_list, dtype=np.float32)
    else:
        bin_path = path if path.endswith(".bin") else path.rsplit(".", 1)[0] + ".bin"
        meta_path = bin_path + ".json"
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"Binary SE requires metadata {meta_path}")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        gin_channels = int(meta["gin_channels"])
        with open(bin_path, "rb") as f:
            raw = f.read()
        se = np.frombuffer(raw, dtype=np.float32)
        if se.size != gin_channels:
            se = np.frombuffer(raw, dtype=np.float32, count=gin_channels)

    se = se.reshape(1, -1, 1)
    return se


def spectrogram_numpy(
    y: np.ndarray,
    n_fft: int,
    hop_length: int,
    win_length: int,
    center: bool = False,
) -> np.ndarray:
    """
    Magnitude spectrogram matching OpenVoice spectrogram_torch (reflect pad, Hann, onesided).
    y: (samples,) float32; returns (n_fft//2 + 1, n_frames) float32.
    """
    pad = (n_fft - hop_length) // 2
    y_pad = np.pad(y.astype(np.float32), (pad, pad), mode="reflect")
    window = np.hanning(win_length).astype(np.float32)
    n_frames = 1 + (len(y_pad) - win_length) // hop_length
    spec_freq = n_fft // 2 + 1
    out = np.zeros((spec_freq, n_frames), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_length
        segment = y_pad[start : start + win_length] * window
        fft = np.fft.rfft(segment, n=n_fft)
        mag = np.sqrt(fft.real ** 2 + fft.imag ** 2 + 1e-6)
        out[:, i] = mag.astype(np.float32)
    return out


def resample_linear(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Simple linear-interpolation resample. Prefer scipy/librosa in non-minimal setups."""
    if orig_sr == target_sr:
        return audio
    duration = len(audio) / orig_sr
    n_out = int(round(duration * target_sr))
    x_old = np.linspace(0, len(audio) - 1, num=len(audio), dtype=np.float32)
    x_new = np.linspace(0, len(audio) - 1, num=n_out, dtype=np.float32)
    return np.interp(x_new, x_old, audio).astype(np.float32)


class OnnxToneConverter:
    """OpenVoice v2 voice conversion via ONNX. CPU only."""

    def __init__(
        self,
        onnx_path: str,
        config_path: Optional[str] = None,
        config: Optional[dict] = None,
    ):
        if config is None and config_path and os.path.isfile(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        if config is None:
            config_path_guess = onnx_path.rsplit(".", 1)[0] + "_config.json"
            if os.path.isfile(config_path_guess):
                with open(config_path_guess, "r", encoding="utf-8") as f:
                    config = json.load(f)
            else:
                raise FileNotFoundError("Need converter config (json dict or path).")
        self.config = config
        self.sampling_rate = int(config["sampling_rate"])
        self.filter_length = int(config["filter_length"])
        self.hop_length = int(config["hop_length"])
        self.win_length = int(config["win_length"])
        self.gin_channels = int(config.get("gin_channels", DEFAULT_GIN_CHANNELS))

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 4
        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

    def convert(
        self,
        audio: np.ndarray,
        src_se: np.ndarray,
        tgt_se: np.ndarray,
        input_sr: Optional[int] = None,
        tau: float = 0.3,
    ) -> np.ndarray:
        """
        Convert audio from source to target tone color.
        audio: (samples,) float32; can be any length/sr; resampled to converter SR if input_sr given.
        src_se, tgt_se: (1, gin_channels, 1) from load_se_from_file or preloaded.
        Returns (samples,) float32 at self.sampling_rate.
        """
        if input_sr is not None and input_sr != self.sampling_rate:
            audio = resample_linear(audio, input_sr, self.sampling_rate)
        spec = spectrogram_numpy(
            audio,
            n_fft=self.filter_length,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )
        spec = spec.astype(np.float32)
        if spec.ndim == 2:
            spec = spec[np.newaxis, ...]
        T = spec.shape[2]
        spec_lengths = np.array([T], dtype=np.int64)
        tau_arr = np.array([tau], dtype=np.float32)
        out, = self.session.run(
            ["audio"],
            {
                "spec": spec,
                "spec_lengths": spec_lengths,
                "sid_src": src_se.astype(np.float32),
                "sid_tgt": tgt_se.astype(np.float32),
                "tau": tau_arr,
            },
        )
        return out[0, 0, :].astype(np.float32)


def apply_tone_conversion(
    audio: np.ndarray,
    sample_rate: int,
    src_se: Union[np.ndarray, str],
    tgt_se: Union[np.ndarray, str],
    converter: Optional[OnnxToneConverter] = None,
    converter_onnx_path: Optional[str] = None,
    converter_config_path: Optional[str] = None,
    tau: float = 0.3,
) -> Tuple[np.ndarray, int]:
    """
    Apply tone conversion to audio. Explicit step API.
    src_se/tgt_se: array (1, C, 1) or path to .json/.bin (with .bin.json).
    Returns (converted_audio, output_sample_rate).
    """
    if converter is None:
        if not converter_onnx_path or not os.path.isfile(converter_onnx_path):
            raise ValueError("Pass converter instance or converter_onnx_path (+ optional converter_config_path).")
        converter = OnnxToneConverter(converter_onnx_path, config_path=converter_config_path)
    if isinstance(src_se, str):
        src_se = load_se_from_file(src_se)
    if isinstance(tgt_se, str):
        tgt_se = load_se_from_file(tgt_se)
    out = converter.convert(audio, src_se, tgt_se, input_sr=sample_rate, tau=tau)
    return out, converter.sampling_rate


def synthesize_then_convert(
    text: str,
    output_path: str,
    target_se_path: str,
    tts=None,
    converter=None,
    src_se_path: Optional[str] = None,
    converter_onnx_path: Optional[str] = None,
    converter_config_path: Optional[str] = None,
    speed: float = 1.0,
    tau: float = 0.3,
    **tts_kw,
) -> str:
    """
    Convenience: run TTS then apply tone conversion and write output_path.
    If tts is None, creates default TTS from tts_onnx. If converter is None, creates
    OnnxToneConverter from converter_onnx_path (and optional converter_config_path).
    src_se_path: path to base (MeloTTS) speaker SE; if None, look for base_se.json next to converter.
    """
    try:
        from .tts_onnx import TTS
    except ImportError:
        from tts_onnx import TTS

    if tts is None:
        tts = TTS(**tts_kw)
    audio, sr = tts.synthesize(text, output_path=None, speed=speed)
    if converter is None:
        if converter_onnx_path and os.path.isfile(converter_onnx_path):
            converter = OnnxToneConverter(converter_onnx_path, config_path=converter_config_path)
        else:
            raise ValueError("Pass converter or converter_onnx_path.")
    if src_se_path is None:
        base_candidates = [
            os.path.join(os.path.dirname(converter_onnx_path or "."), "base_se.json"),
            "base_se.json",
        ]
        for p in base_candidates:
            if os.path.isfile(p):
                src_se_path = p
                break
        if src_se_path is None:
            raise FileNotFoundError("src_se_path not set and base_se.json not found.")
    converted, out_sr = apply_tone_conversion(
        audio, sr, src_se_path, target_se_path, converter=converter, tau=tau
    )
    sf.write(output_path, converted, out_sr)
    return output_path
