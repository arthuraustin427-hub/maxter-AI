"""
MixerAI Backend — FastAPI Audio Processing Server
Run: uvicorn backend:app --reload --port 8000
Requires: pip install fastapi uvicorn python-multipart numpy scipy soundfile librosa
"""

import io
import json
import math
import logging
import tempfile
import os
from typing import Optional

import numpy as np
import scipy.signal as signal
import soundfile as sf
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mixerai")

app = FastAPI(title="MixerAI Backend", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────
# HEALTH
# ─────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}


# ─────────────────────────────────────────
# MAIN PROCESS ENDPOINT
# ─────────────────────────────────────────
@app.post("/process")
async def process_audio(
    file: UploadFile = File(...),
    settings: str = Form("{}"),
):
    """
    Accepts an audio file + JSON settings blob.
    Returns a processed 16-bit WAV file.
    """
    try:
        params = json.loads(settings)
    except json.JSONDecodeError:
        params = {}

    logger.info(f"Processing '{file.filename}' | mode={params.get('mode','balanced')}")

    # ── Read audio ──
    raw = await file.read()
    try:
        audio, sr = sf.read(io.BytesIO(raw), always_2d=True)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Cannot decode audio: {e}")

    # soundfile gives (samples, channels); work as float64
    audio = audio.astype(np.float64)
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]

    logger.info(f"  sr={sr}  shape={audio.shape}  mode={params.get('mode')}")

    processor = AudioProcessor(sr)

    # ── Pipeline ──
    audio = processor.apply_eq(audio, params.get("eq", []))
    audio = processor.apply_compression(audio, params)
    audio = processor.apply_mode_processing(audio, params.get("mode", "balanced"), params)
    audio = processor.apply_stereo_width(audio, params.get("width", 100))
    audio = processor.apply_reverb(audio, params.get("reverb", 0))
    audio = processor.apply_limiter(audio, params.get("limiter", -1.0))
    audio = processor.normalize(audio)

    # ── Encode WAV ──
    buf = io.BytesIO()
    sf.write(buf, audio, sr, subtype="PCM_16", format="WAV")
    buf.seek(0)

    return Response(
        content=buf.read(),
        media_type="audio/wav",
        headers={"Content-Disposition": f'attachment; filename="mixerai_processed.wav"'},
    )


# ─────────────────────────────────────────
# AUDIO PROCESSOR
# ─────────────────────────────────────────
class AudioProcessor:
    EQ_FREQS = [60, 125, 250, 500, 1000, 2000, 8000, 16000]

    def __init__(self, sr: int):
        self.sr = sr

    # ── EQ ──
    def apply_eq(self, audio: np.ndarray, gains_db: list) -> np.ndarray:
        if not gains_db:
            return audio
        out = audio.copy()
        for i, gain_db in enumerate(gains_db[:len(self.EQ_FREQS)]):
            if abs(gain_db) < 0.1:
                continue
            freq = self.EQ_FREQS[i]
            nyq = self.sr / 2.0
            if freq >= nyq:
                continue
            w0 = freq / nyq
            Q = 1.4
            A = 10 ** (gain_db / 40.0)
            alpha = math.sin(math.pi * w0) / (2 * Q)
            b0 = 1 + alpha * A
            b1 = -2 * math.cos(math.pi * w0)
            b2 = 1 - alpha * A
            a0 = 1 + alpha / A
            a1 = -2 * math.cos(math.pi * w0)
            a2 = 1 - alpha / A
            b = np.array([b0/a0, b1/a0, b2/a0])
            a = np.array([1.0, a1/a0, a2/a0])
            for ch in range(out.shape[1]):
                out[:, ch] = signal.lfilter(b, a, out[:, ch])
        return out

    # ── COMPRESSION ──
    def apply_compression(self, audio: np.ndarray, params: dict) -> np.ndarray:
        threshold_db = params.get("threshold", -24.0)
        ratio = max(1.0, params.get("ratio", 4.0))
        attack_ms = params.get("attack", 10.0)
        release_ms = params.get("release", 250.0)
        makeup_db = params.get("makeup", 0.0)

        threshold_lin = 10 ** (threshold_db / 20.0)
        makeup_lin = 10 ** (makeup_db / 20.0)
        attack_coef = math.exp(-1.0 / (self.sr * attack_ms / 1000.0))
        release_coef = math.exp(-1.0 / (self.sr * release_ms / 1000.0))

        out = audio.copy()
        envelope = np.zeros(audio.shape[0])
        env = 0.0
        for i in range(len(envelope)):
            level = float(np.max(np.abs(audio[i])))
            if level > env:
                env = attack_coef * env + (1 - attack_coef) * level
            else:
                env = release_coef * env + (1 - release_coef) * level
            envelope[i] = env

        gain = np.ones_like(envelope)
        over = envelope > threshold_lin
        gain[over] = (threshold_lin + (envelope[over] - threshold_lin) / ratio) / np.maximum(envelope[over], 1e-10)
        gain = gain[:, np.newaxis] * makeup_lin

        out *= gain
        return out

    # ── MODE-SPECIFIC PROCESSING ──
    def apply_mode_processing(self, audio: np.ndarray, mode: str, params: dict) -> np.ndarray:
        handlers = {
            "vocal_forward": self._mode_vocal_forward,
            "bass_heavy":    self._mode_bass_heavy,
            "broadcast":     self._mode_broadcast,
            "lo_fi":         self._mode_lo_fi,
            "cinematic":     self._mode_cinematic,
            "aggressive":    self._mode_aggressive,
        }
        fn = handlers.get(mode)
        if fn:
            audio = fn(audio, params)
        # apply warmth and presence regardless
        audio = self._apply_warmth(audio, params.get("warmth", 30))
        audio = self._apply_presence(audio, params.get("presence", 50))
        return audio

    def _mode_vocal_forward(self, audio, params):
        # Boost presence band 2-5kHz, slight high-pass
        audio = self._highpass(audio, 80)
        audio = self._shelf(audio, 2000, 2.5)
        return audio

    def _mode_bass_heavy(self, audio, params):
        audio = self._shelf_low(audio, 200, 3.0)
        audio = self._highpass(audio, 30)
        return audio

    def _mode_broadcast(self, audio, params):
        audio = self._highpass(audio, 100)
        audio = self._lowpass(audio, 16000)
        audio = self._shelf(audio, 3000, 1.5)
        return audio

    def _mode_lo_fi(self, audio, params):
        audio = self._lowpass(audio, 8000)
        audio = self._add_warmth_saturation(audio, 0.3)
        audio = self._shelf_low(audio, 200, 2.0)
        return audio

    def _mode_cinematic(self, audio, params):
        audio = self._highpass(audio, 40)
        audio = self._shelf(audio, 8000, 1.5)
        audio = self._shelf_low(audio, 100, 1.5)
        return audio

    def _mode_aggressive(self, audio, params):
        audio = self._add_warmth_saturation(audio, 0.15)
        audio = self._shelf(audio, 4000, 2.0)
        audio = self._shelf_low(audio, 80, 2.0)
        return audio

    def _apply_warmth(self, audio, warmth_pct):
        if warmth_pct <= 0:
            return audio
        gain = warmth_pct / 100.0
        return self._shelf_low(audio, 250, gain * 3.0)

    def _apply_presence(self, audio, pres_pct):
        if pres_pct <= 0:
            return audio
        gain = (pres_pct - 50) / 100.0 * 4.0
        if abs(gain) < 0.1:
            return audio
        return self._shelf(audio, 3000, gain)

    # ── FILTERS ──
    def _highpass(self, audio, freq):
        nyq = self.sr / 2.0
        f = min(freq / nyq, 0.99)
        b, a = signal.butter(2, f, btype='high')
        out = audio.copy()
        for ch in range(out.shape[1]):
            out[:, ch] = signal.filtfilt(b, a, out[:, ch])
        return out

    def _lowpass(self, audio, freq):
        nyq = self.sr / 2.0
        f = min(freq / nyq, 0.99)
        b, a = signal.butter(2, f, btype='low')
        out = audio.copy()
        for ch in range(out.shape[1]):
            out[:, ch] = signal.filtfilt(b, a, out[:, ch])
        return out

    def _shelf(self, audio, freq, gain_db):
        """High shelf boost/cut"""
        out = audio.copy()
        nyq = self.sr / 2.0
        f = min(freq / nyq, 0.95)
        A = 10 ** (gain_db / 40.0)
        w0 = math.pi * f
        alpha = math.sin(w0) / 2.0 * math.sqrt((A + 1/A) * (1/0.707 - 1) + 2)
        cos_w0 = math.cos(w0)
        b0 =     A*( (A+1) + (A-1)*cos_w0 + 2*math.sqrt(A)*alpha )
        b1 =  -2*A*( (A-1) + (A+1)*cos_w0                         )
        b2 =     A*( (A+1) + (A-1)*cos_w0 - 2*math.sqrt(A)*alpha )
        a0 =         (A+1) - (A-1)*cos_w0 + 2*math.sqrt(A)*alpha
        a1 =     2*( (A-1) - (A+1)*cos_w0                         )
        a2 =         (A+1) - (A-1)*cos_w0 - 2*math.sqrt(A)*alpha
        b = np.array([b0/a0, b1/a0, b2/a0])
        a = np.array([1.0, a1/a0, a2/a0])
        for ch in range(out.shape[1]):
            out[:, ch] = signal.lfilter(b, a, out[:, ch])
        return out

    def _shelf_low(self, audio, freq, gain_db):
        """Low shelf boost/cut"""
        out = audio.copy()
        nyq = self.sr / 2.0
        f = min(freq / nyq, 0.95)
        A = 10 ** (gain_db / 40.0)
        w0 = math.pi * f
        cos_w0 = math.cos(w0)
        alpha = math.sin(w0) / 2.0 * math.sqrt((A + 1/A) * (1/0.707 - 1) + 2)
        b0 =     A*( (A+1) - (A-1)*cos_w0 + 2*math.sqrt(A)*alpha )
        b1 =   2*A*( (A-1) - (A+1)*cos_w0                         )
        b2 =     A*( (A+1) - (A-1)*cos_w0 - 2*math.sqrt(A)*alpha )
        a0 =         (A+1) + (A-1)*cos_w0 + 2*math.sqrt(A)*alpha
        a1 =    -2*( (A-1) + (A+1)*cos_w0                         )
        a2 =         (A+1) + (A-1)*cos_w0 - 2*math.sqrt(A)*alpha
        b = np.array([b0/a0, b1/a0, b2/a0])
        a = np.array([1.0, a1/a0, a2/a0])
        for ch in range(out.shape[1]):
            out[:, ch] = signal.lfilter(b, a, out[:, ch])
        return out

    def _add_warmth_saturation(self, audio, amount):
        """Soft-clip saturation for warmth"""
        out = audio.copy()
        k = 2.0 * amount
        out = np.tanh(out * (1 + k)) / (1 + k * 0.5)
        return out

    # ── STEREO WIDTH ──
    def apply_stereo_width(self, audio: np.ndarray, width_pct: float) -> np.ndarray:
        if audio.shape[1] < 2:
            return audio
        width = width_pct / 100.0
        L, R = audio[:, 0], audio[:, 1]
        mid  = (L + R) * 0.5
        side = (L - R) * 0.5
        side *= width
        out = audio.copy()
        out[:, 0] = mid + side
        out[:, 1] = mid - side
        return out

    # ── REVERB ──
    def apply_reverb(self, audio: np.ndarray, reverb_pct: float) -> np.ndarray:
        if reverb_pct < 1:
            return audio
        wet = reverb_pct / 100.0
        # Simple Schroeder reverb with comb + allpass filters
        out = audio.copy()
        rt60 = 0.5 + wet * 2.5  # reverb time 0.5 - 3s
        comb_delays = [1557, 1617, 1491, 1422]  # samples at 44100
        allpass_delays = [225, 556]
        scale = float(self.sr) / 44100.0

        for ch in range(out.shape[1]):
            dry = audio[:, ch].copy()
            combs = []
            for d in comb_delays:
                d_scaled = max(1, int(d * scale))
                g = math.exp(-3.0 * d_scaled / (rt60 * self.sr))
                combs.append(self._comb_filter(dry, d_scaled, g))
            wet_sig = sum(combs) / len(combs)
            for d in allpass_delays:
                d_scaled = max(1, int(d * scale))
                wet_sig = self._allpass_filter(wet_sig, d_scaled, 0.5)
            out[:, ch] = (1 - wet * 0.7) * dry + wet * 0.7 * wet_sig

        return out

    def _comb_filter(self, x, delay, g):
        out = np.zeros_like(x)
        buf = np.zeros(delay)
        idx = 0
        for i in range(len(x)):
            out[i] = x[i] + g * buf[idx]
            buf[idx] = out[i]
            idx = (idx + 1) % delay
        return out

    def _allpass_filter(self, x, delay, g):
        out = np.zeros_like(x)
        buf = np.zeros(delay)
        idx = 0
        for i in range(len(x)):
            bufval = buf[idx]
            out[i] = -g * x[i] + bufval + g * bufval
            buf[idx] = x[i] + g * bufval
            idx = (idx + 1) % delay
        return out

    # ── LIMITER ──
    def apply_limiter(self, audio: np.ndarray, ceiling_db: float) -> np.ndarray:
        ceiling = 10 ** (ceiling_db / 20.0)
        lookahead = int(0.001 * self.sr)
        out = audio.copy()
        gain = np.ones(len(audio))
        peak = np.max(np.abs(audio), axis=1)
        for i in range(len(peak)):
            if peak[i] > ceiling:
                g = ceiling / peak[i]
                start = max(0, i - lookahead)
                gain[start:i+1] = np.minimum(gain[start:i+1], np.linspace(gain[start], g, i+1-start))
        out *= gain[:, np.newaxis]
        return out

    # ── NORMALIZE ──
    def normalize(self, audio: np.ndarray, target_db: float = -0.3) -> np.ndarray:
        target = 10 ** (target_db / 20.0)
        peak = np.max(np.abs(audio))
        if peak < 1e-10:
            return audio
        return audio * (target / peak)


# ─────────────────────────────────────────
# RUN
# ─────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("\n🎛  MixerAI Backend — http://localhost:8000\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
