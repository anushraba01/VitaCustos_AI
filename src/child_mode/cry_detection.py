#!/usr/bin/env python3
"""
Infant Cry Detection & Classification — Raspberry Pi
With Telegram Alert System
"""
import os, sys, json, time, argparse, warnings, logging
import threading, queue
from datetime import datetime
from pathlib import Path
from collections import deque, Counter

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("CryDetector")

try:
    import numpy as np, librosa, joblib, requests
except ImportError as e:
    sys.exit(f"Missing: {e}\npip3 install numpy librosa joblib requests")

try:
    import tflite_runtime.interpreter as tflite
    TF_RUNTIME = "tflite_runtime"
except ImportError:
    try:
        import tensorflow as tf; tflite = tf.lite; TF_RUNTIME = "tensorflow"
    except ImportError:
        sys.exit("Install: pip3 install tflite-runtime")

try:
    import pyaudio; PYAUDIO_OK = True
except ImportError:
    PYAUDIO_OK = False

try:
    import soundfile as sf; SF_OK = True
except ImportError:
    SF_OK = False

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CONFIGURATION  —  SET YOUR TELEGRAM TOKEN & CHAT ID HERE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONFIG = {
    # ---------- TELEGRAM SETTINGS (REQUIRED) ----------
    "TELEGRAM_TOKEN"        : "YOUR_BOT_TOKEN_HERE",   # Get from @BotFather
    "TELEGRAM_CHAT_ID"      : "YOUR_CHAT_ID_HERE",     # Get from @userinfobot
    "telegram_enabled"      : True,
    "telegram_cooldown_sec" : 30,     # min seconds between repeat alerts
    "telegram_send_audio"   : True,   # attach audio clip to alert
    "telegram_alert_classes": None,   # None=all, or ["belly_pain","hungry"]
    "telegram_min_confidence": 0.60,  # only alert above this confidence
    # ---------- AUDIO ----------
    "sample_rate"   : 22050,
    "duration"      : 4,
    "hop_length"    : 512,
    "n_fft"         : 2048,
    "n_mfcc"        : 40,
    "n_mels"        : 128,
    "fmax"          : 8000,
    "mel_width"     : 173,
    # ---------- VAD ----------
    "vad_energy_thresh" : 0.02,
    "vad_zcr_thresh"    : 0.30,
    "vad_frame_len"     : 1024,
    # ---------- INFERENCE ----------
    "confidence_thresh"  : 0.55,
    "smoothing_window"   : 5,
    "inference_interval" : 1.0,
    # ---------- MIC ----------
    "chunk_size"    : 1024,
    "mic_channels"  : 1,
    # ---------- PATHS ----------
    "model_dir"     : "./models",
    "cnn_model"     : "cnn_cry_detector.tflite",
    "hybrid_model"  : "hybrid_cry_detector.tflite",
    "scaler_file"   : "scaler.pkl",
    "pca_file"      : "pca.pkl",
    "labels_file"   : "label_classes.npy",
    "weights_file"  : "ensemble_weights.npy",
    "config_file"   : "config.json",
    # ---------- LOGGING ----------
    "log_csv"           : "./cry_log.csv",
    "save_audio_clips"  : True,
    "audio_clips_dir"   : "./detected_cries",
}

CRY_INFO = {
    "hungry"     : {"emoji":"🍼","label":"HUNGRY",     "urgency":"MEDIUM","advice":"Baby is hungry — time to feed!"},
    "belly_pain" : {"emoji":"😣","label":"BELLY PAIN", "urgency":"HIGH",  "advice":"Baby may have gas/colic — try gentle tummy massage."},
    "burping"    : {"emoji":"💨","label":"BURPING",    "urgency":"LOW",   "advice":"Baby needs to burp — hold upright and pat gently."},
    "discomfort" : {"emoji":"😖","label":"DISCOMFORT", "urgency":"MEDIUM","advice":"Baby is uncomfortable — check diaper/temperature."},
    "tired"      : {"emoji":"😴","label":"TIRED",      "urgency":"LOW",   "advice":"Baby is sleepy — dim lights and reduce stimulation."},
    "unknown"    : {"emoji":"❓","label":"UNKNOWN",    "urgency":"NONE",  "advice":""},
}
URGENCY_ICON = {"HIGH":"🔴","MEDIUM":"🟡","LOW":"🟢","NONE":"⚪"}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TELEGRAM ALERT SYSTEM
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TelegramAlerter:
    """Non-blocking Telegram alert system with cooldown, filtering and audio upload."""
    API = "https://api.telegram.org/bot{token}/{method}"

    def __init__(self, token, chat_id, enabled=True):
        self.token   = token
        self.chat_id = str(chat_id)
        self.enabled = enabled and token != "YOUR_BOT_TOKEN_HERE"
        self._last   = {}
        self._lock   = threading.Lock()
        self._q      = queue.Queue()
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()
        if self.enabled:
            ok, info = self._call("getMe")
            if ok:
                log.info(f"Telegram OK — bot @{info['result'].get('username','?')}")
            else:
                log.warning(f"Telegram error: {info}")
        else:
            if not self.enabled and token == "YOUR_BOT_TOKEN_HERE":
                log.warning("Telegram disabled — edit TELEGRAM_TOKEN in CONFIG")

    def _call(self, method, data=None, files=None, timeout=10):
        url = self.API.format(token=self.token, method=method)
        try:
            if files:
                r = requests.post(url, data=data, files=files, timeout=timeout)
            else:
                r = requests.post(url, json=data, timeout=timeout)
            r.raise_for_status()
            return True, r.json()
        except Exception as e:
            return False, str(e)

    def _loop(self):
        while True:
            task = self._q.get()
            if task is None:
                break
            try:
                self._dispatch(task)
            except Exception as e:
                log.debug(f"Telegram dispatch error: {e}")
            self._q.task_done()

    def _make_text(self, result):
        cls  = result.get("class","unknown")
        conf = result.get("confidence",0.0)
        info = CRY_INFO.get(cls, CRY_INFO["unknown"])
        ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        probs = result.get("probabilities",{})
        prob_str = ""
        for c, p in sorted(probs.items(), key=lambda x:x[1], reverse=True):
            bar = "█"*int(p*10)+"░"*(10-int(p*10))
            em  = CRY_INFO.get(c,{}).get("emoji","")
            mk  = " ◄" if c==cls else ""
            prob_str += f"  {em}{c:<12s}[{bar}]{p*100:.0f}%{mk}\n"
        urgency = info["urgency"]
        return (
            f"{info['emoji']} *BABY CRY DETECTED!*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{URGENCY_ICON.get(urgency,'⚪')} *Urgency:* {urgency}\n"
            f"📋 *Type:* {info['label']}\n"
            f"📊 *Confidence:* {conf*100:.1f}%\n"
            f"🕐 *Time:* {ts}\n\n"
            f"💡 *Action:* {info['advice']}\n\n"
            f"📈 *Probabilities:*\n```\n{prob_str}```"
        )

    def _dispatch(self, task):
        if task.get("_type") == "text_only":
            self._call("sendMessage",
                       {"chat_id":self.chat_id,"text":task["text"],"parse_mode":"Markdown"})
            return

        result   = task["result"]
        audio_np = task.get("audio_np")
        cls      = result.get("class","unknown")
        text     = self._make_text(result)

        # Text message with retry
        for attempt in range(3):
            ok, _ = self._call("sendMessage",
                               {"chat_id":self.chat_id,"text":text,"parse_mode":"Markdown"})
            if ok:
                log.info(f"Telegram sent: {cls} ({result.get('confidence',0)*100:.1f}%)")
                break
            time.sleep(2)

        # Audio clip
        if CONFIG["telegram_send_audio"] and audio_np is not None and SF_OK:
            try:
                import io
                buf = io.BytesIO()
                sf.write(buf, audio_np, CONFIG["sample_rate"], format="WAV")
                buf.seek(0)
                fname = f"cry_{cls}_{datetime.now().strftime('%H%M%S')}.wav"
                ok2, _ = self._call("sendVoice",
                    data={"chat_id":self.chat_id,
                          "caption":f"{CRY_INFO.get(cls,{}).get('emoji','')} {cls} audio clip"},
                    files={"voice":(fname, buf, "audio/wav")})
                if ok2:
                    log.info("Telegram audio clip sent")
            except Exception as e:
                log.debug(f"Audio upload error: {e}")

    def send_alert(self, result, audio_np=None):
        if not self.enabled:
            return
        cls  = result.get("class","unknown")
        conf = result.get("confidence",0.0)
        allowed = CONFIG.get("telegram_alert_classes")
        if allowed and cls not in allowed:
            return
        if conf < CONFIG.get("telegram_min_confidence",0):
            return
        with self._lock:
            elapsed = time.time() - self._last.get(cls, 0)
            if elapsed < CONFIG["telegram_cooldown_sec"]:
                log.debug(f"Cooldown: {CONFIG['telegram_cooldown_sec']-elapsed:.0f}s left for {cls}")
                return
            self._last[cls] = time.time()
        self._q.put({"result":result,"audio_np":audio_np})

    def send_text(self, text):
        if self.enabled:
            self._q.put({"_type":"text_only","text":text})

    def send_startup(self, classes):
        if not self.enabled:
            return
        ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cls_str = "  ".join(CRY_INFO.get(c,{}).get("emoji","")+" "+c for c in classes)
        self.send_text(
            f"👶 *Infant Cry Monitor — STARTED*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🟢 System is now monitoring your baby\n"
            f"🕐 Started: {ts}\n"
            f"🎯 Detecting: `{cls_str}`\n"
            f"📱 You will receive instant alerts when a cry is detected."
        )

    def send_shutdown(self):
        if self.enabled:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.send_text(f"🔴 *Infant Cry Monitor — STOPPED*\n🕐 Stopped: {ts}")
            time.sleep(2)

    def stop(self):
        self._q.put(None)
        self._worker.join(timeout=5)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  AUDIO UTILITIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def load_pad(src, sr=None, max_len=None):
    sr = sr or CONFIG["sample_rate"]
    max_len = max_len or sr*CONFIG["duration"]
    if isinstance(src,(str,Path)):
        try:
            y,_ = librosa.load(str(src), sr=sr, mono=True)
        except Exception as e:
            log.error(f"Load error: {e}"); return None
    else:
        y = src.astype(np.float32)
        if np.abs(y).max()>1.0: y=y/32768.0
    pk = np.abs(y).max()
    if pk>0: y=y/pk*0.95
    if len(y)<max_len: y=np.pad(y,(0,max_len-len(y)))
    return y[:max_len].astype(np.float32)

def vad(y):
    fl=CONFIG["vad_frame_len"]; rms,zcr=[],[]
    for s in range(0,len(y)-fl,fl//2):
        f=y[s:s+fl]
        rms.append(np.sqrt(np.mean(f**2)))
        zcr.append(np.mean(np.abs(np.diff(np.sign(f))))/2)
    if not rms: return False,0.0,0.0
    mr=float(np.mean(rms)); mx=float(np.max(rms)); mz=float(np.mean(zcr))
    return mx>CONFIG["vad_energy_thresh"] and 0.01<mz<CONFIG["vad_zcr_thresh"], mr, mz

def mel2d(y, sr=None):
    sr=sr or CONFIG["sample_rate"]
    m=librosa.feature.melspectrogram(y=y,sr=sr,n_mels=CONFIG["n_mels"],
        n_fft=CONFIG["n_fft"],hop_length=CONFIG["hop_length"],fmax=CONFIG["fmax"])
    m=librosa.power_to_db(m,ref=np.max); w=CONFIG["mel_width"]
    if m.shape[1]<w: m=np.pad(m,((0,0),(0,w-m.shape[1])))
    m=m[:,:w]; m=(m-m.min())/(m.max()-m.min()+1e-8)
    return m[np.newaxis,...,np.newaxis].astype(np.float32)

def feat1d(y, sr=None):
    sr=sr or CONFIG["sample_rate"]; f=[]
    mfcc=librosa.feature.mfcc(y=y,sr=sr,n_mfcc=CONFIG["n_mfcc"],
        n_fft=CONFIG["n_fft"],hop_length=CONFIG["hop_length"])
    for a in [mfcc,librosa.feature.delta(mfcc),librosa.feature.delta(mfcc,order=2)]:
        f+=[a.mean(1),a.std(1),a.max(1),a.min(1)]
    mel=librosa.feature.melspectrogram(y=y,sr=sr,n_mels=CONFIG["n_mels"],
        n_fft=CONFIG["n_fft"],hop_length=CONFIG["hop_length"])
    mdb=librosa.power_to_db(mel,ref=np.max); f+=[mdb.mean(1),mdb.std(1)]
    ch=librosa.feature.chroma_stft(y=y,sr=sr,n_fft=CONFIG["n_fft"],hop_length=CONFIG["hop_length"])
    f+=[ch.mean(1),ch.std(1)]
    for fn in [
        lambda:librosa.feature.spectral_centroid(y=y,sr=sr),
        lambda:librosa.feature.spectral_bandwidth(y=y,sr=sr),
        lambda:librosa.feature.spectral_rolloff(y=y,sr=sr),
        lambda:librosa.feature.spectral_contrast(y=y,sr=sr,n_fft=CONFIG["n_fft"]),
        lambda:librosa.feature.tonnetz(y=librosa.effects.harmonic(y),sr=sr),
        lambda:librosa.feature.zero_crossing_rate(y),
        lambda:librosa.feature.rms(y=y),
    ]:
        try: a=fn(); f+=[a.mean(1),a.std(1)]
        except: pass
    try:
        f0,_,_=librosa.pyin(y,fmin=50,fmax=1000,sr=sr)
        fc=f0[~np.isnan(f0)] if f0 is not None else np.array([0.0])
        f.append(np.array([fc.mean(),fc.std(),fc.max()] if len(fc) else [0,0,0]))
    except: f.append(np.array([0.0,0.0,0.0]))
    return np.concatenate([x.flatten() for x in f]).astype(np.float32).reshape(1,-1)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TFLite ENGINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TFLiteEngine:
    def __init__(self, p, threads=4):
        if TF_RUNTIME=="tflite_runtime":
            self.interp=tflite.Interpreter(model_path=str(p),num_threads=threads)
        else:
            self.interp=tflite.Interpreter(model_path=str(p))
        self.interp.allocate_tensors()
        self.inp=self.interp.get_input_details()
        self.out=self.interp.get_output_details()
        log.info(f"Loaded: {Path(p).name}")
    def _put(self,i,d):
        det=self.inp[i]
        if det["dtype"]==np.int8:
            sc,zp=det["quantization"]; d=(d/sc+zp).astype(np.int8)
        self.interp.set_tensor(det["index"],d.astype(det["dtype"]))
    def _get(self):
        det=self.out[0]; o=self.interp.get_tensor(det["index"])
        if det["dtype"]==np.int8:
            sc,zp=det["quantization"]; o=(o.astype(np.float32)-zp)*sc
        return o.astype(np.float32)
    def predict(self,*inputs):
        for i,x in enumerate(inputs): self._put(i,x)
        self.interp.invoke(); return self._get()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CLASSIFIER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class InfantCryClassifier:
    def __init__(self, model_dir=None):
        self.md=Path(model_dir or CONFIG["model_dir"])
        self.classes=None; self.cnn=None; self.hybrid=None
        self.ml=None; self.scaler=None; self.pca=None
        self.weights=np.array([0.4,0.4,0.2])
        self._buf=deque(maxlen=CONFIG["smoothing_window"])
        self._init()

    def _init(self):
        md=self.md
        lp=md/CONFIG["labels_file"]; cp=md/CONFIG["config_file"]
        if lp.exists():
            self.classes=list(np.load(str(lp),allow_pickle=True))
        elif cp.exists():
            c=json.load(open(cp))
            self.classes=c.get("classes",[])
            for k in ["sample_rate","n_mfcc","n_mels","hop_length","n_fft","fmax","mel_width"]:
                if k in c: CONFIG[k]=c[k]
        else:
            self.classes=["hungry","belly_pain","burping","discomfort","tired"]
        log.info(f"Classes: {self.classes}")
        for attr,key in [("cnn","cnn_model"),("hybrid","hybrid_model")]:
            p=md/CONFIG[key]
            if p.exists():
                try: setattr(self,attr,TFLiteEngine(str(p)))
                except Exception as e: log.warning(f"{attr}: {e}")
        for attr,key in [("scaler","scaler_file"),("pca","pca_file")]:
            p=md/CONFIG[key]
            if p.exists(): setattr(self,attr,joblib.load(str(p))); log.info(f"{attr} loaded")
        mp=md/"ml_ensemble.pkl"
        if mp.exists():
            try: self.ml=joblib.load(str(mp)); log.info("ML ensemble loaded")
            except Exception as e: log.warning(f"ML: {e}")
        wp=md/CONFIG["weights_file"]
        if wp.exists():
            w=np.load(str(wp)); self.weights=w[:3]/w[:3].sum()
        n=sum(x is not None for x in [self.cnn,self.hybrid,self.ml])
        if n==0:
            raise RuntimeError(
                "No models in ./models/\n"
                "Extract cry_rpi_models.zip (from Colab) into ./models/")
        log.info(f"Ready — {n}/3 models | CNN={self.weights[0]:.2f} HYB={self.weights[1]:.2f} ML={self.weights[2]:.2f}")

    def _f1d(self,y):
        f=feat1d(y)
        if self.scaler: f=self.scaler.transform(f)
        if self.pca:    f=self.pca.transform(f)
        return f.astype(np.float32)

    def predict(self,y):
        is_cry,energy,zcr=vad(y)
        base={"is_cry":is_cry,"energy":energy,"zcr":zcr,
              "timestamp":datetime.now().isoformat()}
        if not is_cry:
            return {**base,"class":"unknown","confidence":0.0,"probabilities":{}}
        ps,ws=[],[]
        if self.cnn:
            try:
                p=self.cnn.predict(mel2d(y))
                if p.shape[-1]==len(self.classes): ps.append(p[0]); ws.append(self.weights[0])
            except Exception as e: log.debug(f"CNN: {e}")
        if self.hybrid:
            try:
                n=len(self.hybrid.inp)
                p=self.hybrid.predict(mel2d(y),self._f1d(y)) if n==2 else self.hybrid.predict(mel2d(y))
                if p.shape[-1]==len(self.classes): ps.append(p[0]); ws.append(self.weights[1])
            except Exception as e: log.debug(f"Hybrid: {e}")
        if self.ml:
            try:
                p=self.ml.predict_proba(self._f1d(y))
                if p.shape[-1]==len(self.classes): ps.append(p[0]); ws.append(self.weights[2])
            except Exception as e: log.debug(f"ML: {e}")
        if not ps:
            return {**base,"class":"unknown","confidence":0.0,"probabilities":{}}
        ws=np.array(ws); ws/=ws.sum()
        prob=np.average(np.array(ps),weights=ws,axis=0)
        idx=int(np.argmax(prob)); cls=self.classes[idx]; conf=float(prob[idx])
        self._buf.append(cls)
        smoothed=Counter(self._buf).most_common(1)[0][0]
        return {**base,"class":smoothed,"raw_class":cls,"confidence":conf,
                "probabilities":{self.classes[i]:float(prob[i]) for i in range(len(self.classes))}}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DISPLAY & LOG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def print_result(r):
    sep="─"*62; print(f"\n{sep}")
    print(f"  🕐 {r.get('timestamp','')}")
    if not r.get("is_cry"):
        print(f"  🔇 No cry — energy={r.get('energy',0):.4f}  zcr={r.get('zcr',0):.3f}")
        print(sep); return
    cls=r["class"]; conf=r["confidence"]; info=CRY_INFO.get(cls,CRY_INFO["unknown"])
    print(f"  {URGENCY_ICON.get(info['urgency'],'⚪')} URGENCY: {info['urgency']}")
    print(f"  {info['emoji']} TYPE: {info['label']}")
    print(f"  📊 Confidence: {conf*100:.1f}%  |  Energy: {r.get('energy',0):.4f}")
    print(f"  💡 {info['advice']}")
    if r.get("probabilities"):
        print()
        for c,p in sorted(r["probabilities"].items(),key=lambda x:x[1],reverse=True):
            bar="█"*int(p*20)+"░"*(20-int(p*20)); em=CRY_INFO.get(c,{}).get("emoji","")
            mk=" ◄" if c==cls else ""
            print(f"    {em} {c:<12s} [{bar}] {p*100:5.1f}%{mk}")
    print(sep)

def log_csv(r,path=None):
    path=path or CONFIG["log_csv"]; new=not Path(path).exists()
    try:
        with open(path,"a") as f:
            if new: f.write("timestamp,is_cry,class,confidence,energy,zcr\n")
            f.write(f"{r.get('timestamp','')},{r.get('is_cry',False)},"
                    f"{r.get('class','unknown')},{r.get('confidence',0):.4f},"
                    f"{r.get('energy',0):.4f},{r.get('zcr',0):.4f}\n")
    except: pass

def save_clip(y,cls):
    if not SF_OK: return None
    os.makedirs(CONFIG["audio_clips_dir"],exist_ok=True)
    p=os.path.join(CONFIG["audio_clips_dir"],f"{cls}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")
    try: sf.write(p,y,CONFIG["sample_rate"]); return p
    except: return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MICROPHONE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class MicCapture:
    def __init__(self):
        self._q=queue.Queue(); self._pa=None; self._st=None
    def start(self):
        if not PYAUDIO_OK: raise RuntimeError("pip3 install pyaudio")
        self._pa=pyaudio.PyAudio()
        dev=self._pa.get_default_input_device_info()
        log.info(f"Mic: {dev.get('name','default')}")
        self._st=self._pa.open(format=pyaudio.paInt16,channels=CONFIG["mic_channels"],
            rate=CONFIG["sample_rate"],input=True,frames_per_buffer=CONFIG["chunk_size"],
            stream_callback=lambda d,f,t,s:(self._q.put(np.frombuffer(d,np.int16)),pyaudio.paContinue)[1])
        self._st.start_stream(); log.info("Mic started")
    def read(self,secs):
        needed=int(CONFIG["sample_rate"]*secs)
        nchunks=(needed+CONFIG["chunk_size"]-1)//CONFIG["chunk_size"]
        col=[]; t0=time.time()
        while len(col)<nchunks:
            if time.time()-t0>secs+2: break
            try: col.append(self._q.get(timeout=0.5))
            except queue.Empty: pass
        if not col: return None
        return np.concatenate(col)[:needed].astype(np.float32)/32768.0
    def stop(self):
        if self._st: self._st.stop_stream(); self._st.close()
        if self._pa: self._pa.terminate()
        log.info("Mic stopped")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MODES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_live(clf, tg, duration=None):
    if not PYAUDIO_OK: print("pip3 install pyaudio required"); return
    win=CONFIG["duration"]
    print(f"\n{'='*62}")
    print(f"  INFANT CRY MONITOR — LIVE  (Ctrl+C to stop)")
    print(f"  Telegram: {'ON' if tg.enabled else 'OFF'}")
    print(f"{'='*62}\n")
    tg.send_startup(clf.classes)
    mic=MicCapture(); last=False; monitored=0
    try:
        mic.start(); time.sleep(0.5); t0=time.time()
        while True:
            if duration and time.time()-t0>duration:
                print(f"\nDuration {duration}s reached."); break
            audio=mic.read(win)
            if audio is None: continue
            monitored+=win; r=clf.predict(audio)
            if r:
                crying=r.get("is_cry",False); conf=r.get("confidence",0.0)
                if crying!=last or (crying and conf>=CONFIG["confidence_thresh"]):
                    print_result(r); log_csv(r)
                if crying and CONFIG.get("save_audio_clips"):
                    save_clip(audio,r["class"])
                if crying and conf>=CONFIG["confidence_thresh"]:
                    tg.send_alert(r,audio_np=audio)
                last=crying
            time.sleep(max(0,CONFIG["inference_interval"]-win*0.5))
    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        mic.stop(); tg.send_shutdown(); tg.stop()
        print(f"Monitored: {monitored:.0f}s | Log: {CONFIG['log_csv']}")

def run_file(clf, tg, path):
    p=Path(path)
    if not p.exists(): print(f"Not found: {p}"); return
    files=list(p.glob("**/*.wav")) if p.is_dir() else [p]
    print(f"\nProcessing {len(files)} file(s)...")
    for f in files:
        print(f"\n  {f.name}")
        y=load_pad(f)
        if y is None: continue
        r=clf.predict(y); print_result(r); log_csv(r)
        if r.get("is_cry") and r.get("confidence",0)>=CONFIG["confidence_thresh"]:
            tg.send_alert(r,audio_np=y)
    tg.stop()

def run_batch(clf, dataset_dir):
    from collections import defaultdict
    dp=Path(dataset_dir)
    if not dp.exists(): print(f"Not found: {dp}"); return
    classes=sorted([d for d in os.listdir(dp) if (dp/d).is_dir()])
    if not classes: print("No subdirectories found"); return
    print(f"\nBatch eval — {dp} | Classes: {classes}\n")
    correct=defaultdict(int); total=defaultdict(int)
    cm=defaultdict(lambda:defaultdict(int))
    for tc in classes:
        wavs=list((dp/tc).glob("*.wav"))
        print(f"  {tc} ({len(wavs)})...",end=" ",flush=True)
        for wf in wavs:
            y=load_pad(wf)
            if y is None: continue
            r=clf.predict(y); pc=r["class"] if r.get("is_cry") else "unknown"
            total[tc]+=1; cm[tc][pc]+=1
            if pc==tc: correct[tc]+=1
        print(f"Acc: {correct[tc]/max(total[tc],1)*100:.1f}%")
    ac=sum(correct.values()); at=sum(total.values())
    print(f"\nOVERALL: {ac/max(at,1)*100:.2f}% ({ac}/{at})")
    for tc in classes:
        print(f"  {tc:<16s}: {correct[tc]/max(total[tc],1)*100:.1f}%")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def parse_args():
    p=argparse.ArgumentParser(description="Infant Cry Detector + Telegram Alerts")
    p.add_argument("--mode",choices=["live","file","batch"],default="live")
    p.add_argument("--input",type=str,default=None)
    p.add_argument("--models",type=str,default=None)
    p.add_argument("--duration",type=int,default=None)
    p.add_argument("--threshold",type=float,default=None)
    p.add_argument("--no-telegram",action="store_true")
    p.add_argument("--tg-token",type=str,default=None)
    p.add_argument("--tg-chat",type=str,default=None)
    p.add_argument("--verbose",action="store_true")
    return p.parse_args()

def main():
    args=parse_args()
    if args.verbose: logging.getLogger().setLevel(logging.DEBUG)
    if args.threshold: CONFIG["confidence_thresh"]=args.threshold
    if args.tg_token:  CONFIG["TELEGRAM_TOKEN"]=args.tg_token
    if args.tg_chat:   CONFIG["TELEGRAM_CHAT_ID"]=args.tg_chat
    if args.no_telegram: CONFIG["telegram_enabled"]=False
    if args.models:    CONFIG["model_dir"]=args.models

    print("\n"+"═"*62)
    print("  INFANT CRY DETECTION & CLASSIFICATION — Raspberry Pi")
    print(f"  Runtime : {TF_RUNTIME}")
    print(f"  Mode    : {args.mode.upper()}")
    print(f"  Models  : {CONFIG['model_dir']}")
    print(f"  Telegram: {'ENABLED' if CONFIG['telegram_enabled'] else 'DISABLED'}")
    print("═"*62+"\n")

    try:
        clf=InfantCryClassifier(model_dir=args.models)
    except RuntimeError as e:
        print(str(e)); sys.exit(1)

    tg=TelegramAlerter(
        token  =CONFIG["TELEGRAM_TOKEN"],
        chat_id=CONFIG["TELEGRAM_CHAT_ID"],
        enabled=CONFIG["telegram_enabled"])

    if   args.mode=="live":  run_live(clf,tg,duration=args.duration)
    elif args.mode=="file":
        if not args.input: print("--input required"); sys.exit(1)
        run_file(clf,tg,args.input)
    elif args.mode=="batch":
        if not args.input: print("--input required"); sys.exit(1)
        run_batch(clf,args.input)

if __name__=="__main__":
    main()