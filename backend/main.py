import os, re, uuid, json, math, wave, zipfile, shutil, subprocess, threading
from pathlib import Path
from datetime import timedelta
from urllib.parse import urlparse
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(os.getenv("DATA_DIR", "/tmp/closetape"))
ROOT.mkdir(parents=True, exist_ok=True)
JOBS = {}
JOB_SLOT = threading.BoundedSemaphore(1)  # One video-processing job at a time on small Render instances.
app = FastAPI(title="Closetape ClipMaker API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "*").split(","), allow_methods=["GET","POST"], allow_headers=["*"])

class JobRequest(BaseModel):
    url: str
    limit: int = Field(default=10, ge=1, le=20)
    transcribe: bool = True

def valid_vod(url):
    p = urlparse(url)
    return p.scheme == "https" and p.hostname in {"twitch.tv","www.twitch.tv","m.twitch.tv"} and re.fullmatch(r"/videos/\d+/?", p.path) is not None

def run(cmd, timeout=None):
    return subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)

def label(seconds):
    return str(timedelta(seconds=max(0,int(seconds)))) if seconds >= 3600 else str(timedelta(seconds=max(0,int(seconds))))[2:]

def audio_energy(video, work):
    """Calculate five-second RMS windows without loading the full soundtrack into RAM."""
    wav = work / "audio.wav"
    vals = []
    try:
        run(["ffmpeg","-nostdin","-loglevel","error","-threads","1","-filter_threads","1","-y","-i",str(video),"-vn","-ac","1","-ar","16000","-f","wav",str(wav)], timeout=1800)
        import array
        with wave.open(str(wav), "rb") as f:
            rate = f.getframerate()
            step = rate * 5
            position = 0
            while True:
                raw = f.readframes(step)
                if not raw:
                    break
                samples = array.array("h")
                samples.frombytes(raw)
                if samples:
                    rms = math.sqrt(sum(x * x for x in samples) / len(samples))
                    vals.append((position / rate, rms))
                position += len(samples)
        return vals
    finally:
        wav.unlink(missing_ok=True)

def detect(video, work, want_transcript, limit, job_id):
    energy=audio_energy(video,work)
    duration=max((t for t,_ in energy),default=0)+5
    transcript=[]
    model_name=os.getenv("FASTER_WHISPER_MODEL","")
    if want_transcript and model_name:
        try:
            from faster_whisper import WhisperModel
            model=WhisperModel(model_name, device=os.getenv("WHISPER_DEVICE","cpu"), compute_type=os.getenv("WHISPER_COMPUTE_TYPE","int8"))
            segments,_=model.transcribe(str(video), vad_filter=True)
            transcript=[{"start":s.start,"end":s.end,"text":s.text.strip()} for s in segments]
        except Exception as e:
            JOBS[job_id]["message"]="Распознавание речи недоступно; ищем пики звука"
    # Audio peaks are a transparent fallback, not semantic AI highlights.
    ranked=[]
    if transcript:
        words=re.compile(r"\b(вау|нет|что|ах|ого|жесть|давай|боже|смешно|лол|победа|убил|клатч|nice|wow|no|what|oh|lol|insane|crazy|win|clutch|omg)\b",re.I)
        for s in transcript:
            center=(s["start"]+s["end"])/2
            nearby=[v for t,v in energy if abs(t-center)<12]
            sound=max(nearby,default=0)
            text=s["text"]
            score=(len(words.findall(text))*2.8)+(1.2 if "!" in text else 0)+min(sound/7000,3)
            ranked.append((score,max(0,s["start"]-18),min(duration,s["end"]+22),text))
    else:
        for t,v in energy:
            ranked.append((v,t-20,t+25,"Громкий / динамичный момент"))
    ranked.sort(reverse=True)
    chosen=[]
    for score,start,end,title in ranked:
        start=max(0,start);end=min(duration,end)
        if end-start<8: continue
        if all(abs(start-x[0])>35 for x in chosen):
            chosen.append((start,end,title,score))
        if len(chosen)>=limit: break
    if not chosen:
        chosen=[(max(0,t-15),min(duration,t+25),"Фрагмент записи",v) for t,v in energy[:limit]]
    return sorted(chosen)

def process(job_id, req):
    job=JOBS[job_id]; work=ROOT/job_id
    try:
        work.mkdir(parents=True,exist_ok=True)
        job.update(status="downloading",progress=8,message="Скачиваем доступный Twitch VOD…")
        source=work/"source.mp4"
        run(["yt-dlp","--no-warnings","--no-progress","--no-playlist","--format","bv*[height<=720]+ba/b[height<=720]/best[height<=720]","--merge-output-format","mp4","-o",str(work/"source.%(ext)s"),req.url],timeout=7200)
        found=list(work.glob("source.*"))
        found=[p for p in found if p.suffix.lower() in {".mp4",".mkv",".webm"}]
        if not found: raise RuntimeError("yt-dlp не вернул видеофайл")
        source=found[0]
        probe=json.loads(run(["ffprobe","-v","quiet","-show_format","-of","json",str(source)]).stdout)
        duration=float(probe["format"]["duration"])
        if duration<10: raise RuntimeError("Запись слишком короткая")
        job.update(status="analyzing",progress=28,message="Анализируем звук и речь…")
        clips=detect(source,work,req.transcribe,req.limit,job_id)
        job.update(status="exporting",progress=62,message="Экспортируем MP4-клипы в 16:9…")
        output=work/"clips";output.mkdir(exist_ok=True);items=[]
        for i,(start,end,title,score) in enumerate(clips,1):
            dest=output/f"closetape_{i:02d}.mp4"
            run(["ffmpeg","-nostdin","-loglevel","error","-threads","1","-filter_threads","1","-y","-ss",str(start),"-i",str(source),"-t",str(end-start),"-map","0:v:0","-map","0:a?","-vf","scale=w=min(1280\\,iw):h=min(720\\,ih):force_original_aspect_ratio=decrease:force_divisible_by=2","-c:v","libx264","-preset","ultrafast","-crf","27","-threads","1","-c:a","aac","-b:a","128k","-movflags","+faststart",str(dest)],timeout=1800)
            items.append({"title":title[:100],"start":start,"end":end,"start_label":label(start),"end_label":label(end),"duration":label(end-start),"file":dest.name,"download_url":f"/api/jobs/{job_id}/clips/{dest.name}"})
        archive=work/"closetape_clips.zip"
        with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
            for p in output.glob("*.mp4"): z.write(p,p.name)
        job.update(status="done",progress=100,message="Готово",clips=items,zip_url=f"/api/jobs/{job_id}/zip")
    except Exception as e:
        print(f"Closetape job {job_id} failed: {type(e).__name__}: {e}", flush=True)
        job.update(status="error",progress=100,message="Обработка остановлена",error=str(e)[:500])
    finally:
        JOB_SLOT.release()

@app.get("/api/health")
def health(): return {"ok":True,"service":"Closetape ClipMaker"}

@app.post("/api/jobs")
def create_job(req: JobRequest):
    if not valid_vod(req.url): raise HTTPException(400,"Нужна прямая HTTPS-ссылка Twitch вида https://www.twitch.tv/videos/123456")
    if not JOB_SLOT.acquire(blocking=False):
        raise HTTPException(429,"Сервер уже обрабатывает другой стрим. Дождитесь завершения текущей задачи.")
    job_id=uuid.uuid4().hex
    JOBS[job_id]={"id":job_id,"status":"queued","progress":2,"message":"Задача создана","clips":[]}
    try:
        threading.Thread(target=process,args=(job_id,req),daemon=True).start()
    except Exception:
        JOB_SLOT.release()
        JOBS.pop(job_id, None)
        raise
    return {"id":job_id,"status":"queued"}

@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job=JOBS.get(job_id)
    if not job: raise HTTPException(404,"Задача не найдена или сервер перезапустился")
    return {k:v for k,v in job.items() if k not in {"error"}} | ({"error":job["error"]} if job.get("error") else {})

@app.get("/api/jobs/{job_id}/clips/{filename}")
def get_clip(job_id: str, filename: str):
    if not re.fullmatch(r"closetape_\d+\.mp4",filename): raise HTTPException(400,"Недопустимое имя файла")
    path=ROOT/job_id/"clips"/filename
    if not path.is_file(): raise HTTPException(404,"Клип не найден")
    return FileResponse(path,media_type="video/mp4",filename=filename)

@app.get("/api/jobs/{job_id}/zip")
def get_zip(job_id: str):
    path=ROOT/job_id/"closetape_clips.zip"
    if not path.is_file(): raise HTTPException(404,"Архив ещё не готов")
    return FileResponse(path,media_type="application/zip",filename="closetape_clips.zip")
