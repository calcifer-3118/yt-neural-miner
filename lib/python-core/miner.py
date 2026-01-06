import sys
import json
import os
import shutil
import argparse
import traceback
import threading
import multiprocessing
import subprocess
import urllib.parse
import time
from pathlib import Path
from dotenv import load_dotenv

# --- STABILITY SETTINGS ---
os.environ["TQDMDISABLE"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except: pass

sys.path.append(str(Path(__file__).resolve().parent))

# GLOBALS
USER_CWD = Path(os.getcwd())
OUTPUT_ROOT = USER_CWD / "output"
SKIP_CURRENT_STAGE = threading.Event()

# Load Env
env_path = USER_CWD / ".env"
if env_path.exists(): load_dotenv(dotenv_path=env_path)

# =========================
# HELPER: DIRECT WORKER
# =========================
def worker_wrapper(func, args, result_queue):
    """Executes engine. Prints DIRECTLY to stdout."""
    try:
        res = func(*args)
        result_queue.put(res)
    except Exception as e:
        print(f"❌ Engine Error: {e}", flush=True)
        result_queue.put(None)

def run_skippable_stage(stage_name, target_func, args):
    SKIP_CURRENT_STAGE.clear()
    
    result_q = multiprocessing.Queue()
    p = multiprocessing.Process(target=worker_wrapper, args=(target_func, args, result_q))
    p.start()
    
    while p.is_alive():
        if SKIP_CURRENT_STAGE.is_set():
            p.terminate()
            p.join()
            print("SKIP_ACK", flush=True)
            return None
        time.sleep(0.1)
    
    p.join()
    if not result_q.empty(): return result_q.get()
    return None

def input_listener():
    """Reads stdin for control commands."""
    while True:
        try:
            line = sys.stdin.readline()
            if not line: break
            if line.strip().lower() == "skip": 
                SKIP_CURRENT_STAGE.set()
        except: break

# =========================
# DOWNLOADER
# =========================
def get_yt_info(url, cookies_arg=None):
    import yt_dlp
    opts = {'quiet': True, 'nocheckcertificate': True}
    if cookies_arg and os.path.exists(cookies_arg): opts['cookiefile'] = cookies_arg
    try:
        with yt_dlp.YoutubeDL(opts) as ydl: return ydl.extract_info(url, download=False)
    except: return None

def download_source(url, paths, video_id, cookies_arg=None):
    # Check Exists
    if paths["video_file"].exists() and paths["video_file"].stat().st_size > 1024:
        print(f"PRG:Downloading:Local File Found:100", flush=True)
        if paths["metadata"].exists():
            try: return json.loads(paths["metadata"].read_text(encoding="utf-8"))
            except: pass
        info = get_yt_info(url, cookies_arg)
        return info if info else {"id": video_id, "title": f"Video {video_id}", "duration": 0}

    print("PRG:Downloading:Starting...:0", flush=True)
    
    import yt_dlp
    def progress_hook(d):
        if d['status'] == 'downloading':
            try:
                p = d.get('_percent_str', '0%').replace('%','')
                print(f"PRG:Downloading:{p}:100", flush=True)
            except: pass

    out_tmpl = str(paths["folder"] / "video.%(ext)s")
    ydl_opts = {
        'format': 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': out_tmpl,
        'noplaylist': True,
        'progress_hooks': [progress_hook],
        'quiet': True, 'no_warnings': True, 'nocheckcertificate': True,
        'postprocessors': [{'key': 'FFmpegVideoConvertor', 'preferedformat': 'mp4'}]
    }
    if cookies_arg and os.path.exists(cookies_arg): ydl_opts['cookiefile'] = cookies_arg

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            
        if not paths["video_file"].exists():
            for f in paths["folder"].glob("video.*"):
                if f.suffix != ".mp4": shutil.move(f, paths["video_file"]); break
        
        print("PRG:Audio Extraction:50:100", flush=True)
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(paths["video_file"]), 
            "-vn", "-acodec", "libmp3lame", str(paths["audio_file"])
        ], check=True, stdin=subprocess.DEVNULL)
        print("PRG:Audio Extraction:100:100", flush=True)
        return info
    except Exception as e:
        print(f"❌ Download Failed: {e}", flush=True)
        raise e

# =========================
# DB SYNC
# =========================
def push_to_db(meta, paths):
    db_url = os.getenv("MINER_DB_URL") or os.getenv("DATABASE_URL")
    if not db_url: return False
    if "?" in db_url: db_url = db_url.split("?")[0]

    print("PRG:DB Sync:Connecting...:10", flush=True)

    try:
        import psycopg2
        print("PRG:DB Sync:Loading AI Model...:20", flush=True)
        sys.stdout.flush()
        
        os.environ["CUDA_VISIBLE_DEVICES"] = "" 
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("BAAI/bge-m3", device="cpu")

        rich_meta = json.loads(paths["metadata"].read_text(encoding='utf-8')) if paths["metadata"].exists() else {}
        transcript = paths["transcript"].read_text(encoding='utf-8') if paths["transcript"].exists() else ""
        narrative = paths["narrative"].read_text(encoding='utf-8') if paths["narrative"].exists() else ""
        emotions = json.loads(paths["emotions"].read_text(encoding='utf-8')) if paths["emotions"].exists() else []
        
        final_title = rich_meta.get("title", meta.get("title", "Unknown"))
        final_duration = int(rich_meta.get("duration", meta.get("duration", 0)))
        final_id = rich_meta.get("id", meta.get("id"))
        final_summary = rich_meta.get("Summary", meta.get("description", ""))
        final_singers = rich_meta.get("singers", ["Unknown"])

        print("PRG:DB Sync:Generating Vectors...:50", flush=True)
        sys.stdout.flush()

        combined_text = f"Title: {final_title}\nSummary: {final_summary}\nVisuals: {narrative}\nLyrics: {transcript}"
        visual_vec = embedder.encode(narrative).tolist() if narrative else None
        transcript_vec = embedder.encode(transcript).tolist() if transcript else None
        combined_vec = embedder.encode(combined_text).tolist()

        conn = psycopg2.connect(db_url, connect_timeout=10)
        cur = conn.cursor()

        cur.execute("INSERT INTO \"Artist\" (name, \"createdAt\") VALUES (%s, NOW()) ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name RETURNING id", (final_singers[0],))
        artist_id = cur.fetchone()[0]

        cur.execute("""
            INSERT INTO "Song" ("title", "ytVideoId", "durationSeconds", "album", "movie", "language", "country", "cast", "musicDirector", "lyricist", "officialLyrics", "singers", "summary", "artistId", "createdAt")
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT ("ytVideoId") DO UPDATE SET "title" = EXCLUDED."title" RETURNING id
        """, (
            final_title, final_id, final_duration,
            rich_meta.get("movie"), rich_meta.get("movie"),
            rich_meta.get("language"), rich_meta.get("country"),
            rich_meta.get("cast", []), rich_meta.get("musicDirector"),
            rich_meta.get("lyricist"), rich_meta.get("officialLyrics"),
            final_singers, final_summary, artist_id
        ))
        song_id = cur.fetchone()[0]

        cur.execute("""
            INSERT INTO "SongContext" ("songId", "visualDescription", "transcript", "emotionalTags", "visualVector", "transcriptVector", "combinedVector", "updatedAt")
            VALUES (%s, %s, %s, %s, %s::vector, %s::vector, %s::vector, NOW())
            ON CONFLICT ("songId") DO UPDATE SET "updatedAt" = NOW()
        """, (
            song_id, narrative, transcript, emotions,
            str(visual_vec) if visual_vec else None,
            str(transcript_vec) if transcript_vec else None,
            str(combined_vec)
        ))

        conn.commit()
        print("PRG:DB Sync:Complete:100", flush=True)
        return True

    except Exception as e:
        print(f"❌ DB Sync Error: {e}", flush=True)
        return False

# =========================
# MAIN
# =========================
def main():
    # Only print PRG signals, no text logs
    if 'MINER_DB_URL' in os.environ:
        os.environ['DATABASE_URL'] = os.environ['MINER_DB_URL']

    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--process", action="append")
    parser.add_argument("--mode", default="local")
    parser.add_argument("--cleanup", action="store_true")
    parser.add_argument("--models_dir")
    parser.add_argument("--cookies")
    parser.add_argument("--sync_only", action="store_true")
    parser.add_argument("--non-interactive", action="store_true")
    args = parser.parse_args()

    if args.models_dir:
        os.environ["HF_HOME"] = args.models_dir
        os.environ["MINER_CACHE_ROOT"] = args.models_dir

    if not args.non_interactive:
        t = threading.Thread(target=input_listener, daemon=True)
        t.start()

    try:
        vid_id = args.url.split("v=")[-1].split("&")[0]
        folder = OUTPUT_ROOT / vid_id
        folder.mkdir(parents=True, exist_ok=True)
        paths = {
            "folder": folder,
            "metadata": folder / "metadata.json",
            "transcript": folder / "transcript.txt",
            "narrative": folder / "video_narrative.txt",
            "emotions": folder / "emotions.json",
            "audio_file": folder / "audio.mp3",
            "video_file": folder / "video.mp4"
        }

        # --- SYNC MODE ---
        if args.sync_only:
            print("PRG:Sync:Checking Files:10", flush=True)
            meta = {"id": vid_id, "title": "Synced Video", "duration": 0} 
            if paths["metadata"].exists():
                try: meta.update(json.loads(paths["metadata"].read_text(encoding="utf-8")))
                except: pass
            
            if meta["title"] == "Synced Video":
                i = get_yt_info(args.url, args.cookies)
                if i: meta.update(i)

            if push_to_db(meta, paths):
                if args.cleanup: shutil.rmtree(folder)
            else:
                sys.exit(1)
            sys.exit(0)

        # --- RUN MODE ---
        meta = download_source(args.url, paths, vid_id, args.cookies)
        
        stages_list = args.process if args.process else ["all"]
        flat_stages = []
        for s in stages_list: flat_stages.extend(s.split(','))
        req_stages = set(flat_stages)
        if "all" in req_stages: req_stages = {"metadata", "audio", "video", "emotions"}

        # 1. METADATA
        if "metadata" in req_stages:
            if paths["metadata"].exists():
                print("PRG:Metadata:Cached:100", flush=True)
            else:
                print("PRG:Metadata:Initializing...:0", flush=True)
                from metadata_engine import extract_metadata_smartly
                res = run_skippable_stage("Metadata", extract_metadata_smartly, (meta["title"], meta.get("description", "")))
                if res: 
                    res["title"] = meta.get("title")
                    res["duration"] = meta.get("duration")
                    res["id"] = meta.get("id")
                    paths["metadata"].write_text(json.dumps(res, indent=2), encoding="utf-8")
                    print("PRG:Metadata:100:100", flush=True)

        # 2. AUDIO
        if "audio" in req_stages:
            if paths["transcript"].exists():
                print("PRG:Audio:Cached:100", flush=True)
            else:
                print("PRG:Audio:Initializing...:0", flush=True)
                from audio_engine import process_audio
                res = run_skippable_stage("Audio", process_audio, (str(paths["audio_file"]),))
                if res: 
                    paths["transcript"].write_text(res, encoding="utf-8")
                    print("PRG:Audio:100:100", flush=True)

        # 3. VIDEO
        if "video" in req_stages:
            if paths["narrative"].exists(): 
                print("PRG:Video:Cached:100", flush=True)
            else:
                print("PRG:Video:Initializing...:0", flush=True)
                from video_engine import analyze_full_video
                res = run_skippable_stage("Video", analyze_full_video, (str(paths["video_file"]),)) 
                if res: paths["narrative"].write_text(res, encoding="utf-8")

        # 4. EMOTIONS
        if "emotions" in req_stages:
            if paths["emotions"].exists(): 
                print("PRG:Emotions:Cached:100", flush=True)
            else:
                print("PRG:Emotions:Initializing...:0", flush=True)
                from emotion_engine import derive_emotion
                context = meta["title"]
                if paths["narrative"].exists(): context += paths["narrative"].read_text(encoding="utf-8")
                res = run_skippable_stage("Emotions", derive_emotion, (context,))
                if res: 
                    paths["emotions"].write_text(json.dumps(res, indent=2), encoding="utf-8")
                    print("PRG:Emotions:100:100", flush=True)

        if args.mode == "db":
            if push_to_db(meta, paths):
                if args.cleanup: shutil.rmtree(folder)
            else:
                sys.exit(1)

    except Exception as e:
        print(f"❌ Fatal Error: {e}", flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()