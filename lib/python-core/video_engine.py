import os
import sys
import gc
import shutil
import math
import time
from pathlib import Path

# --- SAFE IMPORTS ---
try:
    import torch
    import cv2
    from PIL import Image
    import psutil
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
except ImportError as e:
    print(f"\n❌ CRITICAL ERROR: Missing dependency: {e}", flush=True)
    def analyze_full_video(path): return f"Error: Missing dependency {e}"
    sys.modules[__name__].analyze_full_video = analyze_full_video

# --- CONFIGURATION ---
def get_system_profile():
    profile = {
        "device": "cpu",
        "dtype": torch.float32,
        "max_pixels": 640 * 480,
        "chunk_duration": 45, # Longer chunks for better continuity since video is short
        "max_frames_per_batch": 8,
        "model_id": "Qwen/Qwen2-VL-2B-Instruct",
        "desc": "Low-End (CPU)"
    }

    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        profile["device"] = "cuda"
        profile["dtype"] = torch.bfloat16
        
        # Since video is < 7 mins, we can be aggressive with quality even on mid-range cards
        if vram_gb >= 22: # RTX A6000 / 3090 / 4090
            profile["model_id"] = "Qwen/Qwen2-VL-7B-Instruct"
            profile["max_pixels"] = 1280 * 1280
            profile["chunk_duration"] = 60 
            profile["max_frames_per_batch"] = 48
            profile["desc"] = f"Ultra-High-End (GPU {int(vram_gb)}GB)"
        elif vram_gb >= 15: 
            profile["model_id"] = "Qwen/Qwen2-VL-7B-Instruct"
            profile["max_pixels"] = 1024 * 768
            profile["chunk_duration"] = 60
            profile["max_frames_per_batch"] = 24
            profile["desc"] = f"High-End (GPU {int(vram_gb)}GB)"
        else: 
            profile["model_id"] = "Qwen/Qwen2-VL-2B-Instruct"
            profile["max_pixels"] = 720 * 480
            profile["chunk_duration"] = 45
            profile["max_frames_per_batch"] = 12
            profile["desc"] = f"Mid-Range (GPU {int(vram_gb)}GB)"
    
    return profile

def get_sampling_fps(duration_seconds):
    # For < 7 mins, we want high detail
    if duration_seconds < 60: return 2.0
    return 1.0 # 1 frame every second is excellent for narrative

def open_video_robust(video_path):
    abs_path = str(Path(video_path).resolve())
    backends = [
        (cv2.CAP_FFMPEG, "FFMPEG"),
        (cv2.CAP_MSMF, "MediaFoundation"),
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_ANY, "Auto")
    ]
    for backend_id, name in backends:
        cap = cv2.VideoCapture(abs_path, backend_id)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                print(f"   ✅ Opened video using backend: {name}", flush=True)
                return cap
            cap.release()
    return None

def analyze_full_video(video_path):
    video_path = Path(video_path)
    frames_root = video_path.parent / "frames_cache"
    
    print(f"\n👁️ Watching {video_path.name}...", flush=True)
    if not video_path.exists(): return "Analysis Failed: Video file missing."

    # 1. SETUP
    if frames_root.exists(): shutil.rmtree(frames_root)
    frames_root.mkdir(exist_ok=True)

    profile = get_system_profile()
    print(f" ⚙️  Hardware: {profile['desc']} | Model: {profile['model_id']}", flush=True)

    # 2. OPEN VIDEO
    cap = open_video_robust(video_path)
    if cap is None: return "Analysis Failed: Video Corrupted."

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = total_frames / fps
    target_fps = get_sampling_fps(total_duration)
    
    print(f" 📹 Stats: {int(total_duration)}s | {total_frames} frames | Target: {target_fps} FPS", flush=True)

    # ==========================================
    # PHASE 1: EXTRACTION
    # ==========================================
    print("\n📥 PHASE 1: Extracting Frames...", flush=True)
    
    chunk_dur = profile["chunk_duration"]
    num_chunks = math.ceil(total_duration / chunk_dur)
    chunks_data = [] 

    for chunk_idx in range(num_chunks):
        start_time = chunk_idx * chunk_dur
        end_time = min((chunk_idx + 1) * chunk_dur, total_duration)
        
        chunk_folder = frames_root / f"chunk_{chunk_idx:03d}"
        chunk_folder.mkdir(exist_ok=True)
        
        actual_chunk_len = end_time - start_time
        needed_frames = int(actual_chunk_len * target_fps)
        final_frame_count = max(1, min(needed_frames, profile["max_frames_per_batch"]))
        step_sec = actual_chunk_len / final_frame_count
        
        extracted_count = 0
        for k in range(final_frame_count):
            sec_mark = start_time + (k * step_sec)
            frame_no = int(sec_mark * fps)
            if frame_no >= total_frames: break

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            if ret:
                h, w, _ = frame.shape
                # Resize early to save space
                if profile["max_pixels"] < (w * h):
                    scale = math.sqrt(profile["max_pixels"] / (w * h))
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                
                cv2.imwrite(str(chunk_folder / f"frame_{k:03d}.jpg"), frame)
                extracted_count += 1

        if extracted_count > 0:
            chunks_data.append({
                "start": start_time,
                "end": end_time,
                "path": chunk_folder
            })
            print(f"   ✅ Chunk {chunk_idx+1}/{num_chunks}: Extracted {extracted_count} frames.", flush=True)

    cap.release()
    if not chunks_data: return "Analysis Failed: No frames found."

    # ==========================================
    # PHASE 2: MODEL LOADING
    # ==========================================
    print("\n🧠 PHASE 2: Loading AI Model...", flush=True)
    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            profile["model_id"], 
            torch_dtype=profile["dtype"], 
            device_map="auto" if profile["device"] == "cuda" else None,
            low_cpu_mem_usage=True
        )
        if profile["device"] == "cpu": model = model.to("cpu")
        processor = AutoProcessor.from_pretrained(profile["model_id"])
    except Exception as e:
        return f"AI Model Failed: {e}"

    # ==========================================
    # PHASE 3: SCENE ANALYSIS
    # ==========================================
    print("\n📝 PHASE 3: Analyzing Scenes...", flush=True)
    full_narrative_log = []
    
    for i, chunk in enumerate(chunks_data):
        prog = int((i / len(chunks_data)) * 100)
        print(f"PRG:Video Analysis:{prog}:100", flush=True)
        
        chunk_frames = []
        for f in sorted(list(chunk["path"].glob("*.jpg"))):
            chunk_frames.append(Image.open(f))
            
        if not chunk_frames: continue

        # Context passing for continuity
        context_str = ""
        if full_narrative_log:
            # Pass the last 2 sentences of context
            context_str = f"PREVIOUSLY: {full_narrative_log[-1][-200:]}."

        prompt = f"""
        {context_str}
        Describe this video segment ({int(chunk['start'])}s to {int(chunk['end'])}s).
        Focus on:
        1. Action flow (What is happening?)
        2. Visual atmosphere (Lighting, Colors).
        3. Character interactions and emotions.
        """

        try:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "video", "video": chunk_frames, "fps": 1.0},
                    {"type": "text", "text": prompt}
                ]
            }]
            
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(model.device)
            
            # Generous token limit for detailed description
            max_toks = 1024 if profile["device"] == "cuda" else 512
            
            generated_ids = model.generate(**inputs, max_new_tokens=max_toks)
            output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            chunk_desc = output_text.split("assistant")[-1].strip()
            
            print(f"   ✨ Analyzed Chunk {i+1}", flush=True)
            timestamped_desc = f"[Time: {int(chunk['start'])}s-{int(chunk['end'])}s] {chunk_desc}"
            full_narrative_log.append(timestamped_desc)

        except Exception as e:
            print(f"   ⚠️ Chunk {i+1} Failed: {e}", flush=True)

        # Cleanup
        del chunk_frames, inputs, generated_ids
        if 'image_inputs' in locals(): del image_inputs
        if 'video_inputs' in locals(): del video_inputs
        gc.collect()
        if profile["device"] == "cuda": torch.cuda.empty_cache()
        shutil.rmtree(chunk["path"])

    # ==========================================
    # PHASE 4: FINAL STORY SUMMARY
    # ==========================================
    print("\n📖 PHASE 4: Writing Master Story...", flush=True)
    print("PRG:Video Analysis:Generating Summary...:95:100", flush=True)

    if not full_narrative_log: return "No narrative generated."

    # Since video is < 7 mins, we can feed ALL logs at once
    combined_logs = "\n".join(full_narrative_log)
    
    summary_prompt = f"""
    Read the following scene logs from a video and write a COMPLETE STORY SUMMARY.
    
    INSTRUCTIONS:
    1. Write a single, cohesive narrative (like a short story).
    2. Ignore timestamps. Connect the events logically.
    3. Describe the beginning, the middle conflict/action, and the ending resolution.
    4. Capture the mood and character motivations.
    
    SCENE LOGS:
    {combined_logs}
    
    COMPLETE STORY:
    """
    
    try:
        messages = [{"role": "user", "content": [{"type": "text", "text": summary_prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], padding=True, return_tensors="pt").to(model.device)
        
        # Max tokens ensuring completion
        generated_ids = model.generate(**inputs, max_new_tokens=2048)
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        master_summary = output_text.split("assistant")[-1].strip()
        
    except Exception as e:
        master_summary = f"Summary generation failed: {e}"

    # Cleanup
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    try: shutil.rmtree(frames_root)
    except: pass

    # FINAL FORMATTED OUTPUT
    final_output = f"""
=== MASTER STORY SUMMARY ===
{master_summary}

=== SCENE-BY-SCENE LOG ===
{combined_logs}
"""
    print("PRG:Video Analysis:100:100", flush=True)
    return final_output