import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
import gc
import cv2
import shutil
from pathlib import Path
from PIL import Image

def analyze_full_video(video_path):
    print(f"\n👁️ Watching {video_path}...", flush=True)
    video_path = Path(video_path)
    frames_dir = video_path.parent / "frames"
    
    # 1. SETUP OUTPUT FOLDER
    if frames_dir.exists(): shutil.rmtree(frames_dir)
    frames_dir.mkdir(exist_ok=True)

    def send_prg(curr, tot):
        print(f"PRG:Video Analysis:{curr}:{tot}", flush=True)

    # 2. LOAD MODEL
    try:
        has_cuda = torch.cuda.is_available()
        # Qwen2-VL-2B is efficient and capable for this task
        MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct" 
        print(f"   ⏳ Loading {MODEL_ID}...", flush=True)
        
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_ID, 
            torch_dtype=torch.bfloat16 if has_cuda else torch.float32, 
            device_map="auto" if has_cuda else "cpu",
            low_cpu_mem_usage=True
        )
        processor = AutoProcessor.from_pretrained(MODEL_ID)
    except Exception as e:
        print(f"   ❌ Model Load Error: {e}", flush=True)
        return ""

    # 3. SMART FRAME EXTRACTION
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"   📹 Extracting frames (~{int(duration)}s video)...", flush=True)
    
    # Sampling Logic:
    # - Short videos (<1m): High density (2 fps) to catch quick actions.
    # - Medium videos (1-5m): Standard density (1 fps) for narrative.
    # - Long videos (>5m): Low density (0.5 fps) to fit context window.
    interval = int(fps) 
    if duration < 60: interval = int(fps / 2)
    elif duration > 300: interval = int(fps * 2)
    
    target_indices = range(0, total_frames, interval)
    total_targets = len(target_indices)
    
    extracted_frames_pil = [] # Keep in memory for AI
    count = 0

    for i in target_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            # Convert BGR (OpenCV) to RGB (PIL)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            extracted_frames_pil.append(pil_img)
            
            # Save to Disk (Requirement)
            frame_path = frames_dir / f"frame_{count:04d}.jpg"
            pil_img.save(frame_path)
            
            count += 1
        
        if count % 10 == 0:
            send_prg(count, total_targets)

    cap.release()
    print(f"   ✅ Captured {len(extracted_frames_pil)} frames.", flush=True)
    
    # 4. GENERATE NARRATIVE
    send_prg(95, 100) # Analyzing phase
    print("   🧠 Analyzing visual story...", flush=True)
    
    # Semantic Search Optimized Prompt
    prompt = """You are an expert video analyst for a semantic search engine. 
Describe this video in extreme detail. Focus on:
1. The chronological STORY (Start, Middle, End). Mention any plot twists (e.g., "the boyfriend was an undercover cop").
2. Specific SCENES & ACTIONS (e.g., "man painting his wife on snow", "girl riding a bike", "running by a train").
3. VISUAL DETAILS (Scenery, colors, lighting, objects).
4. Characters and their emotions.
Provide a comprehensive summary."""

    messages = [{
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": extracted_frames_pil, # Pass list of PIL images
                "fps": 1.0, 
            },
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
    
    # Generate Description
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    description = output_text.split("assistant")[-1].strip()
    
    send_prg(100, 100)
    
    # 5. CLEANUP
    try:
        print(f"   🧹 Cleaning up frame cache...", flush=True)
        shutil.rmtree(frames_dir)
    except Exception as e:
        print(f"   ⚠️ Frame cleanup failed: {e}", flush=True)

    # Free Memory
    del model, processor, inputs, video_inputs, extracted_frames_pil
    gc.collect()
    if has_cuda: torch.cuda.empty_cache()

    return description