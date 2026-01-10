from anyio import Path
import whisper
import ollama
import torch
import gc
import re
import os
import sys
from difflib import SequenceMatcher

# STRICTER Filters for Audio Hallucinations
HALLUCINATION_PHRASES = [
    "subscribe to", "subscribe channel", "like and share", 
    "comment below", "thanks for watching", "copyright", 
    "all rights reserved", "follow us on", "press the bell icon",
    "prastutra", "video by", "audio by", "Praastuti", "praastuti", "praastuti"
]

def is_similar(a, b, threshold=0.85):
    """Checks if two lines are nearly identical."""
    return SequenceMatcher(None, a, b).ratio() > threshold

def clean_hallucinations(text):
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        lower_line = line.lower()
        
        # 2. Filter known spam phrases
        if any(phrase in lower_line for phrase in HALLUCINATION_PHRASES):
            continue
            
        # 3. Filter Repetitive Loops (e.g., "na kar de na kar de")
        if len(line) > 10 and len(set(line.split())) < 4:
            continue

        # 4. Filter Adjacent Duplicates
        if len(cleaned_lines) > 0:
            if is_similar(line, cleaned_lines[-1]):
                continue
                
        cleaned_lines.append(line)
            
    return "\n".join(cleaned_lines)

def clean_llm_response(text):
    """Removes conversational filler from LLM output."""
    patterns = [
        r"^here is the transliterated text[:\s]*",
        r"^here is the transliteration[:\s]*",
        r"^here are the lyrics[:\s]*",
        r"^sure,? here is .*[:\s]*",
        r"^transliteration[:\s]*",
        r"^romanized text[:\s]*",
        r"^output[:\s]*"
    ]
    for p in patterns:
        text = re.sub(p, "", text, flags=re.IGNORECASE).strip()
    return text

def detect_language_smartly(model, audio, device):
    """Scans the MIDDLE of the audio to find the true language."""
    print("PRG:Audio:Scanning Language...:100", flush=True)
    
    audio_data = whisper.load_audio(audio)
    duration = len(audio_data) / 16000
    
    # Check middle of the song
    check_point = duration * 0.5
    start_sample = int(check_point * 16000)
    end_sample = start_sample + (30 * 16000)
    
    if end_sample > len(audio_data): 
        start_sample = 0
        end_sample = min(len(audio_data), 30 * 16000)
    
    segment = audio_data[start_sample:end_sample]
    segment = whisper.pad_or_trim(segment)
    
    n_mels = model.dims.n_mels 
    mel = whisper.log_mel_spectrogram(segment, n_mels=n_mels).to(model.device)
    
    _, probs = model.detect_language(mel)
    best_lang = max(probs, key=probs.get)
    
    print(f"   🔍 Detected Language: '{best_lang}'", flush=True)
    return best_lang

def process_audio(audio_path):
    # ==========================================
    # PHASE 1: TRANSCRIPTION (Whisper Only)
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    custom_cache = os.getenv("MINER_CACHE_ROOT", None)
    
    print(f"PRG:Audio:Loading Whisper ({DEVICE})...:100", flush=True)
    
    try:
        if custom_cache:
            whisper_model = whisper.load_model("large-v3", device=DEVICE, download_root=custom_cache)
        else:
            whisper_model = whisper.load_model("large-v3", device=DEVICE)

        transcript_text = ""
        locked_language = "hi"

        print(f"👂 Listening to {Path(audio_path).name}...", flush=True)

        # 1. Detect language
        locked_language = detect_language_smartly(whisper_model, audio_path, DEVICE)

        # 2. Transcribe
        print(f"PRG:Audio:Transcribing ({locked_language})...:100", flush=True)
        
        result = whisper_model.transcribe(
            audio_path,
            language=locked_language,
            task="transcribe",
            beam_size=5,
            best_of=5,
            temperature=0.0,
            compression_ratio_threshold=2.4, # Fix loops
            logprob_threshold=-1.0,          # Fix bad audio
            no_speech_threshold=0.6,
            condition_on_previous_text=False
        )
        
        raw_text = result['text'].strip()
        transcript_text = clean_hallucinations(raw_text)
        
        preview = transcript_text[:100].replace('\n', ' ')
        print(f"   📝 Transcript Preview: {preview}...", flush=True)

    except Exception as e:
        print(f"❌ Audio Processing Error: {e}", flush=True)
        return ""
    
    finally:
        # 3. UNLOAD WHISPER
        print("PRG:Audio:Cleaning VRAM...:100", flush=True)
        if 'whisper_model' in locals(): del whisper_model
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ==========================================
    # PHASE 2: ROMANIZATION (Llama 3 Only)
    # ==========================================
    final_text = transcript_text
    latin_langs = ['en', 'es', 'fr', 'de', 'it', 'pt', 'nl']
    
    if locked_language not in latin_langs and transcript_text:
        print(f"   🔠 Romanizing {locked_language} script...", flush=True)
        print("PRG:Audio:Romanizing (Llama 3)...:100", flush=True)
        
        prompt = f"""
        Task: Convert these {locked_language} lyrics to Roman/Latin script (Phonetic style).
        
        INPUT:
        "{transcript_text}"
        
        RULES:
        1. OUTPUT ONLY THE LYRICS. NO "Here is..." or "Transliteration:".
        2. Fix repetitions/loops.
        3. Use simple English letters (No diacritics).
        4. START DIRECTLY with the first word.
        """
        
        try:
            stream = ollama.chat(model='llama3', messages=[ 
                {'role': 'system', 'content': 'You are a raw data converter. You output ONLY the processed text. You are NOT a chat assistant.'}, 
                {'role': 'user', 'content': prompt}
            ], stream=True)
            
            full_content = ""
            token_count = 0
            EST_TOKENS = len(transcript_text.split()) * 1.5 
            
            print("PRG:Audio:0:100", flush=True) 

            for chunk in stream:
                content = chunk['message']['content']
                full_content += content
                token_count += 1
                
                if token_count % 10 == 0:
                    percent = min((token_count / EST_TOKENS) * 100, 99)
                    print(f"PRG:Audio:{int(percent)}:100", flush=True)

            # Apply cleanup filter
            final_text = clean_llm_response(full_content)
            
        except Exception as e:
            print(f"   ⚠️ Romanization failed: {e}. Using raw transcript.", flush=True)

    print("PRG:Audio:100:100", flush=True)
    return final_text