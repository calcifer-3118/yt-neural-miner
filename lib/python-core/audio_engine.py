import whisper
import ollama
import torch
import gc
import re
import os
import sys

# STRICTER Filters
HALLUCINATION_PHRASES = [
    "subscribe to", "subscribe channel", "like and share", 
    "comment below", "thanks for watching", "copyright", 
    "all rights reserved", "follow us on", "press the bell icon",
    "prastutra" 
]

def clean_hallucinations(text):
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        lower_line = line.lower()
        is_spam = any(phrase in lower_line for phrase in HALLUCINATION_PHRASES)
        is_too_short = len(line.strip()) < 2
        
        if len(cleaned_lines) > 0 and line == cleaned_lines[-1]:
            continue

        if not is_spam and not is_too_short:
            cleaned_lines.append(line)
            
    return "\n".join(cleaned_lines)

def detect_language_smartly(model, audio, device):
    """Scans the MIDDLE of the audio to find the true language."""
    print("PRG:Audio:Scanning Language...:100", flush=True)
    audio_data = whisper.load_audio(audio)
    duration = len(audio_data) / 16000
    
    check_point = duration * 0.5
    start_sample = int(check_point * 16000)
    end_sample = start_sample + (30 * 16000)
    if end_sample > len(audio_data): end_sample = len(audio_data)
    
    segment = audio_data[start_sample:end_sample]
    segment = whisper.pad_or_trim(segment)
    
    n_mels = model.dims.n_mels 
    mel = whisper.log_mel_spectrogram(segment, n_mels=n_mels).to(model.device)
    
    _, probs = model.detect_language(mel)
    best_lang = max(probs, key=probs.get)
    
    print(f"   🔍 Scouted Middle of Song: Detected '{best_lang}'", flush=True)
    return best_lang

def process_audio(audio_path):
    # ==========================================
    # PHASE 1: TRANSCRIPTION (Whisper Only)
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    custom_cache = os.getenv("MINER_CACHE_ROOT", None)
    
    print(f"PRG:Audio:Loading Whisper ({DEVICE})...:100", flush=True)
    print(f"\n⏳ Loading Whisper on {DEVICE}...", flush=True)
    
    if custom_cache:
        print(f"   📂 Loading Whisper models from: {custom_cache}", flush=True)
        whisper_model = whisper.load_model("large-v3", device=DEVICE, download_root=custom_cache)
    else:
        whisper_model = whisper.load_model("large-v3", device=DEVICE)

    transcript_text = ""
    locked_language = "en"

    try:
        print(f"👂 Listening to {audio_path}...", flush=True)

        # 1. Detect language
        locked_language = detect_language_smartly(whisper_model, audio_path, DEVICE)

        # 2. Transcribe
        # Whisper is blocking, so we set a static status message
        print(f"PRG:Audio:Transcribing ({locked_language})...:100", flush=True)
        print(f"   📝 Transcribing FULL AUDIO with locked language: [{locked_language}]...", flush=True)
        
        result = whisper_model.transcribe(
            audio_path,
            language=locked_language,
            task="transcribe",
            beam_size=5,
            best_of=5,
            temperature=0.0,
            initial_prompt="Please transcribe all dialogues and lyrics of the song, including any spoken parts or narratives, without repeating words.", 
            fp16=False,
            condition_on_previous_text=False 
        )
        transcript_text = clean_hallucinations(result['text'].strip())
        print(f"   Raw Transcript ({locked_language}):\n{transcript_text[:100]}...", flush=True)

    finally:
        # 3. UNLOAD WHISPER
        print("PRG:Audio:Cleaning VRAM...:100", flush=True)
        print("   🧹 Unloading Whisper to make room for Llama 3...", flush=True)
        del whisper_model
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ==========================================
    # PHASE 2: ROMANIZATION (Llama 3 Only)
    # ==========================================
    final_text = transcript_text
    latin_langs = ['en', 'es', 'fr', 'de', 'it', 'pt', 'nl']
    
    if locked_language not in latin_langs and transcript_text:
        print(f"   🔠 Romanizing {locked_language} script using Llama 3...", flush=True)
        print("PRG:Audio:Romanizing (Llama 3)...:100", flush=True)
        
        prompt = f"""
        Task: Transliterate this {locked_language} text into standard Roman/Latin script (Hinglish style).
        
        STRICT RULES:
        1. Use SIMPLE English letters only (A-Z). 
        2. NO diacritics or accents (DO NOT use: ā, ī, ū, ḍ, ṁ, ñ, ṣ). 
        3. Use 'aa' for long 'a', 'ee' for long 'i', 'oo' for long 'u'.
        4. Example: Write "Zaroorat" instead of "Jarūrata". Write "Mein" instead of "Mẽ". 
        5. Output ONLY the transliterated text.
        
        Input:
        "{transcript_text}"
        """
        
        try:
            # STREAMING ROMANIZATION
            stream = ollama.chat(model='llama3', messages=[ 
                {'role': 'system', 'content': 'You are a colloquial transliterator. You hate special characters.'}, 
                {'role': 'user', 'content': prompt}
            ], stream=True)
            
            full_content = ""
            token_count = 0
            EST_TOKENS = len(transcript_text.split()) * 2 # Rough estimate
            
            print("PRG:Audio:0:100", flush=True) # Reset bar for romanization phase

            for chunk in stream:
                content = chunk['message']['content']
                full_content += content
                token_count += 1
                
                # Update bar every 5 tokens
                if token_count % 5 == 0:
                    percent = min((token_count / EST_TOKENS) * 100, 99)
                    print(f"PRG:Audio:{int(percent)}:100", flush=True)

            final_text = full_content
        except Exception as e:
            print(f"   ⚠️ Romanization failed: {e}", flush=True)
            print("   (Using original script as fallback)", flush=True)

    print("PRG:Audio:100:100", flush=True)
    return final_text