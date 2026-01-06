import ollama
import json
import re
import sys

def derive_emotion(context_text):
    """
    Derives emotional tags from ANY text using Streaming for progress.
    """
    if not context_text or len(context_text) < 10:
        return []

    print("\n🧠 Deriving Emotions (Llama 3)...", flush=True)
    print("PRG:Emotions:Analyzing Context...:100", flush=True)

    prompt = f"""
    Task: Analyze the following content and extract 5-8 precise emotional tags.
    
    INPUT CONTEXT:
    "{context_text[:4000]}"
    
    INSTRUCTIONS:
    1. Identify the core mood (e.g., "Melancholic", "Energetic", "Romantic").
    2. Identify specific feelings (e.g., "Heartbreak", "Hopeful", "Aggressive").
    3. Output ONLY a JSON list of strings.
    4. Do not output broad genres like "Pop" or "Rock". Focus on EMOTION.
    
    OUTPUT FORMAT:
    ["Emotion1", "Emotion2", "Emotion3"]
    """
    
    try:
        # STREAMING RESPONSE
        stream = ollama.chat(model='llama3', messages=[
            {'role': 'system', 'content': 'You are an emotional analysis AI. Output JSON only.'},
            {'role': 'user', 'content': prompt}
        ], stream=True)
        
        full_content = ""
        token_count = 0
        ESTIMATED_TOKENS = 50 # JSON list is short

        print("PRG:Emotions:0:100", flush=True)

        for chunk in stream:
            content = chunk['message']['content']
            full_content += content
            token_count += 1
            
            if token_count % 2 == 0: # Update frequently since response is short
                percent = min((token_count / ESTIMATED_TOKENS) * 100, 99)
                print(f"PRG:Emotions:{int(percent)}:100", flush=True)
        
        print("PRG:Emotions:100:100", flush=True)
        
        # Robust Parsing
        match = re.search(r'\[.*\]', full_content, re.DOTALL)
        if match:
            tags = json.loads(match.group(0))
            # Clean tags
            tags = [t.strip().lower() for t in tags if isinstance(t, str)]
            print(f"   ❤️ Emotions: {tags}", flush=True)
            return tags
            
        return []

    except Exception as e:
        print(f"   ⚠️ Emotion Derivation Failed: {e}", flush=True)
        return []