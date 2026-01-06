import ollama
import json
import re
import ast
import sys
import requests

def robust_json_parse(text):
    # Try to find JSON object in text
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match: return {}
    clean_text = match.group(0)
    try: return json.loads(clean_text)
    except: pass
    try: return ast.literal_eval(clean_text)
    except: pass
    try:
        # Fix common JSON errors from LLMs
        fixed_text = clean_text.replace("'", '"').replace('None', 'null').replace('True', 'true').replace('False', 'false')
        return json.loads(fixed_text)
    except: return {}

def fallback_metadata(title, description):
    """Returns the CORRECT schema if AI fails."""
    return {
        "movie": "Unknown",
        "singers": ["Unknown"],
        "cast": [],
        "language": "Unknown",
        "country": "Unknown",
        "musicDirector": "Unknown",
        "lyricist": "Unknown",
        "officialLyrics": description[:1000] if description else "", 
        "Summary": f"Auto-generated summary for {title}"
    }

def check_ollama_status():
    try:
        response = requests.get("http://localhost:11434/", timeout=2)
        return response.status_code == 200
    except:
        return False

def extract_metadata_smartly(title, description):
    print("\n🧠 Analyzing Metadata...", flush=True)
    
    # 1. Connection Check
    print("PRG:Metadata:Checking Ollama...:100", flush=True)
    if not check_ollama_status():
        print("   ⚠️ Ollama not reachable. Using fallback.", flush=True)
        return fallback_metadata(title, description)

    # 2. STRICT PROMPT
    prompt = f"""
    Task: Extract structured metadata from this YouTube Video.
    
    INPUT:
    Title: "{title}"
    Description: "{description}"
    
    INSTRUCTIONS:
    1. "singers": List main Vocalists.
    2. "movie": Movie/Album name.
    3. "cast": List of Actors.
    4. "language": Main language code (e.g., "hi", "en", "es").
    5. "country": Country of origin (e.g., "India", "USA").
    6. "musicDirector": Composer/Music Director.
    7. "lyricist": Song Writer/Lyricist.
    8. "officialLyrics": THE COMPLETE LYRICS. Do not skip lines. Do not use "...". Do not summarize. If lyrics are not in description, output "Not Available".
    9. "Summary": Detailed summary/synopsis.
    
    OUTPUT FORMAT (JSON ONLY):
    {{
      "movie": "String",
      "singers": ["Name"],
      "cast": ["Name"],
      "language": "String",
      "country": "String",
      "musicDirector": "String",
      "lyricist": "String",
      "officialLyrics": "String",
      "Summary": "String"
    }}
    """
    
    try:
        print("PRG:Metadata:AI Generating...:100", flush=True)
        
        # STREAMING RESPONSE
        stream = ollama.chat(model='llama3', messages=[
            {'role': 'system', 'content': 'You are a JSON metadata extractor. Output valid JSON only.'},
            {'role': 'user', 'content': prompt}
        ], stream=True)
        
        full_content = ""
        token_count = 0
        ESTIMATED_TOKENS = 500 # Slightly higher for full lyrics

        for chunk in stream:
            content = chunk['message']['content']
            full_content += content
            token_count += 1
            
            # Update bar every 5 tokens to show life
            if token_count % 5 == 0:
                percent = min((token_count / ESTIMATED_TOKENS) * 100, 99)
                print(f"PRG:Metadata:{int(percent)}:100", flush=True)

        print("PRG:Metadata:100:100", flush=True)
        
        parsed = robust_json_parse(full_content)
        if not parsed: raise Exception("Empty JSON")
        return parsed

    except Exception as e:
        print(f"   ⚠️ Metadata Error: {e}", flush=True)
        return fallback_metadata(title, description)