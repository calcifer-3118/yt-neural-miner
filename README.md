# Neural Miner

Neural Miner is an advanced, multi-modal AI pipeline designed to mine deep contextual understanding from YouTube videos. It orchestrates a suite of neural networks to extract metadata, transcribe audio, analyze visual storytelling, and derive emotional context—all synced to a structured database.

---

## Features

### Metadata Engine

- Extracts rich metadata including Title, Duration, Cast, Singers, and Summaries.
- Uses Llama 3 (via Ollama) to intelligently parse and structure unstructured video descriptions.
- Auto-corrects missing or malformed fields.

### Audio Engine (Whisper + Romanization)

- **Transcription:** High-accuracy speech-to-text using OpenAI's Whisper.
- **Romanization:** Automatically detects non-English segments (e.g., Hindi, Spanish) and converts them to Romanized Text (colloquial spelling) using Llama 3 for better searchability.
- **Noise Filtering:** Intelligently removes hallucinations and spam phrases (e.g., "Subscribe now").

### Video Engine (Vision Language Model)

- Uses Qwen2-VL-2B-Instruct to "watch" the video.
- Extracts key visual frames and generates a detailed Visual Narrative describing scenery, lighting, actions, and character interactions.

### Emotion Engine

- Analyzes the combined context of Lyrics, Visuals, and Metadata.
- Derives precise Emotional Tags (e.g., Melancholic, Energetic, Romantic) to categorize content by mood.

### Database Sync

- Seamlessly pushes all extracted data to a PostgreSQL database.
- Smart conflict handling: Updates existing records without overwriting critical IDs.
- Vector-ready: Generates embeddings for semantic search (using BAAI/bge-m3).

---

## Prerequisites

Before installing, ensure you have the following dependencies set up:

1. Node.js (v16+)
2. Python (v3.10+) with pip.
3. FFmpeg installed and added to your system PATH.
4. Ollama running locally with the required models:

```bash
ollama pull llama3
ollama pull qwen2vl
```

---

## Installation

Install the package globally via npm:

```bash
npm install -g yt-neural-miner
```

---

## Usage

Neural Miner provides a robust CLI with two main modes: Run and Sync.

### 1. Run Pipeline

Downloads the video and runs the selected analysis engines.

```bash
miner run "[https://www.youtube.com/watch?v=VIDEO_ID](https://www.youtube.com/watch?v=VIDEO_ID)"
```

**Options:**

| Option          | Description                                                               | Default     |
| :-------------- | :------------------------------------------------------------------------ | :---------- |
| `-p, --process` | Select specific engines (`metadata`, `audio`, `video`, `emotions`, `all`) | `all`       |
| `--mode`        | Storage mode (`local` or `db`)                                            | Interactive |
| `--keep`        | Keep local files after DB upload                                          | `false`     |
| `--cookies`     | Path to `cookies.txt` for restricted videos                               | `null`      |

**Example:**

```bash
# Run only Audio & Metadata, save locally
miner run "[https://youtu.be/xyz](https://youtu.be/xyz)" -s audio metadata --mode local
```

### 2. Sync Existing Data

If you have processed videos locally and want to push the cached data to your database later.

```bash
miner sync "[https://www.youtube.com/watch?v=VIDEO_ID](https://www.youtube.com/watch?v=VIDEO_ID)" --db "postgresql://user:pass@localhost:5432/mydb"
```

---

## Output Structure

When running in local mode, artifacts are organized by Video ID:

```text
output/
└── <video_id>/
    ├── video.mp4            # Source Video File
    ├── audio.mp3            # Extracted Audio Track
    ├── metadata.json        # Structured Metadata (JSON)
    ├── transcript.txt       # Cleaned & Romanized Transcript
    ├── video_narrative.txt  # Frame-by-frame Visual Analysis
    └── emotions.json        # List of Derived Emotional Tags
```

---

## Configuration

You can provide your Database URL in three ways (prioritized order):

1. **CLI Flag:**

   ```bash
   miner run URL --db "postgresql://..."
   ```

2. **Interactive Prompt:**
   The CLI will ask you for the URL if it is missing.

3. **Environment Variable:**
   Set `MINER_DB_URL` in your system environment or a `.env` file in the execution directory.

---

## Author

**Dheer Jain**

- GitHub: [calcifer-3118](https://github.com/calcifer-3118)

---

## License

This project is licensed under the ISC License.
