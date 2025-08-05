# 🏭 Factory Video Analyzer

A powerful video analysis tool that summarizes manufacturing processes from factory footage using computer vision and large language models (LLMs). It integrates video frame sampling, image-based reasoning (via Ollama), and a clean Streamlit frontend.

---

## 🔧 Features

✅ Extracts evenly sampled video frames  
✅ Describes visual actions in each frame using LLMs  
✅ Groups frames into chunks and summarizes each chunk  
✅ Detects repetitive manufacturing **cycles**  
✅ Generates step-by-step summaries with annotated images  
✅ Interactive frontend built with Streamlit

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/gk-gokul/factory-video-analyzer.git
cd factory-video-analyzer
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Make Sure Ollama is Running

Install and run Ollama locally:  
➡️ https://ollama.com

Make sure your models (like `qwen2.5vl` or `llava`) are downloaded and accessible at:
```
http://localhost:11434
```

### 4. Start the App

```bash
streamlit run frontend.py
```

Use the sidebar to upload an `.mp4` factory video and choose a vision model (like `llava` or `qwen2.5vl`).

---

## 🧠 How It Works

| Component       | Description |
|----------------|-------------|
| `main_script.py` | Samples frames, generates descriptions, chunks them, and detects cycles |
| `frontend.py`   | Streamlit UI for video upload, model selection, and displaying results |
| Ollama API     | Used to generate LLM-based image descriptions |
| SSIM           | Used to detect repeated cycles in video frames |

---

## 📂 Output Structure

| File/Folder             | Purpose |
|-------------------------|---------|
| `process_summary.json`  | Final output summary (text + frame map) |
| `frame_descriptions.txt`| Raw per-frame LLM descriptions |
| `Cycle Frames/`         | Saved cycle step images |
| `Frames Outputted/`     | Chunk-based frame images |

> These files are auto-generated after successful analysis.

---

## 📸 Sample Output (Optional)

You can add screenshots here later to showcase:
- The Streamlit interface
- Summary results
- Annotated cycle steps

---

## 🛠️ Requirements

- Python 3.8+
- Ollama installed and running locally
- At least one vision-capable model in Ollama (e.g. `qwen2.5vl`, `llava`)

---
