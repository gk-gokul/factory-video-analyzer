import streamlit as st
import os
import json
from PIL import Image
import tempfile

from main_script import (
    sample_evenly,
    get_summary_from_frames,
    ask_ollama,
    analyze_chunks,
    consolidate_chunk_descriptions,
    generate_cycle_steps_summary,
    save_output,
    TOTAL_SAMPLES,
)

# Constants
TOTAL_SAMPLES = 35
CHUNK_SIZE = 7
OUTPUT_FILE = "process_summary.json"
CYCLE_IMAGE_DIR = "Cycle Frames"

# Streamlit UI
st.set_page_config(page_title="Factory Video Analyzer", layout="wide")
st.title("Factory Video Analyzer")

# Sidebar settings
st.sidebar.header("Settings")
uploaded_video = st.sidebar.file_uploader("Upload a factory video", type=["mp4"])
models = [
    "qwen2.5vl:latest",
    "llava-llama3:8b",
    "llava:latest",
    "llama3.2-vision:latest",
    "llama3.2-vision:11b"
]
selected_model = st.sidebar.selectbox("Choose an LLM model", models)
calibrate_button = st.sidebar.button("Calibrate and Analyze")

# Start analysis
if calibrate_button and uploaded_video:
    temp_video_path = os.path.join(tempfile.gettempdir(), uploaded_video.name)
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_video.read())

    st.session_state["analysis_done"] = False

    with st.spinner("Analyzing video and detecting cycles..."):
        sampled_b64, frame_ids, frames_cv2 = sample_evenly(temp_video_path, total_samples=TOTAL_SAMPLES)

        descriptions, _ = get_summary_from_frames(sampled_b64, frame_ids)
        combined = "\n".join([d for d in descriptions if "[Error]" not in d])

        summary_prompt = (
            "You are analyzing a short sequence of factory video frames.\n"
            "Here are frame-level summaries:\n\n" + combined +
            "\n\nBased on this, give a one-sentence summary of what kind of manufacturing process is shown."
        )
        one_line_summary = ask_ollama(summary_prompt, image=sampled_b64[len(sampled_b64) // 2], model=selected_model)

        typical_steps_prompt = (
            f"The following manufacturing process was observed in a short video clip: \"{one_line_summary}\"\n\n"
            f"What are the typical steps involved in this kind of manufacturing process?\n"
            f"Provide a high-level step-by-step breakdown (not just what is visible in the video)."
        )
        typical_steps = ask_ollama(typical_steps_prompt, model=selected_model)

        chunk_summaries, step_frame_map = analyze_chunks(sampled_b64, frame_ids, frames_cv2, chunk_size=CHUNK_SIZE)
        observed_steps = consolidate_chunk_descriptions(step_frame_map)

        cycle_steps_summary = generate_cycle_steps_summary(
            sampled_b64, frame_ids, frames_cv2, step_spacing=5, model=selected_model
        )

        final_output = {
            "one_line_summary": one_line_summary,
            "typical_steps": typical_steps,
            "chunk_summaries": chunk_summaries,
            "step_frame_map": step_frame_map,
            "observed_steps": observed_steps,
            "cycles_summary": cycle_steps_summary
        }
        save_output(final_output, OUTPUT_FILE)
        st.session_state["analysis_done"] = True
        st.success("Analysis completed successfully!")

# Display output after analysis
if st.session_state.get("analysis_done") and os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    st.subheader("One-Line Summary")
    st.markdown(f"> {data.get('one_line_summary', '')}")

    st.subheader("Cycle Steps")
    cycles = data.get("cycles_summary", {})
    for cycle_name, steps_text in cycles.items():
        st.markdown(f"### {cycle_name}")
        for line in steps_text.strip().split("\n"):
            if "(Frame" in line:
                try:
                    step_label = line.split(":")[0].strip()
                    frame_num = step_label.split("Frame ")[-1].replace(")", "")
                    step_num = step_label.split()[1]
                    description = line.split(":", 1)[1].strip()

                    filename = f"{cycle_name} - Step {step_num} (Frame {frame_num}).jpg"
                    image_path = os.path.join(CYCLE_IMAGE_DIR, filename)

                    col1, col2 = st.columns([1, 3])
                    with col1:
                        if os.path.exists(image_path):
                            st.image(Image.open(image_path), caption=step_label, width=200)
                        else:
                            st.warning(f"Missing image: {filename}")
                    with col2:
                        st.markdown(f"**{step_label}**")
                        st.write(description)
                except Exception as e:
                    st.error(f"Failed to parse step: {line}\n{e}")
