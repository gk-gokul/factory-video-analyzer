import os
import cv2
import base64
import json
import requests
import numpy as np
from skimage.metrics import structural_similarity as ssim

VIDEO_PATH = r"D:\SELF\factory\PURE AI\glass.mp4"
OLLAMA_API = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5vl:latest"
TOTAL_SAMPLES = 35
CHUNK_SIZE = 7
OUTPUT_FILE = "process_summary.json"
FRAME_OUTPUT_DIR = "Frames Outputted"

def encode_image_to_base64(img):
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf).decode("utf-8")

def sample_evenly(video_path, total_samples=TOTAL_SAMPLES):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0 or total_samples == 0:
        cap.release()
        return [], [], []
    interval = max(1, total_frames // total_samples)
    sampled_b64, frame_ids, frames_cv2 = [], [], []

    for i in range(0, total_frames, interval):
        if len(sampled_b64) >= total_samples:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        resized = cv2.resize(frame, (512, 512))
        b64 = encode_image_to_base64(resized)
        sampled_b64.append(b64)
        frame_ids.append(i)
        frames_cv2.append(resized)
    cap.release()
    return sampled_b64, frame_ids, frames_cv2

def ask_ollama(prompt, image=None, model=None):
    payload = {
        "model": model or OLLAMA_MODEL,
        "prompt": prompt,
        "images": [image] if image else [],
        "stream": False
    }
    r = requests.post(OLLAMA_API, json=payload)
    r.raise_for_status()
    return r.json().get("response", "").strip()

def get_summary_from_frames(frames_b64, frame_ids):
    descriptions = []
    for idx, (b64_img, frame_num) in enumerate(zip(frames_b64, frame_ids), 1):
        prompt = (
            f"You are observing a single frame (frame number {frame_num}) from a factory video.\n"
            "Briefly describe what is happening in the frame. Focus only on visible action or machinery tasks, not the background."
        )
        try:
            desc = ask_ollama(prompt, b64_img)
            description_line = f"Frame {frame_num}: {desc}"
            descriptions.append(description_line)
            print(f"[{idx}/{len(frames_b64)}] {description_line}")
        except Exception as e:
            error_msg = f"Frame {frame_num}: [Error] - {e}"
            descriptions.append(f"Frame {frame_num}: [Error]")
            print(f"[{idx}/{len(frames_b64)}] {error_msg}")
    return descriptions, frame_num

def analyze_chunks(frames_b64, frame_ids, frames_cv2, chunk_size=CHUNK_SIZE):
    chunks = []
    step_frame_map = []
    os.makedirs(FRAME_OUTPUT_DIR, exist_ok=True)
    for i in range(0, len(frames_b64), chunk_size):
        chunk_frames = frames_b64[i:i + chunk_size]
        chunk_ids = frame_ids[i:i + chunk_size]
        chunk_cv2 = frames_cv2[i:i + chunk_size]
        if not chunk_frames:
            continue
        mid_index = len(chunk_frames) // 2
        mid_frame_id = chunk_ids[mid_index]
        mid_frame_img = chunk_cv2[mid_index]
        prompt = (
            "You are reviewing a sequence of consecutive frames from a factory video.\n"
            "Summarize what unique manufacturing actions or transitions occur across these frames.\n"
            "Focus on machinery or worker activity only. Do not repeat generic statements.\n"
            "Frame range: " + ", ".join([str(fid) for fid in chunk_ids])
        )
        try:
            summary = ask_ollama(prompt, chunk_frames[mid_index])
            chunks.append(summary)
            step_frame_map.append({"step": summary, "frame": mid_frame_id})
            filename = f"Step - {len(step_frame_map)} (Frame {mid_frame_id}).jpg"
            cv2.imwrite(os.path.join(FRAME_OUTPUT_DIR, filename), mid_frame_img)
            print(f"[{i//chunk_size + 1}] Chunk {chunk_ids[0]}–{chunk_ids[-1]} summary: {summary}")
        except Exception as e:
            print(f"Error processing chunk {i//chunk_size + 1}: {e}")
            chunks.append("[Error]")
            step_frame_map.append({"step": "[Error]", "frame": mid_frame_id})
    return chunks, step_frame_map

def consolidate_chunk_descriptions(step_frame_map):
    chunk_descriptions = "\n".join(
        f"Chunk {i+1} (Frame {entry['frame']}): {entry['step']}"
        for i, entry in enumerate(step_frame_map)
        if entry["step"].strip() and entry["step"] != "[Error]"
    )
    prompt = (
        "The following are chunk-level descriptions of a factory welding and assembly process.\n"
        "Each chunk corresponds to a specific frame.\n\n"
        f"{chunk_descriptions}\n\n"
        "Now, rewrite this as a concise list of step-by-step actions actually happening in the video.\n"
        "Only include what is visually observable in each chunk. Each step should:\n"
        "- Be a short single sentence\n"
        "- Refer to the main action \n"
        "- Include the corresponding frame number in parentheses\n"
        "- Avoid redundancy or over-explaining\n\n"
        "Format:\n"
        "Step 1 (Frame XXX): [action]\nStep 2 (Frame YYY): [next action]\n..."
    )
    return ask_ollama(prompt)

def compute_frame_similarity(frameA, frameB):
    grayA = cv2.cvtColor(frameA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(frameB, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(grayA, grayB, full=True)
    return score

def detect_cycles(frames_cv2, threshold=0.92):
    cycles = []
    current_cycle = [0]
    for i in range(1, len(frames_cv2)):
        similarity = compute_frame_similarity(frames_cv2[i], frames_cv2[0])
        if similarity > threshold and len(current_cycle) > 1:
            cycles.append(current_cycle)
            current_cycle = [i]
        else:
            current_cycle.append(i)
    if current_cycle:
        cycles.append(current_cycle)
    return cycles

def generate_cycle_steps_summary(frames_b64, frame_ids, frames_cv2, step_spacing=3, model=None):
    from collections import OrderedDict

    CYCLE_FRAME_DIR = "Cycle Frames"
    os.makedirs(CYCLE_FRAME_DIR, exist_ok=True)

    def format_step_list(descriptions):
        return "\n".join(
            f"Step {i+1} (Frame {fid}): {text}"
            for i, (fid, text) in enumerate(descriptions)
            if text != "[Error]"
        )

    cycles = detect_cycles(frames_cv2)
    print(f"Detected {len(cycles)} cycle(s)")
    cycle_steps_summary = OrderedDict()

    for cycle_idx, cycle_indices in enumerate(cycles, 1):
        step_descriptions = []
        selected_indices = cycle_indices[::step_spacing] or [cycle_indices[0]]

        for step_idx, frame_idx in enumerate(selected_indices, 1):
            frame_b64 = frames_b64[frame_idx]
            frame_id = frame_ids[frame_idx]
            frame_img = frames_cv2[frame_idx]

            prev_desc = step_descriptions[-1][1] if step_descriptions else None

            if prev_desc:
                prompt = (
                    f"This is step {step_idx} of a repetitive factory manufacturing cycle (Frame {frame_id}).\n"
                    f"Previous step description: \"{prev_desc}\"\n"
                    "Now describe the next visible action performed by either a human or a machine.\n"
                    "In one short sentence (max 12 words), be visually accurate and specific."
                )
            else:
                prompt = (
                    f"This is step {step_idx} of a repetitive factory manufacturing cycle (Frame {frame_id}).\n"
                    "In one short sentence (max 12 words), describe the main visible action performed."
                )

            try:
                desc = ask_ollama(prompt, frame_b64)
                step_descriptions.append((frame_id, desc.strip()))
                print(f"[Cycle {cycle_idx}] Step {step_idx} (Frame {frame_id}): {desc}")

                filename = f"Cycle {cycle_idx} - Step {step_idx} (Frame {frame_id}).jpg"
                filepath = os.path.join(CYCLE_FRAME_DIR, filename)
                cv2.imwrite(filepath, frame_img)

            except Exception as e:
                print(f"Error processing frame {frame_id}: {e}")
                step_descriptions.append((frame_id, "[Error]"))

        cycle_steps_summary[f"Cycle {cycle_idx}"] = format_step_list(step_descriptions)

    return cycle_steps_summary

def save_output(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"\n✅ Output saved to: {path}")

# --------------------------
# MAIN SCRIPT
# --------------------------
if __name__ == "__main__":
    print(f"📽️ Sampling {TOTAL_SAMPLES} evenly spaced frames from video...")
    sampled_b64, frame_ids, frames_cv2 = sample_evenly(VIDEO_PATH, total_samples=TOTAL_SAMPLES)

    print("\n📝 Getting frame-level descriptions...")
    descriptions, _ = get_summary_from_frames(sampled_b64, frame_ids)

    with open("frame_descriptions.txt", "w", encoding="utf-8") as f:
        for line in descriptions:
            f.write(line + "\n")
    print("🗂️ Frame descriptions saved to: frame_descriptions.txt")

    print("\n🧠 Generating one-line process summary...")
    combined = "\n".join([d for d in descriptions if "[Error]" not in d])
    summary_prompt = (
        "You are analyzing a short sequence of factory video frames.\n"
        "Here are frame-level summaries:\n\n" + combined +
        "\n\nBased on this, give a one-sentence summary of what kind of manufacturing process is shown."
    )
    one_line_summary = ask_ollama(summary_prompt, sampled_b64[len(sampled_b64)//2])

    typical_steps_prompt = (
        f"The following manufacturing process was observed in a short video clip: \"{one_line_summary}\"\n\n"
        f"What are the typical steps involved in this kind of manufacturing process?\n"
        f"Provide a high-level step-by-step breakdown (not just what is visible in the video)."
    )
    typical_steps = ask_ollama(typical_steps_prompt)

    print("\n🔍 Analyzing frame chunks for visual steps...")
    chunk_summaries, step_frame_map = analyze_chunks(sampled_b64, frame_ids, frames_cv2)

    print("\n🔁 Consolidating observed chunk-level steps...")
    observed_steps = consolidate_chunk_descriptions(step_frame_map)

    print("\n♻️ Detecting and summarizing cycle steps...")
    cycle_steps_summary = generate_cycle_steps_summary(sampled_b64, frame_ids, frames_cv2)

    final_output = {
        "one_line_summary": one_line_summary,
        "typical_steps": typical_steps,
        "chunk_summaries": chunk_summaries,
        "step_frame_map": step_frame_map,
        "observed_steps": observed_steps,
        "cycles_summary": cycle_steps_summary
    }

    save_output(final_output, OUTPUT_FILE)
