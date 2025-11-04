#!/usr/bin/env python3
import os, json, argparse
from pathlib import Path
import requests

def extract_first_json(text: str):
    start = text.find('{')
    if start == -1: return None, text
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == '{': depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                js = text[start:i+1]
                rest = text[i+1:].strip()
                return js, rest
    return None, text

def call_qwen_api(image_path, yolo_json, prompt_path, model_url):
    prompt = Path(prompt_path).read_text(encoding="utf-8")
    yolo_str = json.dumps(json.load(open(yolo_json, "r", encoding="utf-8")), ensure_ascii=False)
    image_data = open(image_path, "rb").read()
    import base64
    image_b64 = base64.b64encode(image_data).decode("utf-8")

    payload = {
        "model": "qwen2-vl-7b",
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                {"type": "text", "text": "YOLO JSON:\n" + yolo_str}
            ]}
        ]
    }

    r = requests.post(model_url, json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    # 嘗試擷取文字
    content = None
    if "choices" in data:
        content = data["choices"][0]["message"]["content"]
    else:
        content = json.dumps(data, ensure_ascii=False)
    return content

def run_one(image_path, json_path, prompt_path, model_url, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    text = call_qwen_api(image_path, json_path, prompt_path, model_url)
    js_str, summary = extract_first_json(text)
    parsed = None
    if js_str:
        try:
            parsed = json.loads(js_str)
        except Exception:
            parsed = None
    raw_path = out_dir / f"{image_path.stem}.raw.txt"
    raw_path.write_text(text, encoding="utf-8")
    if parsed:
        (out_dir / f"{image_path.stem}.report.json").write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / f"{image_path.stem}.summary.txt").write_text(summary if parsed else text, encoding="utf-8")
    print(f"[DONE] {image_path.name}")

def main():
    ap = argparse.ArgumentParser(description="Call Qwen2-VL local API with image+YOLO JSON+Prompt.")
    ap.add_argument("--image", type=str)
    ap.add_argument("--json", type=str)
    ap.add_argument("--prompt-file", type=str, default="ppe_prompt_zh.txt")
    ap.add_argument("--model", type=str, default="http://127.0.0.1:8000/v1/chat/completions")
    ap.add_argument("--out", type=str, default="vlm_outputs")
    ap.add_argument("--batch", action="store_true")
    ap.add_argument("--image-dir", type=str)
    ap.add_argument("--json-dir", type=str)
    args = ap.parse_args()

    out_dir = Path(args.out)
    prompt_path = Path(args.prompt_file)

    if args.batch:
        img_dir = Path(args.image_dir)
        json_dir = Path(args.json_dir)
        for jp in sorted(json_dir.glob("*.json")):
            stem = jp.stem
            ip = next((p for p in img_dir.glob(f"{stem}.*") if p.suffix.lower() in [".jpg", ".png", ".jpeg"]), None)
            if not ip:
                print(f"[WARN] no image for {stem}")
                continue
            run_one(ip, jp, prompt_path, args.model, out_dir)
    else:
        if not args.image or not args.json:
            raise SystemExit("Provide --image and --json or use --batch with --image-dir and --json-dir")
        run_one(Path(args.image), Path(args.json), prompt_path, args.model, out_dir)

if __name__ == "__main__":
    main()
