#!/usr/bin/env python3
# build_ppe_json.py — Convert YOLO detections to structured JSON for PPE compliance (per person + per equipment)

import os, json, argparse
from pathlib import Path

def xyxy_to_center(xyxy):
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2)/2, (y1 + y2)/2)

def center_in_box(c, box):
    cx, cy = c; x1, y1, x2, y2 = box
    return (x1 <= cx <= x2) and (y1 <= cy <= y2)

def l2(a, b):
    ax, ay = a; bx, by = b
    return ((ax - bx)**2 + (ay - by)**2) ** 0.5

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_model(weights: str):
    from ultralytics import YOLO
    return YOLO(weights, task='detect')

def run_model(model, image_path: str, conf: float, iou: float):
    return model.predict(source=image_path, conf=conf, iou=iou, save=False, verbose=False, stream=False, max_det=300)

def choose_best(dets):
    # dets: list of (xyxy, conf)
    return sorted(dets, key=lambda x: x[1], reverse=True)[0] if dets else None

def process_image(img_path: Path, model, conf: float, iou: float, out_dir: Path, type_keys):
    results = list(run_model(model, str(img_path), conf, iou))
    if not results:
        print(f"[WARN] no result for {img_path}")
        return
    r = results[0]
    imw, imh = r.orig_shape[1], r.orig_shape[0]
    names = r.names
    boxes = r.boxes

    dets = []
    persons = []
    for i in range(len(boxes)):
        xyxy = boxes.xyxy[i].tolist()
        conf_i = float(boxes.conf[i].item())
        cls_idx = int(boxes.cls[i].item())
        cls_name = names.get(cls_idx, str(cls_idx))
        det = {"cls_name": cls_name, "cls_idx": cls_idx, "bbox_xyxy": xyxy, "conf": conf_i}
        dets.append(det)
        if cls_name.lower() == "person":
            persons.append(det)

    out_persons = []

    if not persons:
        # image-level fallback
        per_type = {k: [] for k in type_keys}
        unassigned = []
        for d in dets:
            k = d["cls_name"].lower()
            if k in per_type: per_type[k].append((d["bbox_xyxy"], d["conf"]))
            else: unassigned.append(d)
        equip = {k: {"present": False, "bbox_xyxy": None, "conf": 0.0} for k in type_keys}
        for k in type_keys:
            b = choose_best(per_type[k])
            if b: equip[k] = {"present": True, "bbox_xyxy": b[0], "conf": float(b[1])}
        out_persons.append({
            "person_id": 1, "bbox_xyxy": None, "conf": None,
            "equipments": equip, "notes": ["No person boxes; image-level aggregation."]
        })
        img_obj = {"image": img_path.name, "image_size": [int(imw), int(imh)], "persons": out_persons, "unassigned_detections": unassigned}
        (out_dir / f"{img_path.stem}.json").write_text(json.dumps(img_obj, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] Wrote {out_dir / (img_path.stem + '.json')}")
        return

    # with person boxes
    pcenters = [xyxy_to_center(p["bbox_xyxy"]) for p in persons]
    per_person_equips = [{k: [] for k in type_keys} for _ in persons]
    unassigned = []

    for d in dets:
        cname = d["cls_name"].lower()
        if cname == "person": continue
        c = xyxy_to_center(d["bbox_xyxy"])
        inside = [pid for pid, p in enumerate(persons) if center_in_box(c, p["bbox_xyxy"])]
        if len(inside) == 1:
            owner = inside[0]
        elif len(inside) > 1:
            owner = sorted([(pid, l2(c, pcenters[pid])) for pid in inside], key=lambda x: x[1])[0][0]
        else:
            owner = sorted([(pid, l2(c, pcenters[pid])) for pid in range(len(persons))], key=lambda x: x[1])[0][0]
        if cname in type_keys:
            per_person_equips[owner][cname].append((d["bbox_xyxy"], d["conf"]))
        else:
            unassigned.append(d)

    for pid, p in enumerate(persons):
        equip = {k: {"present": False, "bbox_xyxy": None, "conf": 0.0} for k in type_keys}
        for k in type_keys:
            b = choose_best(per_person_equips[pid][k])
            if b: equip[k] = {"present": True, "bbox_xyxy": b[0], "conf": float(b[1])}
        out_persons.append({
            "person_id": pid + 1,
            "bbox_xyxy": [float(x) for x in p["bbox_xyxy"]],
            "conf": float(p["conf"]),
            "equipments": equip,
            "notes": []
        })

    img_obj = {"image": img_path.name, "image_size": [int(imw), int(imh)], "persons": out_persons, "unassigned_detections": unassigned}
    (out_dir / f"{img_path.stem}.json").write_text(json.dumps(img_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Wrote {out_dir / (img_path.stem + '.json')}")

def main():
    ap = argparse.ArgumentParser(description="Convert YOLO detections to structured JSON (PPE per person).")
    ap.add_argument("--weights", type=str, required=True, help="YOLO weights file (your PPE model).")
    ap.add_argument("--imgdir", type=str, required=True, help="Directory of images to run.")
    ap.add_argument("--out", type=str, required=True, help="Output directory for per-image JSON.")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU threshold for NMS.")
    ap.add_argument("--types", type=str, default="helmet,gloves,vest,boots,goggles", help="Comma-separated equipment class names.")
    args = ap.parse_args()

    out_dir = Path(args.out); ensure_dir(out_dir)
    model = load_model(args.weights)
    type_keys = [s.strip().lower() for s in args.types.split(",") if s.strip()]

    imgdir = Path(args.imgdir)
    imgs = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        imgs.extend(imgdir.glob(ext))
    if not imgs:
        raise SystemExit(f"No images found under: {imgdir}")

    for p in imgs:
        process_image(p, model, args.conf, args.iou, out_dir, type_keys)

if __name__ == "__main__":
    main()
