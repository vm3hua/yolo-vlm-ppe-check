# Construction Site PPE Detection + VLM Integration


---

## Project Overview
This project focuses on **construction site safety detection**, combining **YOLOv12n** for object detection and **Qwen2-VL-7B-Instruct** for multimodal reasoning and natural-language reporting.  
The system detects whether each worker correctly wears personal protective equipment (PPE) — such as **helmets, vests, gloves, boots, and goggles** — and automatically generates a **JSON report** and a **Chinese text summary** describing compliance results.

---

## Dataset — Construction-PPE
The dataset used in this project is the **Construction-PPE Dataset**, which contains annotated images of workers wearing or missing safety equipment.  
All annotations are provided in **YOLO format**, suitable for direct training and evaluation.

### Class Distribution

![Construction-PPE Dataset Distribution](images/stats.png)

| Class | Count |
|:------|------:|
| person | 1750 |
| helmet | 1461 |
| gloves | 1632 |
| boots | 1613 |
| goggles | 526 |
| vest | 800 |
| no_helmet | 2265 |
| no_gloves | 485 |
| no_boots | 411 |
| no_vest | 556 |
| no_goggles | 115 |

---

## ⚙️ System Architecture

The overall pipeline consists of five main stages:

