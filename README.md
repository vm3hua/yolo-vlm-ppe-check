# Construction Site PPE Detection + VLM Integration



## Project Overview
This project focuses on **construction site safety detection**, combining **YOLOv12n** for object detection and **Qwen2-VL-7B-Instruct** for multimodal reasoning and natural-language reporting.  
The system detects whether each worker correctly wears personal protective equipment (PPE) — such as **helmets, vests, gloves, boots, and goggles** — and automatically generates a **JSON report** and a **Chinese text summary** describing compliance results.



## Dataset — Construction-PPE
The dataset used in this project is the **Construction-PPE Dataset**, which contains annotated images of workers wearing or missing safety equipment.  
All annotations are provided in **YOLO format**, suitable for direct training and evaluation.

### Class Distribution
![stat](https://hackmd.io/_uploads/Skln8y91-l.png)


## System Architecture

The overall pipeline consists of five main stages:
![ob_vlm](https://hackmd.io/_uploads/HJEzwkckZg.png)
