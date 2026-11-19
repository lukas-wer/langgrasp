# ADR-0001: Project Scope

Status: accepted\
Date: 2026-11-18

## Context

Build, train, optimize and deploy a web-testable system (which will be called LangGrasp) that takes a photo of a scene (several objects) plus a text instruction naming the target object as input and returns where and how a parallel-jaw gripper should grasp the object.\
Motivation: Out of personal interest and as a learning portfolio project under the constraints of no local GPU and a small budget. The learning aspect is also a reason for the stack and the development up to deployment.

## Decision

Scope of the project and to what I commit.

```
 1 DATA            2 BUILD TOOLS         3 TRAINING               4 INFERENCE PIPELINE
 Cornell grasps    WSL, uv, git,         PyTorch, Kaggle,         text → find object (YOLO/SAM)
 (RGB-D + boxes)   ruff, pytest, CI      TensorBoard, Pydantic    → crop → predict grasp
        │                 │                     │                  → memory (DINOv2+Qdrant)
        └─────────────────┴──────────┬──────────┘                  → agent (LangGraph)
                                     ▼
                       5 OPTIMIZATION: ONNX → quantization → pruning → TensorRT → C++/CUDA op
                                     ▼
                       6 DEPLOYMENT: FastAPI → Docker/Compose → Cloud Run → (Kubernetes on kind)
```

Delivery is in layers.\
Afirst presentable version v0.1 (data -> own network -> pipeline -> API -> Docker -> public demo)\
Later phases add:
- v0.2: few-shot memory and an agent
- v0.3: edge-optimization
- v1.0: Kubernetes plus final docs

### 1. Dataset

 **Cornell Grapsing Dataset (Jiang et al., 2011)**
- 885 RGB-D images of about 240 household objects
- about 8,000 grasp rectangles drawn in total, labelled "good" and "bad"
- a rectangle is 5 numbers: centre x, centre y, angle, jaw opening (width), jaw size (length). The project represents grasps this way.
- Why Cornell:
    - standard benchmark with public results to compare to
    - It is small: full training runs do not take long on a free Kaggle GPU (maybe 1 h)
    - it has depth images

> Alternatives considered: Jacquard (maybe updating later to this dataset -> much larger), GraspNet-1Billion / Dex-Net (maybe for related work, they are 3D/6-DoF), Collecting my own data, would take too long/ complicated

### 2. Build tools

WSL + Ubuntu 24.04, VSCode, Git + Github, uv, ruff, pytest, mypy, pre-commit, GitHub Actions (CI), Docker, ADRs
> the same Linux as the cloud platforms, and reproducible dependencies

### 3. Model and Training tools

PyTorch, torchvision, NumPy, OpenCV, Shapely, Pydantic + PyYAML, TensorBoard, scikit-learn, Kaggle (Colab as Backup),
> Preferred Tools and Kaggle gives around 30 free GPU hours per week

Experiments planned:
1. Normalization Study (BatchNorm vs GroupNorm, vs LayerNorm)
2. Depth study: does depth help and can estimated depth (Depth Anything V2) replace a depth sensor
3. Backbone / data-efficiency study: my own CNN vs a frozen DINOv2 backbone with my own decoder

### 4. Inference pipeline

Pretrained models, used but not trained:

| Component | Role |
| --- | --- |
|YOLO-World/YOLOE | Finds the object named in the text |
|SAM / SAM 2 / SAM 3 | Cuts the target out of the scene |
|CLIP | Optional re-ranking of detections |
|Depth Anything V2 Small | Depth for plain photos lacking a depth sensor |
|DINOv2 | Frozen backbone for grapsing. Object embeddings for few-shot memory |
|Qdrant + PostgreSQL + SQLAlchemy | vector DB and request logging |
|LangChain + LangGraph | An agent decides which tool to call (locate, grasp, remember) with explicit control flow and a hard stop |

### 5. Optimization

| Technique | Role | Why |
| --- | --- | --- |
| ONNX (Runtime) | Framework-neutral model format and a fast inference engine | Bridge from PyTorch to TensorRT; baseline latency |
| Quantization | Store weights as 8-bit integers instead of 32-bit floats | Smaller and faster. Post-training first, training-aware if accuracy drops |
| Pruning | Remove weights or whole channels | Compare unstructured (little real speedup) with structured (real speedup) |
| TensorRT | NVIDIA's inference optimizer | industry standard; Is tested on a cloud GPU. Jetson is a separate optional extension |
| C++/CUDA custom op | Own implementation of the peak-finding post processing | post-processing can potentially dominate latency |
|Jetson Orin nano + stereo camera | Edge hardware and 3D camera | Optional, only if I decide to buy the hardware |

### 6. Deployment

FastAPI + Uvicorn, Docker + Docker Compose, Google Cloud Run, Kubernetes (kind) + kubectl, Hugging Face Hub
> Free accounts on Hugging Face Spaces can no longer create Docker Spaces. Cloud Run has free tier that scales to zero. Hugging Face PRO is the fallback. The reason for Cloud run plus Docker is to ship the same image everywhere


## Out-of-Scope:

- Real robot execution: no hardware control, no graps-success test with a real arm
- 3D and 6-DoF grsping: related work only
- Training the big pretrained models from scratch (SAM, YOLO, DINOv2, CLIP)
- Optional, not committed: Jacquard Dataset, Jetson, stereo camera
- Production Concerns: user accounts, authentication, scalingbeyond the free tier, monitoring dashboards

## Consequences

Positives:
- Results are comparable to published numbers
- It is cheap
- The delivery is layered, so a presentable version v0.1 exists early

Negatives/risks:
- Small dataset: about 885 images limit the achievable accuracy and generalization to unseen objects (mitigate with augmentation and object-wise split)
- Free tiers (Kaggle quota, Cloud Run free tier) can change. Fallbacks: Colab, HF Pro, rented GPU.
- Licensing: AGPL (YOLO) influences the license of the repo

## Alternatives

- different project
- larger dataset: e.g. Jacquard as the core, but: cost and time are not justified until the pipeline works
- skip deployment
- fine-tune a big pretrained grasp model instead of writing my own network: faster results, but less learning. I do compare against a pretrained backbone (DINOv2)
