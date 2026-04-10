# UniPath

## A Unified Multimodal Generative Foundation Model for Precision Oncology
<img src="assets/logo.png" width="210px" align="right" />
<!-- [Journal Link]() | [Download Model]() | [Cite](#reference) -->

 [Download Model](https://huggingface.co/UniPath278/UniPath)

<!-- *Nature*  -->


**UniPath** is a unified multimodal framework that integrates hierarchical whole-slide image (WSI) encoding with multimodal instruction tuning, supporting both discriminative tasks (diagnosis, biomarker prediction, treatment response, survival prognosis) and generative tasks (pathology report generation, visual question answering). It is trained on 94,715 WSI–report pairs and 720,595 VQA pairs spanning 13 anatomical sites and 32 cancer types, with external validation on 33,000+ WSIs from 51 hospitals.
## Installation
 
```bash
git clone https://github.com/UniPath278/UniPath
cd UniPath
 
conda create -n unipath python=3.10 -y
conda activate unipath
pip install --upgrade pip
pip install -e .
```

 
## Getting Started
 
### 1. Download Model Weights
 
Request access and download from [Hugging Face](https://huggingface.co/UniPath278/UniPath).
 
```python
from huggingface_hub import login
from transformers import AutoModel
 
login()  # https://huggingface.co/settings/tokens
model = AutoModel.from_pretrained('UniPath278/UniPath', trust_remote_code=True)
```
**Base LLM**: UniPath uses [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B) as the language decoder for generative tasks. Download and place it under the project root:
 
```bash
# Download Qwen2.5-7B
huggingface-cli download Qwen/Qwen2.5-7B --local-dir ./qwen7B
```
Then, you can manually edit xtuner/configs/unipath/stage_4.py and set: 
```bash
llm_name_or_path = "./qwen7B"
```

### 2. Feature Extraction
 
UniPath uses [TRIDENT](https://github.com/mahmoodlab/TRIDENT) for WSI preprocessing and feature extraction. Depending on the task, you need either patch-level or slide-level features:
 
#### Patch-level Features (for Training & Generative Inference)
 
Patch-level features are used for **all training stages** and **generative inference** (report generation & VQA). This uses the standard CONCH v1.5 extraction in TRIDENT — **no code modification is needed**:
 
```bash
python run_batch_of_slides.py --task all --wsi_dir ./wsis --job_dir ./trident_processed --patch_encoder conch_v15 --patch_size 512 --mag 20
```
 
#### Slide-level Features (for Discriminative Inference)
 
Slide-level embeddings are needed for **discriminative tasks** (classification, biomarker prediction, survival prognosis, etc.). Since UniPath's slide encoder shares the same architecture as TITAN, you can directly replace the TITAN model weights with UniPath's and use TITAN's extraction pipeline. See [`./trident_modifications/README.md`](./trident_modifications/README.md) for detailed instructions.
 
After modification, run:
 
```bash
python run_batch_of_slides.py --task all --wsi_dir ./wsis --job_dir ./trident_processed --slide_encoder titan --patch_size 512 --mag 20
```
 

## Inference
 
We provide **pre-extracted demo features** so you can run inference directly without the extraction pipeline. The demo features can be downloaded from [Hugging Face](https://huggingface.co/UniPath278/UniPath).



### Generative Tasks (Report Generation & VQA)
 
Generative tasks take **patch-level features** (CONCH v1.5) directly as input — no slide-level aggregation is needed.
 
```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> \
PYTHONPATH=. \
python ./xtuner/tools/test.py \
  ./xtuner/configs/unipath/stage_4.py \
  --checkpoint <PATH_TO_CHECKPOINT> \
  --vision_weight_path <PATH_TO_TITAN_WEIGHT> \
  --llm_path <PATH_TO_QWEN> \
  --feature_dir <PATH_TO_FEATURE_DIR> \
  --test_slide_csv <PATH_TO_TEST_CSV> \
  --test_output_csv <PATH_TO_OUTPUT_CSV> \
  --local_rank 0
```
 
### Discriminative Tasks (Slide-level Embedding)
 
For prediction tasks (classification, biomarker prediction, survival prognosis, etc.), UniPath first aggregates patch features into a **slide-level embedding** via its hierarchical visual encoder.
 
```bash
python inference_slide_embedding.py \
    --feat_dir ./demo_features \
    --model_path ./checkpoints/unipath \
    --output_dir ./results/slide_embeddings
```
We provide pre-extracted features for demo evaluation.

Download the EBRAINS features from [Hugging Face](https://huggingface.co/UniPath278/UniPath_Model/tree/main/feature) and place them under `./evaluation/`.
Then run the evaluation script directly:
 
```bash
python evaluation/ebrains12_slide_logistic_regression.py
```

## Training
 
UniPath adopts a three-stage training strategy:
 
1. **Stage 1 — Cross-modal alignment**: Aligns visual and linguistic representations using WSI–report pairs.
2. **Stage 2 — End-to-end instruction tuning**: Joint optimization on WSI–report pairs and VQA data.
3. **Stage 3 — Knowledge-augmented refinement**: Incorporates structured pathology knowledge for clinical consistency.
 
```bash
# Stage 1: Cross-modal alignment
python train.py --stage 1 --config configs/stage1_alignment.yaml
 
# Stage 2: Instruction tuning
python train.py --stage 2 --config configs/stage2_instruction.yaml
 
# Stage 3: Knowledge-augmented refinement
python train.py --stage 3 --config configs/stage3_refinement.yaml
```
 
Adjust training commands to match actual scripts
 

 
<!-- Adjust script names and arguments to match actual codebase -->

<!-- Adjust method names to match actual API -->

<!-- ## Demo Notebooks

| Notebook | Description |
|:---------|:------------|
| `notebooks/embedding_demo.ipynb` | Slide embedding extraction |
| `notebooks/report_generation_demo.ipynb` | Pathology report generation |
| `notebooks/vqa_demo.ipynb` | Visual question answering |
| `notebooks/zeroshot_demo.ipynb` | Zero-shot classification | -->

## License

ⓒ [Lab Name]. Released under [CC-BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/deed.en) for non-commercial academic research only.

<!-- ## Reference

```
@article{unipath2025,
  title={A Unified Multimodal Generative Foundation Model for Precision Oncology},
  author={},
  journal={},
  year={2025}
}
``` -->
