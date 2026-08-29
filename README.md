# Visual Token Compression for Efficient Vision-Language Models

Training-free visual token selection for LLaVA-Phi-3-Mini VQA, under a fixed budget of 64 tokens.

## Method

**Clustering-First Hybrid**: K-means clustering (250 clusters, cosine similarity) is applied first to ensure diverse coverage of visual tokens, then the top-64 cluster representatives are re-ranked and selected by CLS attention score from the vision encoder's last layer.

Implemented in `main.ipynb` via `select_image_tokens` and a monkey-patched `get_compressed_image_features`.

## Results

| Method | Accuracy (%) |
|---|---|
| Random Sampling (baseline) | 70.00 |
| CLS Attention (last layer) | 80.00 |
| LLM Attention (layer avg) | 80.00 |
| Greedy Diversity | 76.67 |
| K-means Clustering | 76.67 |
| Attention-First Hybrid | 80.00 |
| **Clustering-First Hybrid (ours)** | **83.33** |

Full ablations and hyperparameter search in `Report.pdf`.

## Structure

```
.
├── main.ipynb              # model loading, token selection, inference, evaluation
├── utils/
│   ├── modeling_llava.py
│   ├── utils.py
│   └── eval_utils.py
├── data/annotations.json   # VQA dataset
├── outputs/                # generation_log.json, submission.csv
└── Report.pdf
```

## Usage

```bash
pip install -q transformers==4.57.2
```

Run `main.ipynb` end to end — it loads the model, selects 64 tokens, runs inference, and saves results to `outputs/`.

## References

- Zhang et al. "[CLS] Attention is All You Need for Training-Free Visual Token Pruning." arXiv, 2024.
- Chen et al. "An Image is Worth 1/2 Tokens After Layer 2." ECCV, 2024.
- Yang et al. "VisionZip: Longer is Better but Not Necessary in Vision Language Models." CVPR, 2025.
- Alvar et al. "DivPrune: Diversity-based Visual Token Pruning." CVPR, 2025.
- Zhang et al. "Beyond Attention or Similarity." NeurIPS, 2026.
- Bolya et al. "Token Merging: Your ViT but Faster." arXiv, 2022.
- Shang et al. "LLaVA-PruMerge: Adaptive Token Reduction for Efficient Large Multimodal Models." ICCV, 2025.
