# M-113 scaffold TODO

Scaffolded from template: `M-038_isic2018_skin_lesion_seg_data-pipeline` (2026-04-20)

## Status
- [x] config.py updated (domain=bus_uclm_breast_lesion_seg, s3_prefix=M-113_BUS-UCLM/raw/, fps=3)
- [ ] core/download.py: update URL / Kaggle slug / HF repo_id
- [ ] src/download/downloader.py: adapt to dataset file layout
- [ ] src/pipeline/_phase2/*.py: adapt raw → frames logic (inherited from M-038_isic2018_skin_lesion_seg_data-pipeline, likely needs rework)
- [ ] examples/generate.py: verify end-to-end on 3 samples

## Task prompt
This breast ultrasound image (BUS-UCLM). Segment breast lesions (benign and malignant) with a red mask.

Fleet runs likely FAIL on first attempt for dataset parsing; iterate based on fleet logs at s3://vbvr-final-data/_logs/.
