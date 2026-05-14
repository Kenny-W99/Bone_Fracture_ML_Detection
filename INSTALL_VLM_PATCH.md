# Install VLM Baseline Patch

Put these files into the root of `Bone_Frature` / `Bone_Fracture_ML_Detection`.

## File placement

- `src/vlm_baseline.py` -> `src/vlm_baseline.py`
- `src/evaluate_vlm.py` -> `src/evaluate_vlm.py`
- `src/make_vlm_subset.py` -> `src/make_vlm_subset.py`
- `src/append_vlm_to_summary.py` -> `src/append_vlm_to_summary.py`
- `requirements-vlm.txt` -> project root
- `README_VLM_BASELINE.md` -> project root
- `.env.example` -> project root
- `env.example.txt` -> project root, visible backup of `.env.example` for macOS Finder

## Copy commands

```bash
unzip VLM_Baseline_Code_Patch_v2.zip
cp -r vlm_baseline_patch_v2/src/* src/
cp vlm_baseline_patch_v2/requirements-vlm.txt .
cp vlm_baseline_patch_v2/README_VLM_BASELINE.md .
cp vlm_baseline_patch_v2/.env.example .
cp vlm_baseline_patch_v2/env.example.txt .
```

## API keys

macOS Finder may hide `.env.example` because files starting with `.` are hidden. If you do not see it, use `env.example.txt`.

```bash
cp env.example.txt .env
open .env
```

Never commit your real `.env` file:

```bash
echo ".env" >> .gitignore
```

## Mock test

```bash
python src/vlm_baseline.py \
  --provider mock \
  --test_csv data/FracAtlas/test.csv \
  --image_dir data/FracAtlas/images \
  --prompt_mode simple \
  --limit 10 \
  --output_csv outputs/vlm/mock_debug.csv

python src/evaluate_vlm.py \
  --predictions_csv outputs/vlm/mock_debug.csv \
  --output_dir outputs/vlm/mock_debug_eval \
  --title "Mock VLM Debug"
```
