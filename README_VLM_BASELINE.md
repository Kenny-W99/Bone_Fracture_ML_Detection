# VLM / Multimodal LLM Baseline Add-on

This patch adds a zero-shot Vision-Language Model (VLM) baseline to the existing FracAtlas CNN fracture detection project.

The point is **not** to use a VLM as a clinical diagnostic system. The point is to benchmark general-purpose multimodal models against your trained CNN transfer-learning models on the same held-out FracAtlas test split.

## Files

Place these files in the repository root:

```text
src/vlm_baseline.py
src/evaluate_vlm.py
src/make_vlm_subset.py
src/append_vlm_to_summary.py
requirements-vlm.txt
.env.example
```

## Install dependencies

```bash
pip install -r requirements-vlm.txt
```

## API keys

Set only the key for the provider you use:

```bash
export OPENAI_API_KEY="..."
export GOOGLE_API_KEY="..."
export ANTHROPIC_API_KEY="..."
```

## First: run without API cost using mock mode

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

Mock mode is only for testing the pipeline. Do **not** report mock results.

## Recommended low-cost debug subset

```bash
python src/make_vlm_subset.py \
  --input_csv data/FracAtlas/test.csv \
  --output_csv data/FracAtlas/test_vlm_debug_40.csv \
  --per_class 20
```

Then run a real provider on only 40 images first.

## OpenAI example

```bash
python src/vlm_baseline.py \
  --provider openai \
  --model gpt-4o-mini \
  --test_csv data/FracAtlas/test_vlm_debug_40.csv \
  --image_dir data/FracAtlas/images \
  --prompt_mode simple \
  --openai_detail low \
  --output_csv outputs/vlm/openai_simple_debug40.csv \
  --resume

python src/evaluate_vlm.py \
  --predictions_csv outputs/vlm/openai_simple_debug40.csv \
  --output_dir outputs/vlm/openai_simple_debug40_eval \
  --title "OpenAI VLM Simple Prompt"
```

## Gemini example

```bash
python src/vlm_baseline.py \
  --provider gemini \
  --model gemini-2.5-flash \
  --test_csv data/FracAtlas/test_vlm_debug_40.csv \
  --image_dir data/FracAtlas/images \
  --prompt_mode simple \
  --output_csv outputs/vlm/gemini_simple_debug40.csv \
  --resume

python src/evaluate_vlm.py \
  --predictions_csv outputs/vlm/gemini_simple_debug40.csv \
  --output_dir outputs/vlm/gemini_simple_debug40_eval \
  --title "Gemini VLM Simple Prompt"
```

## Anthropic / Claude example

```bash
python src/vlm_baseline.py \
  --provider anthropic \
  --model claude-sonnet-4-20250514 \
  --test_csv data/FracAtlas/test_vlm_debug_40.csv \
  --image_dir data/FracAtlas/images \
  --prompt_mode simple \
  --output_csv outputs/vlm/claude_simple_debug40.csv \
  --resume

python src/evaluate_vlm.py \
  --predictions_csv outputs/vlm/claude_simple_debug40.csv \
  --output_dir outputs/vlm/claude_simple_debug40_eval \
  --title "Claude VLM Simple Prompt"
```

## Full 613-image test set

After the 40-image debug run works, use the full test set by switching back to:

```bash
--test_csv data/FracAtlas/test.csv
```

I recommend testing three prompt modes:

```bash
for PROMPT in simple conservative sensitive; do
  python src/vlm_baseline.py \
    --provider openai \
    --model gpt-4o-mini \
    --test_csv data/FracAtlas/test.csv \
    --image_dir data/FracAtlas/images \
    --prompt_mode $PROMPT \
    --openai_detail low \
    --output_csv outputs/vlm/openai_${PROMPT}_test.csv \
    --resume

  python src/evaluate_vlm.py \
    --predictions_csv outputs/vlm/openai_${PROMPT}_test.csv \
    --output_dir outputs/vlm/openai_${PROMPT}_eval \
    --title "OpenAI VLM ${PROMPT} Prompt"
done
```

## Add VLM row to CNN comparison table

```bash
python src/append_vlm_to_summary.py \
  --cnn_summary outputs/results_summary.csv \
  --vlm_metrics outputs/vlm/openai_simple_eval/test_metrics.csv \
  --output_csv outputs/results_summary_with_vlm.csv \
  --model_label "GPT-4o mini VLM (zero-shot)"
```

## Paper language

Use phrasing like:

> To compare task-specific CNN transfer learning with general-purpose vision-language models, I evaluated a zero-shot VLM baseline on the held-out FracAtlas test set. The VLM was not fine-tuned on FracAtlas. Each X-ray was provided to the model with a standardized research-only prompt asking for binary fracture classification and structured JSON output. This baseline is included only for experimental comparison and is not intended for clinical diagnosis.

## Note about `.env.example` on macOS

macOS Finder hides files whose names begin with a dot. This patch includes both `.env.example` and `env.example.txt`. They contain the same template information. If you cannot see `.env.example`, use `env.example.txt` and copy it to `.env`.
