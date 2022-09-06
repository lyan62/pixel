export RENDERER_PATH="configs/renderers/noto_renderer"

python scripts/data/prerendering/prerender_bookcorpus.py \
  --renderer_name_or_path=${RENDERER_PATH} \
  --chunk_size=100000 \
  --repo_id="lyan62/pixel-exp" \
  --split="train" \
  --auth_token="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"