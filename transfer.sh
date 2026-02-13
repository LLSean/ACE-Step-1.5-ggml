#/bin/bash
# 1) 把 text-encoder 导出为 GGUF（可选量化：F16/Q8/Q6/Q4）
for q in Q8 Q4; do
  /opt/miniconda3/envs/ace-ggml-py311/bin/python /Users/fmh/project/ACE-Step-1.5/acestep_ggml/tools/export_safetensors_to_gguf.py \
    --input /Users/fmh/project/ACE-Step-1.5/checkpoints/Qwen3-Embedding-0.6B/model.safetensors \
    --output /Users/fmh/project/ACE-Step-1.5/checkpoints/Qwen3-Embedding-0.6B/model.${q,,}.gguf \
    --arch qwen3 \
    --quant "$q" \
    --ggml-lib /Users/fmh/project/ACE-Step-1.5/acestep_ggml/build/third_party/ggml/src/libggml-base.0.9.5.dylib
done

