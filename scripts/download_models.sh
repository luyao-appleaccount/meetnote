#!/usr/bin/env bash
# 下载说话人分离所需模型(约 50 MB)与官方测试音频到 models/
# 默认走 hf-mirror;网络顺畅想直连 HF 可设置: export HF_ENDPOINT=https://huggingface.co
set -euo pipefail

ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
cd "$(dirname "$0")/.."
mkdir -p models

fetch() {
    local url="$1" out="$2"
    echo "downloading $out ..."
    curl -sL --retry 3 -o "$out" "$url"
}

# pyannote 分段模型(sherpa-onnx 导出版)
fetch "$ENDPOINT/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0/resolve/main/model.onnx" \
      models/model.onnx

# 3D-Speaker CAM++ 中文声纹(zh-cn 16k common)
fetch "$ENDPOINT/csukuangfj/speaker-embedding-models/resolve/main/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx" \
      models/3dspeaker_zh.onnx

# sherpa 官方说话人分离测试音频(供 benchmark_diar.py 的 official_* 用例)
fetch "$ENDPOINT/csukuangfj/speaker-embedding-models/resolve/main/0-four-speakers-zh.wav" \
      models/0-four-speakers-zh.wav
fetch "$ENDPOINT/csukuangfj/speaker-embedding-models/resolve/main/1-two-speakers-en.wav" \
      models/1-two-speakers-en.wav
fetch "$ENDPOINT/csukuangfj/speaker-embedding-models/resolve/main/2-two-speakers-en.wav" \
      models/2-two-speakers-en.wav
fetch "$ENDPOINT/csukuangfj/speaker-embedding-models/resolve/main/3-two-speakers-en.wav" \
      models/3-two-speakers-en.wav

echo "done. models/:"
ls -la models/
