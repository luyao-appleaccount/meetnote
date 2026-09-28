# 模型目录(二进制不入 git)

运行 `bash scripts/download_models.sh` 下载(默认走 hf-mirror / ModelScope):

| 文件 | 来源 | 角色 |
|---|---|---|
| `model.onnx` | csukuangfj/sherpa-onnx-pyannote-segmentation-3-0 | pyannote 分段模型(sherpa 导出版) |
| `3dspeaker_zh.onnx` | csukuangfj/speaker-embedding-models (3dspeaker_speech_campplus_sv_zh-cn_16k-common) | 3D-Speaker CAM++ 中文声纹 |
| `0-four-speakers-zh.wav` 等 | 同上仓库 | sherpa 官方分离测试音频(benchmark 用例) |
| `paraformer-onnx/` | ModelScope: iic/speech_paraformer-large_..._vocab8404-onnx | FunASR 中文转写(int8 量化,~238MB) |
| `punc-onnx/` | ModelScope: iic/punc_ct-transformer_zh-cn-common-vocab272727-onnx | FunASR 中文标点(int8 量化,~283MB) |

whisper 模型不在本目录:首次启动读本地 HF 缓存,没有则自动走 hf-mirror 下载。

可用 `MEETNOTE_MODELS_DIR=/your/path` 重定向模型目录。
