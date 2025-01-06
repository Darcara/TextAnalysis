# TextAnalysis
Sentence splitting, named entity recognition, translation and more

---

## Sentence splitting with SaT / WtP

[Segment Any Text](https://arxiv.org/abs/2406.16678) (June 2024) is the successor to [Where's the Point](https://aclanthology.org/2023.acl-long.398/) (July 2023). The code from both papers is available on [GitHub](https://github.com/segment-any-text/wtpsplit).  
SaT supports 85 languages. The detailed list is available in [their GitHub readme](https://github.com/segment-any-text/wtpsplit?tab=readme-ov-file#supported-languages).   
Models for SaT come in 3 flavors:

* Base models with 1, 3, 6, 9 or 12 layers available on [HuggingFace](https://huggingface.co/collections/segment-any-text/sat-base-models-66718d0a24321d017692b698)  
  More layers means higher accuracy, but longer inference time
* [Low-Rank Adaptation](https://arxiv.org/abs/2106.09685) (LoRA) modules are available for 3 and 12 layer base models in their respective repositories  
  The LoRA modules enable the base models to be adapted to specific domains and styles
* Supervised Mixture (sm) models with 1, 3, 6, 9, 12 layers available on [HuggingFace](https://huggingface.co/collections/segment-any-text/sat-supervised-mixture-sm-models-66718d8c562ee91c16d78f2f)  
  SM models have been trained with a "supervised mixture" of diverse styles and corruptions.
  They score higher both on english and multilingual text.

This project supports the *-sm model family in [onnx](https://onnx.ai/) format.

### Configuration

The SaT-Models benefit greatly from the GPU.
For running on GPU setting the `SessionConfiguration.Batching` to `batch=4` is best.
For running on CPU  setting the `SessionConfiguration.Batching` to `batch=1` with `IterOperationThreads=1` and `IntraOperationThreads=2` will . Higher values for `IntraOperationThreads` will slightly decrease computing time, but use a lot more processing power. It is preferable to sentencize multiple text in parallel.  

A consuming project must nce a proper ONNX-runtime. For Windows deployments Microsoft.ML.OnnxRuntime.DirectML [![Nuget (with prereleases)](https://img.shields.io/nuget/vpre/Microsoft.ML.OnnxRuntime.DirectML)](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.DirectML/) with Microsoft.AI.DirectML [![Nuget (with prereleases)](https://img.shields.io/nuget/vpre/Microsoft.AI.DirectML)](https://www.nuget.org/packages/Microsoft.AI.DirectML/) will yield the best performance.
Setting the `RuntimeIdentifier` in the project csproj to `win-x64` is required.

### Evaluation

The corpora scores are from the [original SaT Github](https://github.com/segment-any-text/wtpsplit?tab=readme-ov-file#supported-languages)  
This benchmark used the novel ["The Adventures of Tom Sawyer, Complete by Mark Twain" from Project Gutenberg](https://www.gutenberg.org/ebooks/74)  
The -model columns give the speed of only the model runtime, whereas -complete includes all pre and post data preparations, including the word tokenization.

| Model                                                            | English Score | Multilingual Score | CPU-model | CPU-complete | GPU-model | GPU-complete |
|:-----------------------------------------------------------------|--------------:|-------------------:|-----------|--------------|-----------|:-------------|
| [sat-1l](https://huggingface.co/segment-any-text/sat-1l)         |          88.5 |               84.3 |           |              |           |              |
| [sat-1l-sm](https://huggingface.co/segment-any-text/sat-1l-sm)   |          88.2 |               87.9 |           |              |           |              |
| [sat-3l](https://huggingface.co/segment-any-text/sat-3l)         |          93.7 |               89.2 |           |              |           |              |
| [sat-3l-sm](https://huggingface.co/segment-any-text/sat-3l-sm)   |          96.5 |               93.5 |           |              |           |              |
| [sat-6l](https://huggingface.co/segment-any-text/sat-6l)         |          94.1 |               89.7 |           |              |           |              |
| [sat-6l-sm](https://huggingface.co/segment-any-text/sat-6l-sm)   |          96.9 |               95.1 |           |              |           |              |
| [sat-9l](https://huggingface.co/segment-any-text/sat-9l)         |          94.3 |               90.3 |           |              |           |              |
| [sat-12l](https://huggingface.co/segment-any-text/sat-12l)       |          94.0 |               90.4 |           |              |           |              |
| [sat-12l-sm](https://huggingface.co/segment-any-text/sat-12l-sm) |          97.4 |               96.0 |           |              |           |              |

### Implementation notes

Word tokenization is done by [sentencepiece](https://github.com/google/sentencepiece) using the [xlm-roberta-base](https://huggingface.co/FacebookAI/xlm-roberta-base/tree/main) ([Alt1](https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-base-sentencepiece.bpe.model), [Alt2](https://github.com/microsoft/BlingFire/raw/refs/heads/master/ldbsrc/xlm_roberta_base/spiece.model)) model.
It is used in C# with the help of the [SentencePieceTokenizer](https://github.com/Darcara/SentencePieceTokenizer) library.

See:

* https://www.kaggle.com/code/samuellongenbach/xlm-roberta-tokenizers-issue/notebook
* https://github.com/google/sentencepiece/issues/1042#issuecomment-2295028056
* Seems to have no resolution, other than re-writing the model, since I don't know how to "modify the indexing scheme to start from 1" ?


## Compiling onnx runtime on Windows

### Prerequisites

Reference https://onnxruntime.ai/docs/build/inferencing.html

* Python 3.12
* CMake
* Visual Studio 2022 (with MSVC v143 C++ x64/x86 BuildTools(v14.41-17.11))
* Make sure the build folder is empty or missing before starting

```bash
git clone https://github.com/microsoft/onnxruntime
-- or --
git fetch
git checkout v1.20.1

onnxruntime> PATH=%PATH%;C:\Program Files\Python312
onnxruntime> build.bat --cmake_path "C:\Program Files\CMake\bin\cmake.exe" --ctest_path "C:\Program Files\CMake\bin\ctest.exe" --config Release --build_shared_lib --parallel --compile_no_warning_as_error --skip_tests --use_mimalloc --use_dml

Currently with problems:
--build_nuget and --use_extensions
```

The result will be in `build\Windows\Release\Release`
