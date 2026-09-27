---
language:
- ar
- az
- bg
- bn
- ca
- cs
- da
- de
- el
- en
- es
- et
- fa
- fi
- fr
- he
- hi
- hr
- hu
- id
- is
- it
- ja
- ka
- kk
- km
- ko
- lt
- lv
- mr
- ms
- nl
- 'no'
- pl
- pt
- ro
- ru
- sk
- sl
- sq
- sr
- sv
- sw
- te
- th
- tl
- tr
- uk
- ur
- uz
- vi
- zh
library_name: sentence-transformers
license: apache-2.0
pipeline_tag: feature-extraction
tags:
- granite
- embeddings
- transformers
- multilingual
- mteb
- feature-extraction
- sentence-similarity
- onnx
- openvino
---

# Granite-Embedding-97M-Multilingual-R2

**Model Summary:** Granite-Embedding-97M-Multilingual-R2 is a 97M parameter dense embedding model from the Granite Embeddings collection for high-quality multilingual text embeddings at minimal compute cost. It produces 384-dimensional vectors with a context length of up to 32,768 tokens. The model supports **200+ languages** (based on the multilingual pretraining corpus of the underlying encoder), with **enhanced support for 52 languages and programming code** that receive explicit retrieval-pair and cross-lingual training. All training data uses permissive, enterprise-friendly licenses, plus IBM-collected and IBM-generated datasets.

> Granite Embedding 97M Multilingual R2 scores **60.3** on [Multilingual MTEB Retrieval (18 tasks)](https://huggingface.co/spaces/mteb/leaderboard) — the highest retrieval score of any open multilingual embedding model under 100M parameters, outperforming the next-best model in its size class (multilingual-e5-small at 50.9) by **+9.4 points** — while being roughly **3× smaller** than the full-size granite-embedding-311m-multilingual-r2. The multilingual R2 model shows strong performance across multilingual information retrieval benchmarks, code retrieval, long-document search, conversational multi-turn, and reasoning retrieval tasks.

### What's New in R2

- **Architecture upgrade:** ModernBERT replaces XLM-RoBERTa, bringing alternating attention, SiLU activations, and rotary position embeddings.
- **Extended context:** 32,768 tokens (up from 512 in R1), enabling long-document and multi-passage retrieval.
- **Compact multilingual vocabulary:** A purpose-trained 180K-token tokenizer preserves broad multilingual coverage while reducing model size.
- **Model pruning:** Layer pruning (22 → 12 layers) from the full-size multilingual model, followed by continued distillation training to recover quality.
- **Broader code coverage:** Code retrieval training set that includes Python, Go, Java, JavaScript, PHP, Ruby, SQL, C, C++.
- **Training advances:** Knowledge distillation from multiple teachers and contrastive fine-tuning yield a **+14.6 point** average gain over the previous-generation granite-embedding-107m-multilingual.
- **Deployment flexibility:** Released with ONNX and OpenVINO models; compatible with vLLM and llama.cpp (GGUF).

The model uses a bi-encoder architecture to generate high-quality embeddings from text inputs such as queries, passages, code, and documents, enabling seamless comparison through cosine similarity. Built using contrastive fine-tuning, knowledge distillation, model pruning, and vocabulary selection, granite-embedding-97m-multilingual-r2 is optimized to ensure strong alignment between query and passage embeddings across many languages while maintaining a compact model size.

The Granite Embedding Multilingual R2 release consists of two multilingual embedding models, both based on the ModernBERT architecture:

- _granite-embedding-311m-multilingual-r2_ (**311M** parameters): with an output embedding size of _768_, replacing _granite-embedding-278m-multilingual_. See [granite-embedding-311m-multilingual-r2](https://huggingface.co/ibm-granite/granite-embedding-311m-multilingual-r2).
- **_granite-embedding-97m-multilingual-r2_** (**97M** parameters): A reduced-size multilingual model built via layer pruning and vocabulary selection from the larger model, with fewer layers and a smaller output embedding size (_384_).

## Model Details

- **Developed by:** Granite Embedding Team, IBM
- **Repository:** [ibm-granite/granite-embedding-models](https://github.com/ibm-granite/granite-embedding-models)
- **Project Page:** [IBM Granite](https://www.ibm.com/granite)
- **Paper:** [Granite Embedding Multilingual R2 Models](https://huggingface.co/papers/2605.13521)
- **Language(s) (NLP):** 200+ languages supported, with enhanced support for 52 languages and programming code (see [full language list](#supported-languages))
- **Release Date**: April 29, 2026
- **License:** [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)

### Supported Languages

The underlying encoder was pretrained on text from **200+ languages**, and we report general-purpose embeddings for any of them. In addition, we provide **enhanced support for 52 languages and programming code** that receive explicit retrieval-pair and cross-lingual training data, producing higher-quality embeddings on retrieval tasks.

<details>
<summary>Click to expand the list of 52 enhanced-support languages</summary>

Albanian (sq), Arabic (ar), Azerbaijani (az), Bengali (bn), Bulgarian (bg), Catalan (ca), Chinese (zh), Croatian (hr), Czech (cs), Danish (da), Dutch (nl), English (en), Estonian (et), Finnish (fi), French (fr), Georgian (ka), German (de), Greek (el), Hebrew (he), Hindi (hi), Hungarian (hu), Icelandic (is), Indonesian (id), Italian (it), Japanese (ja), Kazakh (kk), Khmer (km), Korean (ko), Latvian (lv), Lithuanian (lt), Malay (ms), Marathi (mr), Norwegian (no), Persian (fa), Polish (pl), Portuguese (pt), Romanian (ro), Russian (ru), Serbian (sr), Slovak (sk), Slovenian (sl), Spanish (es), Swahili (sw), Swedish (sv), Tagalog (tl), Telugu (te), Thai (th), Turkish (tr), Ukrainian (uk), Urdu (ur), Uzbek (uz), Vietnamese (vi).

Additionally, the models are trained on **programming code** (Python, Go, Java, JavaScript, PHP, Ruby, SQL, C, C++) and support cross-lingual code retrieval.

</details>

### When to Use This Model

- **Use granite-embedding-97m-multilingual-r2** for latency-sensitive production workloads, edge deployment, or when you need maximum encoding throughput with competitive multilingual quality. At 97M parameters, this model is 3× smaller than the full-size granite-embedding-311m-multilingual-r2 while preserving strong cross-lingual retrieval performance.
- **Use [granite-embedding-311m-multilingual-r2](https://huggingface.co/ibm-granite/granite-embedding-311m-multilingual-r2)** when accuracy is the top priority, and you can afford the throughput of a 311M-parameter model.
- **Use [granite-embedding-english-r2](https://huggingface.co/ibm-granite/granite-embedding-english-r2) or [granite-embedding-small-english-r2](https://huggingface.co/ibm-granite/granite-embedding-small-english-r2)** when your data is predominantly English, as these English-specific models offer optimized performance for monolingual English use cases.

## Usage

**Intended Use:** The model is designed to produce fixed-length vector representations for a given text, which can be used for text similarity, retrieval, and search applications across multiple languages.

For efficient inference, these models support Flash Attention 2. Installing it is optional but can lead to faster encoding:

```shell
pip install flash_attn
```

**Usage with Sentence Transformers:**

The model is compatible with the SentenceTransformer library and is very easy to use:

First, install the sentence transformers library

```shell
pip install sentence_transformers
```

The model can then be used to encode pairs of text and find the similarity between their representations

```python
from sentence_transformers import SentenceTransformer, util

model_path = "ibm-granite/granite-embedding-97m-multilingual-r2"
# Load the Sentence Transformer model
model = SentenceTransformer(model_path)

input_queries = [
    'What is the tallest mountain in Japan?',          # English query
    'Wer hat das Lied Achy Breaky Heart geschrieben?', # German query
    'ドイツの首都はどこですか？',                            # Japanese query
    ]

input_passages = [
    "富士山は、静岡県と山梨県にまたがる活火山で、標高3776.12 mで日本最高峰の独立峰である。",  # Japanese passage
    "Achy Breaky Heart is a country song written by Don Von Tress. Originally titled Don't Tell My Heart and performed by The Marcy Brothers in 1991.",  # English passage
    "Berlin ist die Hauptstadt und ein Land der Bundesrepublik Deutschland. Die Stadt ist with rund 3,7 Millionen Einwohnern die bevölkerungsreichste Kommune Deutschlands.",  # German passage
    ]

# Cross-lingual retrieval: each query should score highest with its matching passage in a different language
query_embeddings = model.encode(input_queries)
passage_embeddings = model.encode(input_passages)

# calculate cosine similarity — expect high scores on the diagonal (EN→JA, DE→EN, JA→DE)
print(util.cos_sim(query_embeddings, passage_embeddings))
# output: tensor([[0.8869, 0.6658, 0.7213],
#         [0.6792, 0.9577, 0.6420],
#        [0.7534, 0.6771, 0.9112]])
```

**Usage with Hugging Face Transformers:**

This is a simple example of how to use the granite-embedding-97m-multilingual-r2 model with the Transformers library and PyTorch. For a complete retrieval workflow including passage encoding and cosine similarity, see the Sentence Transformers example above.

First, install the required libraries

```shell
pip install transformers torch
```

The model can then be used to encode text

```python
import torch
from transformers import AutoModel, AutoTokenizer

model_path = "ibm-granite/granite-embedding-97m-multilingual-r2"

# Load the model and tokenizer
model = AutoModel.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()

input_queries = [
    'What is the tallest mountain in Japan?',          # English query
    'Wer hat das Lied Achy Breaky Heart geschrieben?', # German query
    'ドイツの首都はどこですか？',                            # Japanese query
    ]

# tokenize inputs
tokenized_queries = tokenizer(input_queries, padding=True, truncation=True, return_tensors='pt')

# encode queries
with torch.no_grad():
    model_output = model(**tokenized_queries)
    # Perform pooling. granite-embedding-97m-multilingual-r2 uses CLS Pooling
    query_embeddings = model_output[0][:, 0]

# normalize the embeddings
query_embeddings = torch.nn.functional.normalize(query_embeddings, dim=1)
```

### Optimized Inference and Deployment

**ONNX and OpenVINO:**

Pre-converted ONNX and OpenVINO models are released alongside the PyTorch weights for production deployment. These can be loaded directly via the `backend` parameter in Sentence Transformers:

```python
from sentence_transformers import SentenceTransformer

# ONNX backend
model = SentenceTransformer("ibm-granite/granite-embedding-97m-multilingual-r2", backend="onnx")
embeddings = model.encode(["example text"])

# OpenVINO backend
model = SentenceTransformer("ibm-granite/granite-embedding-97m-multilingual-r2", backend="openvino")
embeddings = model.encode(["example text"])

# OpenVINO INT8 quantized backend (smaller & faster on CPU)
model = SentenceTransformer(
    "ibm-granite/granite-embedding-97m-multilingual-r2",
    backend="openvino",
    model_kwargs={"file_name": "openvino/openvino_model_qint8_quantized.xml"},
)
embeddings = model.encode(["example text"])
```

The ONNX model is compatible with any ONNX Runtime backend (CPU, CUDA, TensorRT, DirectML). The OpenVINO model is optimized for Intel hardware including CPUs and integrated GPUs.

**vLLM:**

The model can be served as an embedding endpoint using [vLLM](https://docs.vllm.ai/):

```shell
vllm serve ibm-granite/granite-embedding-97m-multilingual-r2 --task embed
```

**llama.cpp (GGUF):**

The model can be converted to GGUF format for use with [llama.cpp](https://github.com/ggerganov/llama.cpp):

```shell
# Convert to GGUF
python convert_hf_to_gguf.py ibm-granite/granite-embedding-97m-multilingual-r2 \
    --outfile granite-embedding-97m-multilingual-r2.gguf

# Generate embeddings
llama-embedding -m granite-embedding-97m-multilingual-r2.gguf -p "example text"
```

Note: Ollama does not currently support ModernBERT-based models.

## Evaluation Results

Granite-Embedding-97M-Multilingual-R2 delivers strong retrieval quality at minimal compute cost. At 97M parameters and 384-dimensional embeddings, it offers a compelling accuracy-efficiency tradeoff: it matches the retrieval quality of gte-multilingual-base (a 305M model) at nearly 3× the encoding speed, and gains **+14.6 points** on average over its predecessor granite-embedding-107m-multilingual.

### Multilingual Retrieval Performance

Performance on Multilingual MTEB Retrieval, MTEB English Retrieval, MTEB Code Retrieval, long-document search (LongEmbed), and Reasoning as Retrieval (RaR-b) benchmarks. Scores are averages across tasks; higher is better. Throughput (documents per second) was measured on a single NVIDIA H100 GPU using a sliding window with 512-token chunks.

At nearly 2,900 documents per second, granite-embedding-97m-multilingual-r2 delivers comparable throughput to its R1 predecessor while gaining close to 10 points on multilingual retrieval. It retains the majority of the full-size 311M model's retrieval quality at roughly 3× smaller size and 1.5× higher throughput.

| Model                                     | Parameters (M) | Embedding Size | MTEB ML Retrieval (18) | MTEB Retrieval (eng, v2) (10) | MTEB (Code, v1) (12) | LongEmbed (6) | RaR-b (17) | **AVG**  | Throughput (docs/s) |
| ----------------------------------------- | -------------- | -------------- | ---------------------- | ----------------------------- | -------------------- | ------------- | ---------- | -------- | ------------------: |
| granite-embedding-107m-multilingual       | 107            | 384            | 48.1                   | 47.9                          | 40.7                 | 34.3          | 17.1       | 37.6     |               3,113 |
| granite-embedding-278m-multilingual       | 278            | 768            | 52.2                   | 51.5                          | 48.5                 | 37.7          | 18.9       | 41.8     |               2,164 |
| granite-embedding-311m-multilingual-r2    | 311            | 768            | 65.2                   | 52.6                          | 63.8                 | 71.7          | 28.0       | 56.3     |               1,828 |
| **granite-embedding-97m-multilingual-r2** | **97**         | **384**        | **60.3**               | **50.1**                      | **60.4**             | **65.5**      | **24.9**   | **52.2** |           **2,534** |

## Model Architecture and Key Features

The Granite Embedding Multilingual R2 release consists of two multilingual embedding models, both based on the ModernBERT architecture. The 97M model is derived from the 311M model via layer pruning (22 → 12 layers) and vocabulary selection (262K → 180K tokens), using a compact tokenizer purpose-trained for multilingual coverage at reduced size. The 97M model uses SiLU activation (from the pruned architecture) rather than GeGLU used in the full-size model.

| Feature                   | **granite-embedding-97m-multilingual-r2** | granite-embedding-311m-multilingual-r2 |
| :------------------------ | :---------------------------------------: | :------------------------------------: |
| Embedding size            |                  **384**                  |                  768                   |
| Number of layers          |                  **12**                   |                   22                   |
| Number of attention heads |                  **12**                   |                   12                   |
| Intermediate size         |                 **1536**                  |                  1152                  |
| Activation Function       |                 **SiLU**                  |                 GeGLU                  |
| Vocabulary Size           |                **180,000**                |                262,152                 |
| Max. Sequence Length      |                **32,768**                 |                 32,768                 |
| Matryoshka Dimensions     |                     —                     |        768, 512, 384, 256, 128         |
| # Parameters              |                 **~97M**                  |                 ~311M                  |

## Training and Optimization

The Granite Embedding Multilingual R2 models incorporate key enhancements from the ModernBERT architecture, including:

- Alternating attention lengths to accelerate processing
- Rotary position embeddings for extended sequence length
- A multilingual tokenizer trained on code and text data across 200+ languages
- Flash Attention 2.0 for improved efficiency
- Streamlined parameters, eliminating unnecessary bias terms

The 97M model was built via **layer pruning and vocabulary selection**, starting from the larger granite-embedding-311m-multilingual-r2 and reducing from 22 to 12 transformer layers. A compact, purpose-trained multilingual tokenizer (180K vocabulary) preserves broad multilingual coverage while significantly reducing parameter count. The pruned model was then trained using knowledge distillation with multiple teacher models and contrastive fine-tuning to recover retrieval quality, retaining the majority of the full-size model's performance at 3× smaller size.

## Data Collection

All training data is sourced under permissive, commercial-friendly licenses, making Granite Embedding R2 suitable for unrestricted enterprise deployment.

Training data comes from four key sources:

1. Unsupervised title-body paired data scraped from the web
2. Publicly available paired data with permissive, enterprise-friendly licenses
3. IBM-internal paired data targeting specific technical domains
4. IBM-generated multilingual synthetic data including long-document pairs

For governance, all our data undergoes a data clearance process subject to technical, business, and governance review. This comprehensive process captures critical information about the data, including but not limited to their content description, ownership, intended use, data classification, licensing information, usage restrictions, how the data will be acquired, as well as an assessment of sensitive information (e.g., personal information).

## Infrastructure

We trained the Granite Embedding Multilingual R2 models using IBM's computing cluster, BlueVela Cluster, which is outfitted with NVIDIA H100 80GB GPUs. This cluster provides a scalable and efficient infrastructure for training our models over multiple GPUs.

## Ethical Considerations and Limitations

Granite Embedding 97M Multilingual R2 leverages both permissively licensed open-source and select proprietary data for enhanced performance. The training data for the base language model was filtered to remove text containing hate, abuse, and profanity, though the effectiveness of such filtering may vary across language families.

As a pruned model, granite-embedding-97m-multilingual-r2 trades some accuracy for significantly faster inference and lower resource requirements. Performance varies across languages: higher-resource languages and those in the 52-language enhanced-support set generally achieve better results, while low-resource languages rely on cross-lingual transfer and may exhibit lower retrieval quality — an effect that is more pronounced in this smaller model than in the full-size 311M variant. The reduced vocabulary (180K vs. 262K tokens) may also affect tokenization efficiency for some languages. Longer texts will be truncated to the 32,768-token context limit.

## Resources

- Learn about the latest updates with Granite: https://www.ibm.com/granite
- Get started with tutorials, best practices, and prompt engineering advice: https://www.ibm.com/granite/docs/
- Learn about the latest Granite learning resources: https://ibm.biz/granite-learning-resources

## Citation

```
@misc{awasthy2026graniteembeddingmultilingualr2,
      title={Granite Embedding Multilingual R2 Models},
      author={Parul Awasthy and Aashka Trivedi and Yushu Yang and Ken Barker and Yulong Li and Bhavani Iyer and Martin Franz and Juergen Bross and Meet Doshi and Vignesh P and Vishwajeet Kumar and Todd Ward and Abraham Daniels and Madison Lee and Luis Lastras and Jaydeep Sen and Radu Florian},
      year={2026},
      eprint={2605.13521},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2605.13521},
}
```
