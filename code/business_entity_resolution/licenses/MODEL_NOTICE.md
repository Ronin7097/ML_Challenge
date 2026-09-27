# Model provenance and modifications

The encoder derives from IBM Granite embedding 97M multilingual r2, revision
`835ad14087e140460703cf0fae09f97d469d65c2`:
https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2/tree/835ad14087e140460703cf0fae09f97d469d65c2

The upstream model card declares Apache-2.0 and links to the Apache license.
`Granite-model-card.md` is a copy of that revision's README with trailing whitespace normalized.
`Granite-APACHE-2.0.txt` is the license text from
https://www.apache.org/licenses/LICENSE-2.0.txt . No standalone LICENSE file was
present at the pinned model revision. The model card's examples are upstream
documentation, not instructions used to resolve competition records.

DARPA modified the encoder weights through fine-tuning on supplied competition
training pairs and mined negatives. The resulting weights remain Apache-2.0.
The packaged checkpoint SHA-256 is
`192974ed0c8ed02ee2c48dad6dbf3d7447143b18fe7cd91d2bb8b5a106b931bd`.
The 74 stored tensors contain 97,441,152 scalar parameters, counted from the
packaged safetensors header. These counts exclude the much smaller tree models.

DARPA's original LightGBM pair, base-context and rich-context model files are
provided under the MIT license at `../LICENSE`, as is DARPA-authored source.
The copied competition validator retains its upstream authorship; it is not
represented as DARPA-authored code. AnyAscii 0.3.3 is a generic ISC-licensed
Unicode transliteration dependency, not an external business database.
