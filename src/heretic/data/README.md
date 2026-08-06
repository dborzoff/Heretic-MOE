# Built-in evaluation data

`perplexity_reference_v1.txt` is the frozen corpus used by the Perplexity
scorer when `dataset = "builtin://perplexity-reference-v1"`.

- Size: 241,748 bytes (canonical LF form)
- SHA-256: `1d6f25ca80bd49255212d67d7eff96763ab01abbd472c04b916ec62318857a9d`

The scorer verifies this hash before tokenization. Keeping the corpus in the
package makes local and rented-server measurements byte-identical and removes
the former machine-specific `F:/AI/llamacpp/ppl_test.txt` dependency.
