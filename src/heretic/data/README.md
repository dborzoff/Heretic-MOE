# Built-in evaluation data

`perplexity_reference_v1.txt` is the frozen corpus used by the Perplexity
scorer when `dataset = "builtin://perplexity-reference-v1"`.

- Size: 241,986 bytes
- SHA-256: `49d7e8f6f3eeacc3fd95e8436bb28278746fdfd47994be4d1da46a36a6228fc3`

The scorer verifies this hash before tokenization. Keeping the corpus in the
package makes local and rented-server measurements byte-identical and removes
the former machine-specific `F:/AI/llamacpp/ppl_test.txt` dependency.
