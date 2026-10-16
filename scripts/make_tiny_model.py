"""Create a tiny, randomly initialised Llama-style model + tokenizer on disk.

Useful for smoke-testing the whole pipeline on CPU or without Hugging Face access:

    python scripts/make_tiny_model.py ./tiny-model
    # then use "./tiny-model" (absolute path) as the base model
"""
import sys
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

SPECIALS = ["<|pad|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]
CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def main(out: str):
    out = Path(out)
    corpus = [p.read_text() for p in Path(__file__).parent.parent.glob("examples/*")] * 20
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(corpus, trainers.BpeTrainer(
        vocab_size=1000, special_tokens=SPECIALS, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    hf_tok = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<|pad|>", eos_token="<|im_end|>",
                                     bos_token="<|im_start|>")
    hf_tok.chat_template = CHAT_TEMPLATE
    cfg = LlamaConfig(vocab_size=hf_tok.vocab_size if len(hf_tok) <= hf_tok.vocab_size else len(hf_tok),
                      hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                      pad_token_id=hf_tok.pad_token_id, eos_token_id=hf_tok.eos_token_id,
                      bos_token_id=hf_tok.bos_token_id)
    LlamaForCausalLM(cfg).save_pretrained(out)
    hf_tok.save_pretrained(out)
    print(f"tiny model written to {out.resolve()}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tiny-model")
