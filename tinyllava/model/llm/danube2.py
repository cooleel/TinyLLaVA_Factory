from transformers import MistralForCausalLM, AutoTokenizer

from . import register_llm

@register_llm('danube2')
def return_danube2class():
    def tokenizer_and_post_load(tokenizer):
        tokenizer.pad_token = tokenizer.unk_token
        return tokenizer
    return MistralForCausalLM, (AutoTokenizer, tokenizer_and_post_load)