import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from langchain.llms import HuggingFacePipeline
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

class Generate():
    def __init__(self, hf_auth='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',
                 llm='TinyLlama/TinyLlama-1.1B-Chat-v0.6', temperature=0.1,
                 max_context_length=512, repetition_penalty=1.1):
        # Generate model
        self.model_id = llm
        self.temperature = temperature
        self.max_new_tokens = max_context_length
        self.repetition_penalty = repetition_penalty
        self.hf_auth = hf_auth
        self.model_config = AutoConfig.from_pretrained(self.model_id, use_auth_token = self.hf_auth, target_lang="vi")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            config=self.model_config,
            device_map = 'auto',
            use_auth_token = self.hf_auth,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_auth_token=self.hf_auth)
    def __call__(self, *args, **kwargs):
        generate_text = transformers.pipeline(
            model = self.model, tokenizer = self.tokenizer,
            return_full_text = True,
            task='text-generation',
            temperature = self.temperature,
            max_new_tokens = self.max_new_tokens,
            repetition_penalty = self.repetition_penalty
        )
        llm = HuggingFacePipeline(pipeline=generate_text)

        return llm