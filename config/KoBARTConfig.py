from kobart import get_kobart_tokenizer
from transformers import AutoTokenizer

from SummarizationModule import SummarizationModule
from config.AbstractConfig import AbstractConfig


class KoBARTConfig(AbstractConfig):
    def __init__(self, args, device='cpu'):
        super(KoBARTConfig, self).__init__(args, device)

        if args.model == 'sk_kobart':
            self.tokenizer = get_kobart_tokenizer()
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

        self.tokenizer.add_special_tokens({"additional_special_tokens": [f"[NUM-{i}]" for i in range(10)]})

    def get_tokenizer(self):
        return self.tokenizer

    def get_model(self):
        args = self.args
        tokenizer = self.tokenizer
        device = self.device

        summarization_module = SummarizationModule(args, tokenizer, device)

        return summarization_module
