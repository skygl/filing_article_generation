from SummarizationModule import SummarizationModule
from config.AbstractConfig import AbstractConfig

from transformers import PreTrainedTokenizerFast


class KoGPTConfig(AbstractConfig):
    def __init__(self, args, device='cpu'):
        super(KoGPTConfig, self).__init__(args, device)

        if args.model == 'kogpt2':
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
                "skt/kogpt2-base-v2",
                bos_token='</s>', eos_token='</s>', pad_token='<pad>', unk_token='<unk>', mask_token='<mask>',
            )

    def get_tokenizer(self):
        return self.tokenizer

    def get_model(self):
        args = self.args
        tokenizer = self.tokenizer
        device = self.device

        summarization_module = SummarizationModule(args, tokenizer, device)

        return summarization_module
