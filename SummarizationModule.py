import os
from pathlib import Path
from typing import Union, Tuple, Optional, List

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from kobart import get_pytorch_kobart_model
from torch import nn
from torch.nn import CrossEntropyLoss
from torchmetrics.text.rouge import ROUGEScore
from transformers import AdamW, T5ForConditionalGeneration, GPT2LMHeadModel, BartConfig
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.bart.modeling_bart import BartModel, BartPretrainedModel, shift_tokens_right

BASE_PATH = Path(__file__).parent


class SummarizationModule(pl.LightningModule):
    def __init__(self, args, tokenizer, device):
        super().__init__()
        self._device = device

        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size

        if args.model == 'sk_kobart':
            self.model = BartPGNForConditionalGeneration.from_pretrained(get_pytorch_kobart_model())
        elif args.model in ['gogamza/kobart-base-v1', 'gogamza/kobart-base-v2']:
            self.model = BartPGNForConditionalGeneration.from_pretrained(args.model)
        elif args.model == 'et5':
            model_path = os.path.join(BASE_PATH, '.cache', 'et5')
            self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        elif args.model == 'kogpt2':
            self.model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

        self.tokenizer = tokenizer

        self.rouge = ROUGEScore()

        self.save_hyperparameters()

    def forward(
            self,
            encoder_input_ids,
            encoder_attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            labels=None
    ):
        if isinstance(self.model, T5ForConditionalGeneration):
            outputs = self.model(
                input_ids=encoder_input_ids,
                labels=labels,
            )
        elif isinstance(self.model, GPT2LMHeadModel):
            outputs = self.model(
                input_ids=encoder_input_ids,
                labels=labels
            )
        else:
            outputs = self.model(
                input_ids=encoder_input_ids,
                attention_mask=encoder_attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels
            )

        return outputs

    def step(self, batch, batch_idx, state):
        outputs = self(
            encoder_input_ids=batch['encoder_input_ids'].to(self._device),
            encoder_attention_mask=batch['encoder_attention_mask'].to(self._device),
            decoder_input_ids=batch['decoder_input_ids'].to(self._device),
            decoder_attention_mask=batch['decoder_attention_mask'].to(self._device),
            labels=batch['labels'].to(self._device)
        )

        loss = outputs.loss
        logits = outputs.logits  # shape : (batch, len_decoder_inputs, vocab)

        preds = torch.argmax(logits, dim=2).to(self._device)  # shape : (batch, len_decoder_inputs)
        targets = batch['decoder_input_ids'].to(self._device)

        self.log(f"[{state.upper()} LOSS]", loss, prog_bar=True)
        return {
            'loss': loss,
            'preds': preds,
            'targets': targets
        }

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'valid')

    def training_epoch_end(self, outputs, state='train'):
        train_loss = torch.tensor(0, dtype=torch.float)
        train_rouge1 = torch.tensor(0, dtype=torch.float)
        train_rouge2 = torch.tensor(0, dtype=torch.float)
        train_rougeL = torch.tensor(0, dtype=torch.float)
        total = 0
        for idx in range(len(outputs)):
            for batch_item in range(len(outputs[idx]['preds'])):
                pred = self.tokenizer.decode(outputs[idx]['preds'][batch_item])
                target = self.tokenizer.decode(outputs[idx]['targets'][batch_item])

                rouge_scores = self.rouge(pred, target)
                train_rouge1 += rouge_scores['rouge1_fmeasure']
                train_rouge2 += rouge_scores['rouge2_fmeasure']
                train_rougeL += rouge_scores['rougeL_fmeasure']
                total += 1
            train_loss += outputs[idx]['loss'].cpu().detach()

        train_loss = train_loss / len(outputs)
        train_rouge1 = train_rouge1 / total
        train_rouge2 = train_rouge2 / total
        train_rougeL = train_rougeL / total

        print(
            f'[Epoch {self.trainer.current_epoch} {state.upper()}] Loss:{train_loss}, ROUGE1: {train_rouge1}, ROUGE2: {train_rouge2}, ROUGEL: {train_rougeL}')

    def validation_epoch_end(self, outputs, state='valid'):
        valid_loss = torch.tensor(0, dtype=torch.float)
        valid_rouge1 = torch.tensor(0, dtype=torch.float)
        valid_rouge2 = torch.tensor(0, dtype=torch.float)
        valid_rougeL = torch.tensor(0, dtype=torch.float)
        total = 0
        for idx in range(len(outputs)):
            for batch_item in range(len(outputs[idx]['preds'])):
                pred = self.tokenizer.decode(outputs[idx]['preds'][batch_item])
                target = self.tokenizer.decode(outputs[idx]['targets'][batch_item])

                rouge_scores = self.rouge(pred, target)
                valid_rouge1 += rouge_scores['rouge1_fmeasure']
                valid_rouge2 += rouge_scores['rouge2_fmeasure']
                valid_rougeL += rouge_scores['rougeL_fmeasure']
                total += 1
            valid_loss += outputs[idx]['loss'].cpu().detach()
        valid_loss = valid_loss / len(outputs)
        valid_rouge1 = valid_rouge1 / total
        valid_rouge2 = valid_rouge2 / total
        valid_rougeL = valid_rougeL / total

        # save best by VAL_LOSS
        self.log('VAL_LOSS', valid_loss, on_epoch=True, prog_bar=True)
        self.log('VAL_ROUGE1', valid_rouge1, on_epoch=True, prog_bar=True)
        self.log('VAL_ROUGE2', valid_rouge2, on_epoch=True, prog_bar=True)
        self.log('VAL_ROUGEL', valid_rougeL, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=150)
        return [optimizer], [lr_scheduler]

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)


class BARTPGNAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(BARTPGNAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden_states, encoder_hidden_states):
        # encoder_hidden_states: [bs, input_token_len, hidden_dim]
        # decoder_hidden_states: [bs, output_token_len, hidden_dim]
        input_token_len = encoder_hidden_states.shape[1]
        output_token_len = decoder_hidden_states.shape[1]

        # encoder_hidden_states: [bs, output_token_len, input_token_len, hidden_dim]
        # decoder_hidden_states: [bs, output_token_len, input_token_len, hidden_dim]
        encoder_hidden_states = encoder_hidden_states.unsqueeze(1).repeat(1, output_token_len, 1, 1)
        decoder_hidden_states = decoder_hidden_states.unsqueeze(2).repeat(1, 1, input_token_len, 1)

        # concat: [bs, output_token_len, input_token_len, hidden_dim*2]
        # energy: [bs, output_token_len, input_token_len, hidden_dim]
        concat = torch.cat((encoder_hidden_states, decoder_hidden_states), dim=-1)
        energy = self.attn(concat)
        energy = torch.tanh(energy)

        # attn_weight: [bs, output_token_len, input_token_len, 1]
        # attn_weight: [bs, output_token_len, input_token_len]
        logits = self.v(energy)
        logits = logits.squeeze(-1)
        attn_weight = F.softmax(logits, dim=-1)

        return attn_weight, logits


class BartPGNForConditionalGeneration(BartPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"lm_head.weight",
        "encoder.embed_tokens.weight",
        "decoder.embed_tokens.weight",
    ]

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = BartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        self.hidden_dim = config.d_model
        self.vocab_size = config.vocab_size
        self.attn = BARTPGNAttention(hidden_dim=self.hidden_dim)

        self.sigmoid = nn.Sigmoid()

        self.pointer_gen = nn.Linear(2 * self.hidden_dim, 1)

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        lm_logits = self.lm_head(outputs[0])
        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)

        # encoder_last_hidden_states: [bs, input_token_len, hidden_dim]
        encoder_last_hidden_states = outputs.encoder_last_hidden_state
        input_token_len = encoder_last_hidden_states.shape[1]

        # decoder_last_hidden_states: [bs, output_token_len, hidden_dim]
        decoder_last_hidden_states = outputs.last_hidden_state
        output_token_len = decoder_last_hidden_states.shape[1]

        # attn_weights: [bs, output_token_len, input_token_len]
        # attn_logits: [bs, output_token_len, input_token_len]
        attn_weights, attn_logits = self.attn(decoder_last_hidden_states, encoder_last_hidden_states)

        # context: [bs, output_token_len, hidden_dim]
        context = torch.bmm(attn_weights, encoder_last_hidden_states)

        # concat: [bs, output_token_len, hidden_dim*2]
        # p_gen: [bs, output_token_len, 1]
        # p_gen: [bs, output_token_len]
        concat = torch.cat((context, decoder_last_hidden_states), dim=-1)
        p_gen = self.pointer_gen(concat)
        p_gen = p_gen.squeeze(-1)
        p_gen = self.sigmoid(p_gen)

        # input_vocab_mask: [bs, input_token_len, vocab_size]
        # input_vocab_mask: [bs, vocab_size, input_token_len]
        # input_vocab_mask: [bs, 1, vocab_size, input_token_len]
        # input_vocab_mask: [bs, output_token_len, vocab_size, input_token_len]
        input_vocab_mask = F.one_hot(input_ids, num_classes=self.vocab_size)
        input_vocab_mask = input_vocab_mask.transpose(1, 2)
        input_vocab_mask = input_vocab_mask.unsqueeze(1)
        input_vocab_mask = input_vocab_mask.repeat(1, output_token_len, 1, 1)

        # p_copy: [bs, output_token_len, input_token_len, 1]
        # p_copy: [bs*output_token_len, vocab_size, 1]
        # p_copy: [bs, output_token_len, vocab_size, 1]
        # p_copy: [bs, output_token_len, vocab_size]
        p_copy = attn_logits.unsqueeze(-1)
        input_vocab_mask_ = input_vocab_mask.view(-1, self.vocab_size, input_token_len)
        p_copy_ = p_copy.view(-1, input_token_len, 1)
        p_copy = torch.bmm(input_vocab_mask_, p_copy_)
        p_copy = p_copy.view(-1, output_token_len, self.vocab_size, 1)
        p_copy = p_copy.squeeze(-1)

        p_w_logits = p_gen * lm_logits + (1 - p_gen) * p_copy

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(p_w_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (p_w_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=p_w_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past_key_values is used
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past
