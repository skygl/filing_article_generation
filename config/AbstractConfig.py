import json
import os
import re

import pytorch_lightning as pl
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


def collate_fn(batch):
    inputs, pad_token_ids = zip(*batch)
    pad_token_id = pad_token_ids[0]
    encoder_input_ids_tensors, encoder_attention_mask_tensors, decoder_input_ids_tensors, \
    decoder_attention_mask_tensors, labels_tensors = zip(*inputs)

    batch_first = True

    encoder_input_ids = pad_sequence(encoder_input_ids_tensors, batch_first=batch_first,
                                     padding_value=pad_token_id)
    encoder_attention_mask = pad_sequence(encoder_attention_mask_tensors, batch_first=batch_first,
                                          padding_value=0)
    decoder_input_ids = pad_sequence(decoder_input_ids_tensors, batch_first=batch_first,
                                     padding_value=pad_token_id)
    decoder_attention_mask = pad_sequence(decoder_attention_mask_tensors, batch_first=batch_first,
                                          padding_value=0)
    labels = pad_sequence(labels_tensors, batch_first=batch_first, padding_value=-100)

    return {
        "encoder_input_ids": encoder_input_ids,
        "encoder_attention_mask": encoder_attention_mask,
        "decoder_input_ids": decoder_input_ids,
        "decoder_attention_mask": decoder_attention_mask,
        "labels": labels
    }


class AbstractConfig(object):
    def __init__(self, args, device='cpu'):
        self.device = torch.device(device)
        self.args = args

    def get_data_module(self):
        max_len = self.args.max_len
        batch_size = self.args.batch_size
        data_dir = self.args.data_dir
        num_workers = self.args.num_workers
        model = self.args.model

        tokenizer = self.tokenizer

        number_representation = self.args.number_representation

        data_module = FilingArticlePairDataModule(tokenizer=tokenizer, max_len=max_len, data_dir=data_dir,
                                                  batch_size=batch_size, num_workers=num_workers, model=model,
                                                  number_representation=number_representation)

        return data_module


class FilingArticlePairDataset(Dataset):
    def __init__(self, data_dir, stage, tokenizer, max_len, model, number_representation, type_code=None):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.model = model
        self.type_code = type_code
        self.number_representation = number_representation

        meta_path = os.path.join(data_dir, stage, 'meta.json')
        if not os.path.exists(meta_path):
            raise f"{stage} meta file not exist"

        self.dataset = self.read_json(meta_path)

        if self.type_code is not None:
            self.dataset = self.filter_type_code(self.dataset)

    def find_number(self, text: str) -> [str]:
        # 소수 or ","를 포함한 숫자(소수) or 두자리 이상 정수
        return re.findall(r"\d+(?:\.\d+)+|\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d{2,}", text)

    def separate_by_digit(self, text: str, ensure_position=False, separate_char=" "):
        """
        :param ensure_position: 숫자 3자리 고정 여부
        :param separate_char: 숫자를 구분하는 문자열 (default: " ")
        """
        def add_empty_space(src: str, ensure_position=False):
            if ensure_position is True:
                src = src.zfill(3)
            trg = separate_char.join(src)
            return trg

        numbers = self.find_number(text)

        for number in numbers:
            splited = number.split(".")
            if len(splited) > 2:
                continue
            # 정수 부분
            before_point = splited[0]
            # 소수 부분
            after_point = "" if len(splited) == 1 else splited[1]
            if before_point.find(",") != -1:
                # 공백 포함
                before_point = f"{separate_char},{separate_char}".join(
                    list(map(lambda x: add_empty_space(x, ensure_position=ensure_position), before_point.split(","))))
            else:
                before_point = add_empty_space(before_point)

            after_point = add_empty_space(after_point)

            changed = before_point if len(after_point) == 0 \
                else f"{before_point}{separate_char}.{separate_char}{after_point}"

            # 해당 문자열만 치환
            text = text.replace(number, changed, 1)

        return text

    def replace_to_korean(self, text: str):
        """
        Reference : https://wikidocs.net/189139
        """
        number_dic_ko = ("", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구")
        place_value1_ko = ("", "십", "백", "천")
        place_value2_ko = ("", "만", "억", "조", "경")

        def split_number(number, n):
            """number를 n자리씩 끊어서 리스트로 반환한다."""
            res = []
            div = 10 ** n
            while number > 0:
                number, remainder = divmod(number, div)
                res.append(remainder)
            return res

        def convert_lt_10000(number, delimiter):
            """10000 미만의 수를 한글로 변환한다.
               delimiter가 ''이면 1을 무조건 '일'로 바꾼다."""
            res = ""
            for place, digit in enumerate(split_number(number, 1)):
                if not digit:
                    continue
                if delimiter and digit == 1 and place != 0:
                    num = ""
                else:
                    num = number_dic_ko[digit]
                res = num + place_value1_ko[place] + res
            return res

        def number_to_word_ko(number, delimiter=" "):
            """0 이상의 number를 한글로 바꾼다.
               delimiter를 ''로 지정하면 1을 '일'로 바꾸고 공백을 넣지 않는다."""
            if number == 0:
                return "영"
            word_list = []
            for place, digits in enumerate(split_number(number, 4)):
                if word := convert_lt_10000(digits, delimiter):
                    word += place_value2_ko[place]
                    word_list.append(word)
            res = delimiter.join(word_list[::-1])
            if delimiter and 10000 <= number < 20000:
                res = res[1:]
            return res

        numbers = self.find_number(text)

        for number in numbers:
            splited = number.split(".")
            if len(splited) > 1:
                continue
            integer = splited[0]
            integer = integer.replace(",", "")

            changed = number_to_word_ko(int(integer), '')

            text = text.replace(number, changed, 1)

        return text

    def replace_number_in_filing_content(self, filing_content: str):
        if self.number_representation is None:
            return filing_content
        elif self.number_representation == "decimal":
            return filing_content
        elif self.number_representation == "char":
            return self.separate_by_digit(filing_content)
        elif self.number_representation == "fixed_char":
            return self.separate_by_digit(filing_content, ensure_position=True)
        elif self.number_representation == "underscore":
            return self.separate_by_digit(filing_content, separate_char="_")
        elif self.number_representation == "words":
            return self.replace_to_korean(filing_content)
        raise NotImplementedError("Not Supported Number Representation!")

    def read_json(self, path):
        with open(path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data

    def filter_type_code(self, dataset: [dict]):
        def filter_(x):
            return x['filing']['detail_type_code'] == self.type_code

        return list(filter(filter_, dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        title = item['filing']['title']
        date = item['filing']['date']
        company_name = item['company']['stock_name']
        if isinstance(company_name, float):
            company_name = item['company']['dart_name']
        filing_content = item['filing_content']
        filing_content = filing_content.replace('\xa0', ' ').replace('\n', ' ').replace('\t', ' ')
        filing_content = self.replace_number_in_filing_content(filing_content)
        article = item['article_content']
        if self.number_representation is not None:
            article = self.separate_by_digit(article)

        y = date // 10000
        m = (date // 100) % 100
        d = date % 100
        ymd = f"{y}년 {m}월 {d}일"

        bos_token_id = self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.eos_token_id

        if self.model == 'kobart':
            sep_token = self.tokenizer.eos_token
        elif self.model == 'kogpt2':
            sep_token = self.tokenizer.pad_token
        else:
            sep_token = self.tokenizer.sep_token or self.tokenizer.pad_token or self.tokenizer.eos_token

        # filing : 제목<sep>회사이름<sep>날짜<sep>공시내용
        filing = title + sep_token + company_name + sep_token + ymd + sep_token + filing_content

        if self.model == 'kogpt2':
            sep_token_id = self.tokenizer.pad_token_id
            encoder_input_ids = self.tokenizer.encode(filing)[:self.max_len - 1]
            encoder_input_ids = encoder_input_ids + [sep_token_id]
        else:
            # encoder_input_ids : <s>~</s>
            encoder_input_ids = self.tokenizer.encode(filing)[:self.max_len - 2]
            encoder_input_ids = [bos_token_id] + encoder_input_ids + [eos_token_id]
        encoder_input_ids = torch.LongTensor(encoder_input_ids)
        encoder_attention_mask = [1] * len(encoder_input_ids)
        encoder_attention_mask = torch.LongTensor(encoder_attention_mask)

        # decoder_input_ids : <s>~
        decoder_input_ids = self.tokenizer.encode(article)[:self.max_len - 1]
        decoder_input_ids = [bos_token_id] + decoder_input_ids
        decoder_input_ids = torch.LongTensor(decoder_input_ids)
        decoder_attention_mask = [1] * len(decoder_input_ids)
        decoder_attention_mask = torch.LongTensor(decoder_attention_mask)

        # labels : ~</s>
        if self.model == 'kogpt2':
            encoder_input_len = len(encoder_input_ids)
            decoder_input_ids = self.tokenizer.encode(article)[:self.max_len - 1] + [eos_token_id]
            labels = [-100] * encoder_input_len + decoder_input_ids
            decoder_input_ids = torch.LongTensor(decoder_input_ids)
            encoder_input_ids = torch.cat([encoder_input_ids, decoder_input_ids], dim=0)
            labels = torch.LongTensor(labels)
        else:
            labels = self.tokenizer.encode(article)[:self.max_len - 1]
            labels = labels + [eos_token_id]
            labels = torch.LongTensor(labels)

        if self.model == 'kobart':
            pad_token_id = self.tokenizer.pad_token_id
        elif self.model == 'kogpt2':
            pad_token_id = self.tokenizer.pad_token_id
        else:
            pad_token_id = self.tokenizer.sep_token_id or self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        return (encoder_input_ids, encoder_attention_mask, decoder_input_ids, decoder_attention_mask, labels), \
               pad_token_id


class FilingArticlePairDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, max_len, data_dir, batch_size, num_workers, model, number_representation):
        super(FilingArticlePairDataModule, self).__init__()

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data_dir = data_dir
        self.model = model
        self.number_representation = number_representation

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.setup()

    def setup(self, stage: str = None) -> None:
        self.trainset = FilingArticlePairDataset(self.data_dir, 'train', self.tokenizer, self.max_len, self.model, self.number_representation)
        self.validset = FilingArticlePairDataset(self.data_dir, 'valid', self.tokenizer, self.max_len, self.model, self.number_representation)
        self.testset = FilingArticlePairDataset(self.data_dir, 'test', self.tokenizer, self.max_len, self.model, self.number_representation)

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,
                          collate_fn=collate_fn)

    def valid_dataloader(self):
        return DataLoader(self.validset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                          collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                          collate_fn=collate_fn)
