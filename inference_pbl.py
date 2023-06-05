import argparse
import json
import os
import random
import re
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from kobart import get_kobart_tokenizer
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from SummarizationModule import SummarizationModule
from config.AbstractConfig import FilingArticlePairDataset

BASE_PATH = Path(__file__).parent


def set_random_seed(random_seed):
    np.random.seed(random_seed)
    random.seed(random_seed)

    pl.seed_everything(random_seed)

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    print(f"Set random seed : {random_seed}")


def get_tokenizer(model_arg: str, tokenizer_arg: str):
    if model_arg == 'sk_kobart':
        return get_kobart_tokenizer()
    elif model_arg == 'kobart':
        return AutoTokenizer.from_pretrained(tokenizer_arg)
    elif model_arg == 'kogpt2':
        return PreTrainedTokenizerFast.from_pretrained(
            "skt/kogpt2-base-v2",
            bos_token='</s>', eos_token='</s>', pad_token='<pad>', unk_token='<unk>', mask_token='<mask>',
        )
    raise NotImplementedError(f"{model_arg} not supported!")


def load_model(dir_path, checkpoint_name):
    model_path = os.path.join(dir_path, checkpoint_name)
    model = SummarizationModule.load_from_checkpoint(model_path)
    return model


def generate_article(model: SummarizationModule, model_args: str, input_ids: [int], tokenizer, max_len, device) -> str:
    if model_args == 'kobart' or model_args == 'sk_kobart':
        input_ids_tensor = torch.LongTensor(input_ids)
        input_ids_tensor = input_ids_tensor.to(device)

        # input_ids : [1, n_tokens]
        input_ids_tensor = input_ids_tensor.unsqueeze(0)

        beam_output = model.generate(
            input_ids_tensor,
            max_length=max_len,
            num_beams=10,
            early_stopping=True
        )

        article = tokenizer.decode(beam_output[0], skip_special_tokens=True)

        return article
    elif model_args == 'kogpt2':
        input_ids_tensor = torch.LongTensor(input_ids)

        # input_ids : [1, n_tokens]
        input_ids_tensor = input_ids_tensor.unsqueeze(0)

        beam_output = model.generate(
            input_ids_tensor,
            max_length=max_len * 2,
            num_beams=3,
            early_stopping=True
        )

        if input_ids_tensor.shape[1] == beam_output.shape[1]:
            return ''

        beam_output = beam_output[0, input_ids_tensor.shape[1]:]

        filing_with_article = tokenizer.decode(beam_output, skip_special_tokens=True)

        return filing_with_article
    else:
        raise NotImplementedError(f"{model_args} not supported")


def find_number(text: str):
    return re.findall(r"\d(?:\s\d)*\.(?:\s\d)+|\d(?:\s\d)+", text)


def recover_number(text: str):
    remove_idx = []
    for i in range(2, len(text)):
        if text[i].isdigit() and text[i - 1] == ' ' and text[i - 2].isdigit():
            remove_idx.append(i - 1)
        elif text[i].isdigit() and text[i - 1] == ' ' and text[i - 2] == '.' and i >= 3 and text[i - 3].isdigit():
            remove_idx.append(i - 1)
    recovered_text = ""
    for i in range(len(text)):
        if i not in remove_idx:
            recovered_text += text[i]
    return recovered_text


def generate(model: SummarizationModule, testset: FilingArticlePairDataset, tokenizer, max_len, model_args, device):
    model.eval()

    results = []

    for idx in tqdm(range(len(testset))):
        data = testset.dataset[idx]

        filing_id = data['filing']['id']
        input_ids = testset.get_input(idx).tolist()

        generated_article = generate_article(model, model_args, input_ids, tokenizer, max_len, device)
        generated_article = recover_number(generated_article)

        result = {
            "filing_id": filing_id,
            "article_content": generated_article,
        }

        results.append(result)

    return results


def get_torch_device(use_cpu, gpu_numbers):
    device = torch.device('cpu')
    if use_cpu is True:
        return device

    if isinstance(gpu_numbers, str):
        gpu_numbers = gpu_numbers.split(',')

    if torch.cuda.is_available() and len(gpu_numbers) > 0:
        available_gpus = torch.cuda.device_count()
        selected_gpus = [int(gpu.strip()) for gpu in gpu_numbers]
        valid_gpus = [gpu for gpu in selected_gpus if gpu < available_gpus]
        if len(valid_gpus) > 0:
            device = torch.device(f'cuda:{valid_gpus[0]}')

    return device


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--tokenizer', type=str, default='gogamza/kobart-base-v1')

    parser.add_argument('--gpu_list', type=str, default='0',
                        help="string; make list by splitting by ','")  # gpu list to be used
    parser.add_argument('--dir_path', type=str, default='./checkpoints')
    parser.add_argument('--result_dir', type=str, default='./result')
    parser.add_argument('--checkpoint', type=str, default='last.ckpt')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--max_len', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=1e-5)

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--type_code', type=str, default=None)
    parser.add_argument('--number_representation', type=str, default=None)
    parser.add_argument('--use_cpu', action='store_true')
    parser.add_argument('--preprocess_table', action='store_true')

    args = parser.parse_args()

    model_name = args.model
    tokenizer_arg = args.tokenizer

    tokenizer = get_tokenizer(model_name, tokenizer_arg)

    dir_path = args.dir_path
    seed = args.seed
    checkpoint = args.checkpoint

    model = load_model(dir_path, checkpoint)

    max_len = args.max_len
    data_dir = args.data_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    number_representation = args.number_representation
    preprocess_table = args.preprocess_table

    test_set = FilingArticlePairDataset(data_dir, 'pbl', tokenizer, max_len, model_name, number_representation,
                                       preprocess_table)

    device = get_torch_device(args.use_cpu, args.gpu_list)
    model = model.to(device)

    testset_result = generate(model, test_set, tokenizer, max_len, model_name, device)

    # 결과 파일 저장
    result_dir = args.result_dir
    result_metric_path = os.path.join(result_dir, f"{checkpoint.split('.')[0]}_pbl.json")
    with open(result_metric_path, 'w', encoding='utf-8') as file:
        json.dump(testset_result, file, ensure_ascii=False)
