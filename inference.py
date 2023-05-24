import argparse
import os
import random

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from config import KoBARTConfig

from SummarizationModule import SummarizationModule


def set_random_seed(random_seed):
    np.random.seed(random_seed)
    random.seed(random_seed)

    pl.seed_everything(random_seed)

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    print(f"Set random seed : {random_seed}")


models = {
    'sk_kobart': KoBARTConfig.KoBARTConfig,
    'gogamza/kobart-base-v1': KoBARTConfig.KoBARTConfig,
    'gogamza/kobart-base-v2': KoBARTConfig.KoBARTConfig,
}


def inference(model, dataloader: DataLoader, tokenizer, inference_count=1):
    dataset = dataloader.dataset

    for i in range(inference_count):
        data = dataset[i]

        input_ids = data[0][0]
        input_ids = input_ids.unsqueeze(0)

        beam_output = model.generate(
            input_ids,
            max_length=512,
            num_beams=3,
            early_stopping=True
        )

        print("Output:\n" + 100 * '-')
        print(tokenizer.decode(beam_output[0], skip_special_tokens=True))
        print(100 * '-' + "\n\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--tokenizer', type=str, default='gogamza/kobart-base-v1')

    parser.add_argument('--dir_path', type=str, default='./checkpoints')
    parser.add_argument('--checkpoint', type=str, default='last.ckpt')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=1e-5)

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--inference_count', type=int, default=16)
    parser.add_argument('--use_cpu', action='store_true')

    args = parser.parse_args()

    model_name = args.model
    dir_path = args.dir_path
    seed = args.seed
    checkpoint = args.checkpoint
    inference_count = args.inference_count
    use_cpu = args.use_cpu

    if model_name not in models:
        raise f"model ${model_name} is not supported"

    device = torch.device("cpu" if use_cpu else "cuda")

    config_class = models[model_name]
    config = config_class(args, device)

    model_path = os.path.join(dir_path, checkpoint)

    config_class = models[model_name]
    config = config_class(args, device)

    tokenizer = config.get_tokenizer()
    model = SummarizationModule.load_from_checkpoint(model_path)
    data_module = config.get_data_module()

    test_data_loader = data_module.test_dataloader()

    inference(model, test_data_loader, tokenizer, inference_count=inference_count)
