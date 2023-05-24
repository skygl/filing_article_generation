import argparse
import os
import random

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers

from config import KoBARTConfig, KoGPTConfig


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
    'kogpt2': KoGPTConfig.KoGPTConfig,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--tokenizer', type=str, default='gogamza/kobart-base-v1')
    parser.add_argument('--gpu_list', type=str, default='0',
                        help="string; make list by splitting by ','")  # gpu list to be used
    parser.add_argument('--dir_path', type=str, default='./checkpoints')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--use_cpu', action='store_true')
    parser.add_argument('--number_representation', type=str, default=None)

    args = parser.parse_args()

    model_name = args.model
    dir_path = args.dir_path
    seed = args.seed
    max_epoch = args.epoch
    use_cpu = args.use_cpu

    if model_name not in models:
        raise f"model ${model_name} is not supported"

    set_random_seed(random_seed=seed)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='VAL_LOSS',
        dirpath=dir_path,
        filename='{epoch:02d}-{VAL_LOSS:.3f}-{VAL_ROUGE1:.3f}-{VAL_ROUGE2:.3f}-{VAL_ROUGEL:.3f}',
        verbose=False,
        save_last=True,
        mode='min',
        save_top_k=1,
    )

    tb_logger = pl_loggers.TensorBoardLogger(os.path.join(dir_path))
    lr_logger = pl.callbacks.LearningRateMonitor()

    if use_cpu is True:
        trainer = pl.Trainer(
            default_root_dir=os.path.join(dir_path, 'checkpoints'),
            logger=tb_logger,
            callbacks=[checkpoint_callback, lr_logger],
            max_epochs=max_epoch,
        )

        device = torch.device("cpu")

    else:
        gpu_list = [int(gpu) for gpu in args.gpu_list.split(',')]

        trainer = pl.Trainer(
            default_root_dir=os.path.join(dir_path, 'checkpoints'),
            logger=tb_logger,
            callbacks=[checkpoint_callback, lr_logger],
            max_epochs=max_epoch,
            gpus=gpu_list,
        )

        device = torch.device("cuda")

    config_class = models[model_name]
    config = config_class(args, device)

    tokenizer = config.get_tokenizer()
    model = config.get_model()
    data_module = config.get_data_module()

    trainer.fit(model, train_dataloaders=data_module.train_dataloader(), val_dataloaders=data_module.valid_dataloader())
