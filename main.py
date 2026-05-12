import os
import sys
import argparse

from utils import *
from train_model import *

if __name__ == "__main__":
    # 避免与 SASRec 子目录入口混淆（根目录 main 为 A-LLMRec）
    if "--dataset" in sys.argv or "--skip_preprocess" in sys.argv:
        print(
            "当前是 A-LLMRec 的 main.py（参数为 --pretrain_stage1、--rec_pre_trained_data、--gpu_num 等）。\n"
            "你传入的是 SASRec 训练参数（含 --dataset / --skip_preprocess）。请改用下面其一：\n"
            "  cd pre_train\\sasrec\n"
            "  python main.py --device cuda:0 --dataset Industrial_and_Scientific --skip_preprocess ...\n"
            "或在仓库根目录：\n"
            "  python pre_train/sasrec/main.py --device cuda:0 --dataset Industrial_and_Scientific --skip_preprocess ...\n",
            file=sys.stderr,
        )
        sys.exit(2)

    parser = argparse.ArgumentParser()
    
    # GPU train options
    parser.add_argument("--multi_gpu", action='store_true')
    parser.add_argument('--gpu_num', type=int, default=0)
    
    # model setting
    parser.add_argument("--llm", type=str, default='opt', help='flan_t5, opt, vicuna')
    parser.add_argument("--recsys", type=str, default='sasrec')
    
    # dataset setting
    parser.add_argument("--rec_pre_trained_data", type=str, default='Movies_and_TV')
    
    # train phase setting
    parser.add_argument("--pretrain_stage1", action='store_true')
    parser.add_argument("--pretrain_stage2", action='store_true')
    parser.add_argument("--inference", action='store_true')
    
    # hyperparameters options
    parser.add_argument('--batch_size1', default=32, type=int)
    parser.add_argument('--batch_size2', default=2, type=int)
    parser.add_argument('--batch_size_infer', default=2, type=int)
    parser.add_argument('--maxlen', default=50, type=int)
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument("--stage1_lr", type=float, default=0.0001)
    parser.add_argument("--stage2_lr", type=float, default=0.0001)
    
    args = parser.parse_args()
    
    args.device = 'cuda:' + str(args.gpu_num)
    
    if args.pretrain_stage1:
        train_model_phase1(args)
    elif args.pretrain_stage2:
        train_model_phase2(args)
    elif args.inference:
        inference(args)