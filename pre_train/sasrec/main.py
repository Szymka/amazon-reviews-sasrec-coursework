import argparse
import json
import os
import random
import time

import numpy as np
import torch

from model import SASRec
from data_preprocess import *
from utils import *

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--inference_only', default=False, action='store_true')
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument(
    '--skip_preprocess',
    action='store_true',
    help='跳过 Amazon json.gz 解析；使用仓库根目录下 data/amazon/<dataset>.txt（由 scripts/prepare_allmrec_amazon.py 生成）。',
)
parser.add_argument('--n_workers', default=3, type=int, help='WarpSampler 后台进程数；Windows 上可改为 1。')
parser.add_argument(
    '--eval_num_negatives',
    default=100,
    type=int,
    help='验证/测试评估时每个用户的随机负样本数；候选池大小为该值+1，用于 NDCG@10 / HR@10。',
)
parser.add_argument('--eval_seed', default=42, type=int, help='评估阶段随机负采样与用户子采样的随机种子（可复现）。')
parser.add_argument(
    '--eval_every',
    default=20,
    type=int,
    help='每隔多少个 epoch 做一次验证集+测试集评估（第 1 个 epoch 也会评估）。',
)
parser.add_argument(
    '--metrics_jsonl',
    default=None,
    type=str,
    help='若指定，将每次评估的指标以一行 JSON 追加写入该路径（便于三类别批跑与画图）。',
)

args = parser.parse_args()

if __name__ == '__main__':
    
    # global dataset
    if not args.skip_preprocess:
        preprocess(args.dataset)
    dataset = data_partition(args.dataset)

    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    print('user num:', usernum, 'item num:', itemnum)
    num_batch = len(user_train) // args.batch_size
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))
    
    # dataloader
    sampler = WarpSampler(
        user_train,
        usernum,
        itemnum,
        batch_size=args.batch_size,
        maxlen=args.maxlen,
        n_workers=max(1, args.n_workers),
    )
    # model init
    model = SASRec(usernum, itemnum, args).to(args.device)
    
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass
    
    model.train()
    
    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            kwargs, checkpoint = torch.load(args.state_dict_path, map_location=torch.device(args.device))
            kwargs['args'].device = args.device
            model = SASRec(**kwargs).to(args.device)
            model.load_state_dict(checkpoint)
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            print('pdb enabled for your quick check, pls type exit() if you do not need it')
            import pdb; pdb.set_trace()
    
    if args.inference_only:
        model.eval()
        random.seed(args.eval_seed)
        np.random.seed(args.eval_seed)
        t_test = evaluate(model, dataset, args)
        t_valid = evaluate_valid(model, dataset, args)
        print('valid (NDCG@10: %.4f, HR@10: %.4f)' % (t_valid[0], t_valid[1]))
        print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))
    
    bce_criterion = torch.nn.BCEWithLogitsLoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    
    T = 0.0
    t0 = time.time()
    
    for epoch in tqdm(range(epoch_start_idx, args.num_epochs + 1)):
        if args.inference_only: break
        for step in range(num_batch):
            u, seq, pos, neg = sampler.next_batch()
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            pos_logits, neg_logits = model(u, seq, pos, neg)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)

            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            loss.backward()
            adam_optimizer.step()
            if step % 100 == 0:
                print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item())) # expected 0.4~0.6 after init few epochs
    
        if epoch % args.eval_every == 0 or epoch == 1:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            random.seed(args.eval_seed)
            np.random.seed(args.eval_seed)
            t_test = evaluate(model, dataset, args)
            t_valid = evaluate_valid(model, dataset, args)
            print('\n')
            print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)'
                    % (epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))

            print(str(t_valid) + ' ' + str(t_test) + '\n')
            if args.metrics_jsonl:
                out_path = os.path.abspath(args.metrics_jsonl)
                out_dir = os.path.dirname(out_path)
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                row = {
                    "epoch": epoch,
                    "dataset": args.dataset,
                    "valid_ndcg10": float(t_valid[0]),
                    "valid_hr10": float(t_valid[1]),
                    "test_ndcg10": float(t_test[0]),
                    "test_hr10": float(t_test[1]),
                }
                with open(out_path, "a", encoding="utf-8") as mf:
                    mf.write(json.dumps(row, ensure_ascii=False) + "\n")
            t0 = time.time()
            model.train()
    
        if epoch == args.num_epochs:
            # 权重始终写入 pre_train/sasrec/<dataset>/，与 A-LLMRec RecSys 加载路径一致，且不依赖 cwd
            folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.dataset)
            fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
            fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
            if not os.path.exists(os.path.join(folder, fname)):
                try:
                    os.makedirs(folder)
                except:
                    print()
            torch.save([model.kwargs, model.state_dict()], os.path.join(folder, fname))
    
    sampler.close()
    print("Done")