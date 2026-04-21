import os

from contextlib import nullcontext
from glob import glob
import json
import random
from tqdm import tqdm
import argparse

import pandas as pd
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from datasets import Dataset, load_dataset, concatenate_datasets


import torch
import torch.nn as nn
from torch.nn import functional as F
import transformers

from grader_utils.parse_utils import parse_answer
from constants import *

# ベースモデル p から p^{alpha} をサンプルする power sampling の実装。
# 引数として 1/alpha（温度 τ）を受け取る（デフォルト 0.25 → alpha=4）。
# mcmc_power_samp が p^{alpha} からの MCMC サンプラー本体。


class AutoregressiveSampler:
    """HuggingFace モデルをラップし、トークン単位の log 確率計算を提供する。"""

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        # モデルが受け付ける最大コンテキスト長
        self.block_size = self.model.config.max_position_embeddings

    @torch.no_grad()
    def next_token(self, prefix):
        """prefix を与えたときの次トークンの log 確率分布を返す。"""
        device = self.device
        torch_prefix = torch.tensor([prefix], dtype=torch.long, device=device)
        # コンテキスト長超過時は末尾 block_size トークンのみ使用
        prefix_cond = torch_prefix if torch_prefix.size(1) <= self.block_size else torch_prefix[:, -self.block_size:]
        output = self.model(prefix_cond)
        logits = output.logits[0, -1, :]
        probs = F.softmax(logits, dim=-1)
        return torch.log(probs)


# --- ユーティリティ関数 ---

def normalize(dist):
    """ロジットを softmax で確率分布に正規化する。"""
    return F.softmax(dist, dim=-1)

def dist_product(logit_p, logit_q):
    """2 つの分布のロジットを加算する（確率の積 p*q に対応）。"""
    return logit_p + logit_q

def dist_temp_scale(logit_p, temp):
    """ロジットを温度 τ でスケーリングする（p^{1/τ} に対応）。"""
    return logit_p * torch.tensor(1 / temp, dtype=logit_p.dtype, device=logit_p.device)


def naive_temp(p: AutoregressiveSampler, context, temp, seq_len):
    """
    低温サンプリングによる提案分布からの系列生成。

    温度 τ でサンプリングし、各トークンについて以下の 2 種類の log 確率を返す：
      - log_probs_norm   : log q(x_t | x_{<t})
                           温度スケール後に正規化した提案分布の log 確率
                           = log softmax(logits / τ)[x_t]
      - log_probs_unnorm : (1/τ) * log p(x_t | x_{<t})
                           ベースモデルの log 確率を 1/τ 倍したもの（未正規化）
                           = (1/τ) * log softmax(logits)[x_t]

    これら 2 つは MCMC 受容率の計算で使用される。
    output_logits=True でスケール前のロジットも取得することで、1 回の forward pass で両方得られる。
    """
    c = len(context)
    device = p.device
    tokenizer = p.tokenizer
    input_ids = torch.tensor([context], dtype=torch.long, device=device)
    output = p.model.generate(
        input_ids=input_ids,
        max_new_tokens=seq_len - c,
        do_sample=True,
        temperature=temp,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,   # 温度スケール後のロジット（提案分布用）
        output_logits=True,   # 温度スケール前のロジット（ベースモデル評価用）
    )
    unscaled_logits = torch.stack(output.logits, dim=0)  # shape: (生成長, 1, vocab)
    scaled_logits   = torch.stack(output.scores, dim=0)  # shape: (生成長, 1, vocab)
    tokens = output.sequences[0][c:]   # 生成されたトークン列
    prop   = output.sequences[0].tolist()  # context + 生成トークンの全列

    assert len(tokens) == unscaled_logits.shape[0] == scaled_logits.shape[0]

    # 各ステップで実際に選ばれたトークンの log 確率を抽出
    idx = tokens.view(unscaled_logits.shape[0], 1, 1)

    # (1/τ) * log p_base(x_t | x_{<t}) : ベースモデルの log 確率
    log_probs_unnorm = (1/temp * torch.gather(F.log_softmax(unscaled_logits, dim=-1), -1, idx)).view(-1).tolist()
    # log q(x_t | x_{<t}) : 温度スケール後・正規化済みの提案分布 log 確率
    log_probs_norm   = torch.gather(F.log_softmax(scaled_logits, dim=-1), -1, idx).view(-1).tolist()

    assert len(tokens) == len(log_probs_unnorm) == len(log_probs_norm)

    return prop, log_probs_norm, log_probs_unnorm


def max_swap(p: AutoregressiveSampler, context, temp, mcmc_steps, max_new_tokens, block_num=16):
    """
    alpha → ∞ 極限の power sampling（greedy swap）。

    mcmc_power_samp との違い：受容率を確率的に判定するのではなく、
    提案系列のスコアが現在系列より高い場合（log_r > 0）に必ず受容する。
    これは目標分布の最頻値（MAP）に向かって確定的に進む手法。
    """
    c = len(context)
    print(f'Temp: {temp}')
    gen = context.copy() if context is not None else []
    log_probs_norm = []
    log_probs_unnorm = []

    print(max_new_tokens)
    assert max_new_tokens % block_num == 0
    # 系列全体を block_num 個のブロックに分割して逐次生成
    jump_size = int(max_new_tokens // block_num)
    print(jump_size)
    attempts = 0
    acceptances = 0

    for _ in tqdm(range(block_num)):
        # 現在の系列末尾に次のブロックを生成・追加
        gen, lp_norm, lp_unnorm = naive_temp(p, gen, temp=temp, seq_len=jump_size+len(gen))
        log_probs_norm.extend(lp_norm)
        log_probs_unnorm.extend(lp_unnorm)

        for _ in tqdm(range(mcmc_steps)):
            attempts += 1
            t = len(gen)
            # ランダムな位置 idx を選び、そこ以降を再生成（部分系列の swap）
            idx = random.randint(c, t-1)
            prop, log_prob_prop, target_log_prob_prop = naive_temp(p, gen[:idx], temp=temp, seq_len=t)
            s = len(prop)
            assert(len(log_prob_prop) == s - idx)
            assert(len(target_log_prob_prop) == s - idx)
            log_prob_cur = log_probs_norm.copy()[idx-c:s-c]
            target_log_prob_cur = log_probs_unnorm.copy()[idx-c:s-c]

            # greedy 受容: ベースモデルスコアが改善する場合のみ受容（確率判定なし）
            log_r = sum(target_log_prob_prop) - sum(target_log_prob_cur)
            if log_r > 0:
                acceptances += 1
                gen = prop.copy()
                log_probs_norm[idx-c:] = log_prob_prop.copy()
                log_probs_unnorm[idx-c:] = target_log_prob_prop.copy()

                del prop
                del log_prob_prop
                del target_log_prob_cur

        # EOS が含まれていれば打ち切り
        if p.tokenizer.eos_token_id in gen:
            eos_idx = gen.index(p.tokenizer.eos_token_id)
            gen = gen[:eos_idx + 1]
            log_probs_norm = log_probs_norm[:eos_idx + 1]
            log_probs_unnorm = log_probs_unnorm[:eos_idx + 1]
            acceptance_ratio = acceptances / attempts
            return gen, log_probs_norm, log_probs_unnorm, acceptance_ratio

    acceptance_ratio = acceptances / attempts
    return gen, log_probs_norm, log_probs_unnorm, acceptance_ratio


def mcmc_power_samp(p: AutoregressiveSampler, context, temp, mcmc_steps, max_new_tokens, block_num=16):
    """
    MCMC による power sampling。目標分布 π(x) ∝ p(x)^{alpha}（alpha=1/τ）からサンプルする。

    アルゴリズム概要:
      1. 系列全体を block_num 個のブロックに分割し、ブロック単位で逐次生成する。
      2. 各ブロック生成後に mcmc_steps 回の Metropolis-Hastings swap を試みる。
         - ランダムな位置 idx を選び、gen[:idx] を prefix として再生成（提案系列 prop を得る）
         - 受容率 a = (π(prop) * q(cur)) / (π(cur) * q(prop)) を計算
         - Uniform(0,1) < a ならば prop を受容、さもなくば現在系列を保持

    受容率の対数:
      log_r = Σ log π(prop[i]) - Σ log π(cur[i])
            + Σ log q(cur[i])  - Σ log q(prop[i])

    ここで:
      log π(x_t) = alpha * log p(x_t) = (1/τ) * log p(x_t)  → log_probs_unnorm
      log q(x_t) = log p_τ(x_t)（温度スケール後・正規化済み） → log_probs_norm

    max_swap との違い: log_r > 0 のときのみ受容（greedy）ではなく、
    確率的に受容することで p^{alpha} の真の定常分布に収束する。
    """
    c = len(context)
    print(f'alpha: {1/temp}')
    gen = context.copy() if context is not None else []
    log_probs_norm = []
    log_probs_unnorm = []

    print(max_new_tokens)
    assert max_new_tokens % block_num == 0
    jump_size = int(max_new_tokens // block_num)
    print(jump_size)
    attempts = 0
    acceptances = 0

    for _ in tqdm(range(block_num)):
        # 次のブロックを低温提案分布でサンプルして系列に追加
        gen, lp_norm, lp_unnorm = naive_temp(p, gen, temp=temp, seq_len=jump_size+len(gen))
        log_probs_norm.extend(lp_norm)
        log_probs_unnorm.extend(lp_unnorm)

        for _ in tqdm(range(mcmc_steps)):
            attempts += 1
            t = len(gen)
            # swap 起点をランダムに選択（prefix は固定）
            idx = random.randint(c, t-1)
            # gen[:idx] を prefix として提案系列を生成
            prop, log_prob_prop, target_log_prob_prop = naive_temp(p, gen[:idx], temp=temp, seq_len=t)
            s = len(prop)
            assert(len(log_prob_prop) == s - idx)
            assert(len(target_log_prob_prop) == s - idx)

            # idx 以降のトークン列について現在系列と提案系列の log 確率を比較
            log_prob_cur = log_probs_norm.copy()[idx-c:s-c]
            target_log_prob_cur = log_probs_unnorm.copy()[idx-c:s-c]

            # Metropolis-Hastings 受容率（対数）
            # log_r = log(π(prop)/π(cur)) + log(q(cur)/q(prop))
            log_r = sum(target_log_prob_prop) + sum(log_prob_cur) - sum(target_log_prob_cur) - sum(log_prob_prop)

            # 確率的受容判定
            if np.random.rand() < np.exp(log_r):
                acceptances += 1
                gen = prop.copy()
                log_probs_norm[idx-c:] = log_prob_prop.copy()
                log_probs_unnorm[idx-c:] = target_log_prob_prop.copy()

                del prop
                del log_prob_prop
                del target_log_prob_cur

        # EOS が出現したら系列を打ち切って早期終了
        if p.tokenizer.eos_token_id in gen:
            eos_idx = gen.index(p.tokenizer.eos_token_id)
            gen = gen[:eos_idx + 1]
            log_probs_norm = log_probs_norm[:eos_idx + 1]
            log_probs_unnorm = log_probs_unnorm[:eos_idx + 1]
            acceptance_ratio = acceptances / attempts
            return gen, log_probs_norm, log_probs_unnorm, acceptance_ratio

    acceptance_ratio = acceptances / attempts
    return gen, log_probs_norm, log_probs_unnorm, acceptance_ratio


def format_prompt(question, model, tokenizer, cot=True):
    """モデルごとにプロンプトフォーマットを適用する。

    base モデル（qwen / qwen_math）はプレーンテキスト形式。
    instruct モデル（phi / tulu / *_grpo）は chat template を適用。
    """
    if model in ("qwen", "qwen_math"):
        format_str = PROMPT + question
        format_str += COT if cot else BASE

    elif model in ("qwen_math_grpo", "phi_grpo", "phi", "tulu"):
        content_str = PROMPT + question
        content_str += COT if cot else BASE
        answer_context = [{"role": "user", "content": content_str}]
        format_str = tokenizer.apply_chat_template(answer_context, tokenize=False, add_generation_prompt=True)

    return format_str
