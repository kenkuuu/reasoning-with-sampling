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
        prefix_cond = (
            torch_prefix
            if torch_prefix.size(1) <= self.block_size
            else torch_prefix[:, -self.block_size :]
        )
        output = self.model(prefix_cond)
        logits = output.logits[0, -1, :]
        probs = F.softmax(logits, dim=-1)
        return torch.log(probs)


@torch.no_grad()
def get_kv_cache(p: AutoregressiveSampler, tokens):
    """tokens 列の KV cache を計算して返す。swap 提案の prefix 再計算コスト削減に使用。"""
    input_ids = torch.tensor([tokens], dtype=torch.long, device=p.device)
    output = p.model(input_ids, use_cache=True)
    return output.past_key_values


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


def naive_temp(
    p: AutoregressiveSampler,
    context,
    temp,
    seq_len,
    past_key_values=None,
    past_length=0,
    return_entropy=False,
):
    """
    低温サンプリングによる提案分布からの系列生成。

    past_key_values / past_length を渡すと context[:past_length] の KV cache を再利用し、
    context[past_length:] 分だけ prefill するため swap 提案の計算コストが削減される。

    戻り値:
      prop             : 生成された全トークン列（context + 新規生成）
      log_probs_norm   : log q(x_t | x_{<t})  温度スケール後・正規化済み
      log_probs_unnorm : (1/τ) * log p(x_t | x_{<t})  ベースモデルの log 確率
      out_kv           : generate 完了時点の KV cache（次呼び出しでの再利用用）
    """
    c = len(context)
    device = p.device
    tokenizer = p.tokenizer

    # KV cache を使う場合は未処理トークンのみを input_ids として渡す
    # edge case: past_length >= c（入力が空になる）はキャッシュなしで処理
    using_kv = (past_key_values is not None) and (past_length < c)
    if using_kv:
        input_tokens = context[past_length:]
    else:
        input_tokens = context
        past_key_values = None

    input_ids = torch.tensor([input_tokens], dtype=torch.long, device=device)
    output = p.model.generate(
        input_ids=input_ids,
        past_key_values=past_key_values,
        max_new_tokens=seq_len - c,
        do_sample=True,
        temperature=temp,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,  # 温度スケール後のロジット（提案分布用）
        output_logits=True,  # 温度スケール前のロジット（ベースモデル評価用）
    )
    unscaled_logits = torch.stack(output.logits, dim=0)  # shape: (生成長, 1, vocab)
    scaled_logits = torch.stack(output.scores, dim=0)  # shape: (生成長, 1, vocab)

    n_input = len(input_tokens)
    gen_tokens = output.sequences[0][n_input:]  # 新規生成トークン列
    # KV cache 使用時は cached prefix + input_ids + 生成トークン で完全系列を再構築
    if using_kv:
        prop = context[:past_length] + output.sequences[0].tolist()
    else:
        prop = output.sequences[0].tolist()

    assert len(gen_tokens) == unscaled_logits.shape[0] == scaled_logits.shape[0]

    idx = gen_tokens.view(unscaled_logits.shape[0], 1, 1)
    log_probs_unnorm = (
        (1 / temp * torch.gather(F.log_softmax(unscaled_logits, dim=-1), -1, idx))
        .view(-1)
        .tolist()
    )
    log_probs_norm = (
        torch.gather(F.log_softmax(scaled_logits, dim=-1), -1, idx).view(-1).tolist()
    )

    assert len(gen_tokens) == len(log_probs_unnorm) == len(log_probs_norm)

    out_kv = getattr(output, "past_key_values", None)

    if return_entropy:
        # ベースモデルの確率分布からトークンごとのエントロピーを計算
        # unscaled_logits は温度スケール前のロジット: shape (gen_len, 1, vocab)
        probs = F.softmax(unscaled_logits.squeeze(1), dim=-1)  # (gen_len, vocab)
        entropies = (-(probs * torch.log(probs.clamp(min=1e-10))).sum(dim=-1)).tolist()
        return prop, log_probs_norm, log_probs_unnorm, out_kv, entropies

    return prop, log_probs_norm, log_probs_unnorm, out_kv


def max_swap(
    p: AutoregressiveSampler, context, temp, mcmc_steps, max_new_tokens, block_num=16
):
    """
    alpha → ∞ 極限の power sampling（greedy swap）。

    mcmc_power_samp との違い：受容率を確率的に判定するのではなく、
    提案系列のスコアが現在系列より高い場合（log_r > 0）に必ず受容する。
    これは目標分布の最頻値（MAP）に向かって確定的に進む手法。
    """
    c = len(context)
    print(f"Temp: {temp}")
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

    context_kv = get_kv_cache(p, context)

    for _ in tqdm(range(block_num)):
        # 現在の系列末尾に次のブロックを生成・追加
        gen, lp_norm, lp_unnorm, _ = naive_temp(
            p, gen, temp=temp, seq_len=jump_size + len(gen)
        )
        log_probs_norm.extend(lp_norm)
        log_probs_unnorm.extend(lp_unnorm)

        for _ in tqdm(range(mcmc_steps)):
            attempts += 1
            t = len(gen)
            # ランダムな位置 idx を選び、そこ以降を再生成（部分系列の swap）
            idx = random.randint(c, t - 1)
            # context の KV cache を再利用: gen[c:idx] だけ prefill し context 分をスキップ
            prop, log_prob_prop, target_log_prob_prop, _ = naive_temp(
                p,
                gen[:idx],
                temp=temp,
                seq_len=t,
                past_key_values=context_kv,
                past_length=c,
            )
            s = len(prop)
            assert len(log_prob_prop) == s - idx
            assert len(target_log_prob_prop) == s - idx
            log_prob_cur = log_probs_norm.copy()[idx - c : s - c]
            target_log_prob_cur = log_probs_unnorm.copy()[idx - c : s - c]

            # greedy 受容: ベースモデルスコアが改善する場合のみ受容（確率判定なし）
            log_r = sum(target_log_prob_prop) - sum(target_log_prob_cur)
            if log_r > 0:
                acceptances += 1
                gen = prop.copy()
                log_probs_norm[idx - c :] = log_prob_prop.copy()
                log_probs_unnorm[idx - c :] = target_log_prob_prop.copy()

                del prop
                del log_prob_prop
                del target_log_prob_cur

        # EOS が含まれていれば打ち切り
        if p.tokenizer.eos_token_id in gen:
            eos_idx = gen.index(p.tokenizer.eos_token_id)
            gen = gen[: eos_idx + 1]
            log_probs_norm = log_probs_norm[: eos_idx + 1]
            log_probs_unnorm = log_probs_unnorm[: eos_idx + 1]
            acceptance_ratio = acceptances / attempts
            return gen, log_probs_norm, log_probs_unnorm, acceptance_ratio

    acceptance_ratio = acceptances / attempts
    return gen, log_probs_norm, log_probs_unnorm, acceptance_ratio


def mcmc_power_samp(
    p: AutoregressiveSampler, context, temp, mcmc_steps, max_new_tokens, block_num=16
):
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
    print(f"alpha: {1/temp}")
    gen = context.copy() if context is not None else []
    log_probs_norm = []
    log_probs_unnorm = []

    print(max_new_tokens)
    assert max_new_tokens % block_num == 0
    jump_size = int(max_new_tokens // block_num)
    print(jump_size)
    attempts = 0
    acceptances = 0

    # context の KV cache を一度だけ計算し、全 swap 提案で再利用
    # swap では gen[c:idx] だけ prefill すればよい（context の再計算をスキップ）
    context_kv = get_kv_cache(p, context)

    for _ in tqdm(range(block_num)):
        # 次のブロックを低温提案分布でサンプルして系列に追加
        gen, lp_norm, lp_unnorm, _ = naive_temp(
            p, gen, temp=temp, seq_len=jump_size + len(gen)
        )
        log_probs_norm.extend(lp_norm)
        log_probs_unnorm.extend(lp_unnorm)

        for _ in tqdm(range(mcmc_steps)):
            attempts += 1
            t = len(gen)
            # swap 起点をランダムに選択（prefix は固定）
            idx = random.randint(c, t - 1)
            # gen[:idx] を prefix として提案系列を生成（context KV cache を再利用）
            prop, log_prob_prop, target_log_prob_prop, _ = naive_temp(
                p,
                gen[:idx],
                temp=temp,
                seq_len=t,
                past_key_values=context_kv,
                past_length=c,
            )
            s = len(prop)
            assert len(log_prob_prop) == s - idx
            assert len(target_log_prob_prop) == s - idx

            # idx 以降のトークン列について現在系列と提案系列の log 確率を比較
            log_prob_cur = log_probs_norm.copy()[idx - c : s - c]
            target_log_prob_cur = log_probs_unnorm.copy()[idx - c : s - c]

            # Metropolis-Hastings 受容率（対数）
            # log_r = log(π(prop)/π(cur)) + log(q(cur)/q(prop))
            log_r = (
                sum(target_log_prob_prop)
                + sum(log_prob_cur)
                - sum(target_log_prob_cur)
                - sum(log_prob_prop)
            )

            # 確率的受容判定
            if np.random.rand() < np.exp(log_r):
                acceptances += 1
                gen = prop.copy()
                log_probs_norm[idx - c :] = log_prob_prop.copy()
                log_probs_unnorm[idx - c :] = target_log_prob_prop.copy()

                del prop
                del log_prob_prop
                del target_log_prob_cur

        # EOS が出現したら系列を打ち切って早期終了
        if p.tokenizer.eos_token_id in gen:
            eos_idx = gen.index(p.tokenizer.eos_token_id)
            gen = gen[: eos_idx + 1]
            log_probs_norm = log_probs_norm[: eos_idx + 1]
            log_probs_unnorm = log_probs_unnorm[: eos_idx + 1]
            acceptance_ratio = acceptances / attempts
            return gen, log_probs_norm, log_probs_unnorm, acceptance_ratio

    acceptance_ratio = acceptances / attempts
    return gen, log_probs_norm, log_probs_unnorm, acceptance_ratio


def make_annealing_schedule(alpha_start=1.0, alpha_end=4.0, n_steps=10):
    """ステップ n に応じて alpha_start → alpha_end へ線形に増加するスケジュールを返す。"""

    def schedule(n):
        if n_steps <= 1:
            return alpha_end
        return alpha_start + (alpha_end - alpha_start) * (n - 1) / (n_steps - 1)

    return schedule


def entropy_guided_block_mcmc(
    p: AutoregressiveSampler,
    context,
    mcmc_steps,
    max_new_tokens,
    block_size,
    tau_avg,
    tau_max,
    beta,
    temp,
):
    """
    Self-Contained Entropy-Guided Block MCMC。
    目標分布 π(x) ∝ p(x)^{1/temp}、提案分布として温度 temp のベースモデルを使用。

    アルゴリズム:
      ブロックごとに:
        1. ブロックを生成し、各トークンのエントロピー H_t を記録
        2. デュアル・トリガー: H_avg <= tau_avg かつ H_max <= tau_max なら確信ありとしてスキップ
        3. トリガー発動時は mcmc_steps 回の MH ループを実行
           - ロールバック位置 t を q(t) ∝ exp(β*H_t) でサンプル
           - t 以降を再生成し、受容率 A で受容/棄却
           - A = p(new)^α / p(curr)^α × Z(curr)/Z(new) × p_prop(curr)/p_prop(new)

    Returns: (gen, acceptance_ratio)
    """
    c = len(context)
    alpha = 1.0 / temp  # 目標分布のべき乗指数
    gen = context.copy()
    context_kv = get_kv_cache(p, context)

    assert max_new_tokens % block_size == 0
    num_blocks = max_new_tokens // block_size

    total_attempts = 0
    total_acceptances = 0

    for _ in tqdm(range(num_blocks)):
        block_start = len(gen)

        # Step 1: ブロック生成（エントロピーも記録）
        gen, lp_norm, lp_unnorm, _, entropies = naive_temp(
            p,
            gen,
            temp=temp,
            seq_len=block_start + block_size,
            past_key_values=context_kv,
            past_length=c,
            return_entropy=True,
        )
        # lp_base[i] = log p_base(x_t) = temp * lp_unnorm[i]（lp_unnorm = (1/temp)*log p_base）
        block_lp_norm = list(lp_norm)
        block_lp_base = [temp * lu for lu in lp_unnorm]
        block_entropies = list(entropies)
        actual_block_len = len(block_entropies)  # EOS 早期終了時は block_size より短い

        # EOS がブロック生成中に出た場合は即終了
        if p.tokenizer.eos_token_id in gen[block_start:]:
            eos_pos = gen.index(p.tokenizer.eos_token_id, block_start)
            return gen[: eos_pos + 1], total_acceptances / max(total_attempts, 1)

        # Step 2: デュアル・トリガー
        H_avg = sum(block_entropies) / actual_block_len
        H_max = max(block_entropies)

        if H_avg <= tau_avg and H_max <= tau_max:
            if p.tokenizer.eos_token_id in gen[block_start:]:
                eos_pos = gen.index(p.tokenizer.eos_token_id, block_start)
                return gen[: eos_pos + 1], total_acceptances / max(total_attempts, 1)
            continue

        # Step 3: エントロピー誘導型 MH ループ
        for _ in tqdm(range(mcmc_steps)):
            total_attempts += 1

            # ロールバック位置を q(t) ∝ exp(β*H_t) でサンプル
            beta_h = torch.tensor(
                [beta * h for h in block_entropies], dtype=torch.float32
            )
            t_rel = torch.multinomial(torch.softmax(beta_h, dim=0), 1).item()
            t_abs = block_start + t_rel
            log_Z_curr = torch.logsumexp(beta_h, dim=0).item()

            # t_abs 以降を再生成
            n_new = actual_block_len - t_rel
            prop, prop_lp_norm, prop_lp_unnorm, _, prop_entropies = naive_temp(
                p,
                gen[:t_abs],
                temp=temp,
                seq_len=t_abs + n_new,
                past_key_values=context_kv,
                past_length=c,
                return_entropy=True,
            )
            prop_lp_base = [temp * lu for lu in prop_lp_unnorm]

            # 提案系列のブロック全体エントロピー（prefix 部分は不変）
            new_block_entropies = block_entropies[:t_rel] + list(prop_entropies)
            beta_h_new = torch.tensor(
                [beta * h for h in new_block_entropies], dtype=torch.float32
            )
            log_Z_new = torch.logsumexp(beta_h_new, dim=0).item()

            # 受容率（対数）
            # log A = α*(log p(new) - log p(curr))   目標分布の比
            #       + (log Z_curr - log Z_new)         ロールバック提案の正規化補正
            #       + (log p_prop(curr) - log p_prop(new))  提案分布の比
            log_A = (
                alpha * (sum(prop_lp_base) - sum(block_lp_base[t_rel:]))
                + (log_Z_curr - log_Z_new)
                + (sum(block_lp_norm[t_rel:]) - sum(prop_lp_norm))
            )

            if np.random.rand() < np.exp(log_A):
                total_acceptances += 1
                gen = prop
                block_lp_norm = block_lp_norm[:t_rel] + list(prop_lp_norm)
                block_lp_base = block_lp_base[:t_rel] + prop_lp_base
                block_entropies = new_block_entropies

        # EOS チェック
        if p.tokenizer.eos_token_id in gen[block_start:]:
            eos_pos = gen.index(p.tokenizer.eos_token_id, block_start)
            return gen[: eos_pos + 1], total_acceptances / max(total_attempts, 1)

    return gen, total_acceptances / max(total_attempts, 1)


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
        format_str = tokenizer.apply_chat_template(
            answer_context, tokenize=False, add_generation_prompt=True
        )

    return format_str
