import os
import json
import random
from tqdm import tqdm
import argparse

import pandas as pd
import numpy as np
import torch
import transformers
from torch.nn import functional as F

from grader_utils.parse_utils import parse_answer
from constants import *
from power_samp_utils import (
    AutoregressiveSampler,
    entropy_guided_block_mcmc,
    format_prompt,
)


if __name__ == "__main__":
    # --- 引数 ---
    # --temperature : サンプリング温度 τ（目標分布 p^{1/τ}、提案分布も同じ温度）
    # --block_size  : 1ブロックのトークン数（max_new_tokens を割り切れる値）
    # --tau_avg     : デュアル・トリガーの平均エントロピー閾値
    # --tau_max     : デュアル・トリガーの最大エントロピー閾値
    # --beta        : ロールバック位置提案のエントロピー強調係数
    # --mcmc_steps  : 各ブロックの MH ステップ数（トリガー発動時のみ実行）
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_str", type=str, default="results/")
    parser.add_argument(
        "--model",
        type=str,
        default="qwen_math",
        choices=["qwen", "qwen_math", "phi", "tulu", "qwen_math_grpo", "phi_grpo"],
    )
    parser.add_argument("--dataset", type=str, default="MATH")
    parser.add_argument("--cot", type=bool, default=True)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch_idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=3072)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--tau_avg", type=float, default=1.5)
    parser.add_argument("--tau_max", type=float, default=3.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--mcmc_steps", type=int, default=10)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = args.model
    device = args.device
    cot = args.cot
    save_str = os.path.join(args.save_str, model)
    os.makedirs(save_str, exist_ok=True)

    # モデル文字列の解決
    model_map = {
        "qwen": "Qwen/Qwen2.5-7B",
        "qwen_math": "Qwen/Qwen2.5-Math-7B",
        "qwen_math_grpo": "stellalisy/rethink_rlvr_reproduce-ground_truth-qwen2.5_math_7b-lr5e-7-kl0.00-step150",
        "phi": "microsoft/Phi-3.5-mini-instruct",
        "tulu": "allenai/Llama-3.1-Tulu-3-8B-DPO",
    }
    model_str = model_map[model]

    # データセット読み込み
    if args.dataset == "MATH":
        dataset = json.load(open("data/MATH500.json", "r"))

    print(f"model={model}  device={device}  block_size={args.block_size}")
    print(f"tau_avg={args.tau_avg}  tau_max={args.tau_max}  beta={args.beta}")
    print(f"temp={args.temperature} (alpha={1/args.temperature:.2f})  mcmc_steps={args.mcmc_steps}")

    # モデルロード
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_str, trust_remote_code=True
    )
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_str,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="sdpa",
        trust_remote_code=True,
    ).to(device)
    autoreg_sampler = AutoregressiveSampler(hf_model, tokenizer, device)

    print("model loaded")
    results = []

    start = 125 * args.batch_idx
    end = 125 * (args.batch_idx + 1)

    for problem, data in tqdm(enumerate(dataset[start:end]), desc="MATH entropy MCMC"):
        question = data["prompt"]
        answer = data["answer"]

        input_text = format_prompt(question, model, tokenizer, cot)
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
        prefix = [idx.item() for idx in input_ids[0]]

        # ベースライン: 標準サンプリング (temperature=1.0)
        std_output = hf_model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=True,
        )

        # 提案手法: Entropy-Guided Block MCMC
        eg_output, acceptance_ratio = entropy_guided_block_mcmc(
            autoreg_sampler,
            prefix,
            mcmc_steps=args.mcmc_steps,
            max_new_tokens=args.max_new_tokens,
            block_size=args.block_size,
            tau_avg=args.tau_avg,
            tau_max=args.tau_max,
            beta=args.beta,
            temp=args.temperature,
        )

        # デコード
        std_ids = std_output[0][:, len(input_ids[0]) :].squeeze().cpu()
        eg_ids = torch.tensor(eg_output[len(prefix):], dtype=torch.long, device=device).cpu()

        std_completion = tokenizer.decode(std_ids, skip_special_tokens=True)
        eg_completion = tokenizer.decode(eg_ids, skip_special_tokens=True)

        std_answer = parse_answer(std_completion)
        eg_answer = parse_answer(eg_completion)

        print(
            f"[{problem}] std={std_answer}  eg={eg_answer}  correct={answer}  accept={acceptance_ratio:.3f}"
        )

        results.append(
            {
                "question": question,
                "correct_answer": answer,
                "std_completion": std_completion,
                "std_answer": std_answer,
                "eg_completion": eg_completion,
                "eg_answer": eg_answer,
                "acceptance_ratio": acceptance_ratio,
            }
        )

    # 保存: {model}_math_entropy_mcmc_{block_size}_{tau_avg}_{tau_max}_{batch_idx}_{seed}.csv
    fname = (
        f"{model}_math_entropy_mcmc"
        f"_{args.temperature}_{args.block_size}_{args.tau_avg}_{args.tau_max}"
        f"_{args.batch_idx}_{args.seed}.csv"
    )
    pd.DataFrame(results).to_csv(os.path.join(save_str, fname), index=False)
    print(f"Saved to {os.path.join(save_str, fname)}")
