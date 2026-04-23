import argparse
import pandas as pd
from pathlib import Path
from grader_utils.math_grader import grade_answer


def safe_grade(ans, correct_ans):
    try:
        return int(grade_answer(ans, correct_ans))
    except Exception:
        return 0


def eval_math(fname):
    print(fname)
    df = pd.read_csv(fname)
    total = len(df)
    is_entropy = "eg_answer" in df.columns

    std_correct = sum(safe_grade(df["std_answer"][i], df["correct_answer"][i]) for i in range(total))

    if is_entropy:
        method_correct = sum(safe_grade(df["eg_answer"][i], df["correct_answer"][i]) for i in range(total))
        return std_correct, None, method_correct, total
    else:
        naive_correct = sum(safe_grade(df["naive_answer"][i], df["correct_answer"][i]) for i in range(total))
        mcmc_correct = sum(safe_grade(df["mcmc_answer"][i], df["correct_answer"][i]) for i in range(total))
        return std_correct, naive_correct, mcmc_correct, total


def math_results(fnames):
    std_total = naive_total = mcmc_total = eg_total = 0
    total = 0
    is_entropy = None

    for fname in fnames:
        std, naive, method, n = eval_math(fname)
        std_total += std
        total += n
        if naive is None:
            is_entropy = True
            eg_total += method
        else:
            is_entropy = False
            naive_total += naive
            mcmc_total += method

    denom = max(total, 1)
    print(f"Files evaluated: {len(fnames)}")
    print(f"Total questions: {total}")
    print(f"Std accuracy:   {std_total / denom:.3f}")
    if is_entropy:
        print(f"EG-MCMC accuracy: {eg_total / denom:.3f}")
        return {"std_acc": std_total / denom, "eg_acc": eg_total / denom}
    else:
        print(f"Naive accuracy: {naive_total / denom:.3f}")
        print(f"MCMC accuracy:  {mcmc_total / denom:.3f}")
        return {"std_acc": std_total / denom, "naive_acc": naive_total / denom, "mcmc_acc": mcmc_total / denom}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str)
    args = parser.parse_args()

    folder = Path(args.folder)
    fnames = sorted(str(p) for p in folder.glob("*.csv"))
    math_results(fnames)
