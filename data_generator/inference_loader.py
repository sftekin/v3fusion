import re
import os
import sys
import glob
import difflib
import numpy as np
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)

from data_generator.data_loader import DataCreator
from configs import prompt_formats


infer_dir = os.path.join(PROJECT_DIR, "results", "inference")
base_models_dir = os.path.join(PROJECT_DIR, "results", "inference_base_models")


def load_infer_prob_data(model_names, task_name, ds_split):
    prob_data = []
    labels = None
    for mn in model_names:
        data_path = os.path.join(infer_dir, task_name, ds_split, f"{mn}_output.csv")
        data_df = pd.read_csv(data_path)

        arr_path = os.path.join(infer_dir, task_name, ds_split, f"{mn}_prob.npy")
        prob_arr = np.load(arr_path)

        start_chr = 'A'
        choices = []
        for i in range(prob_arr.shape[1]):
            choices.append(start_chr)
            start_chr = chr(ord(start_chr) + 1)

        labels = []
        answers = data_df["answer"].values.astype(str)
        for ans in answers:
            labels.append(choices.index(ans))
        labels = np.array(labels)

        if task_name == "mmmu_pro" and mn != "llava-v1.6-vicuna-13b-hf":
            labels = np.delete(labels, (1017), axis=0)
            prob_arr = np.delete(prob_arr, (1017), axis=0)
        
        prob_data.append(prob_arr)
    
    prob_data = np.concatenate(prob_data, axis=1)
    data = np.concatenate([prob_data, labels[:, None]], axis=1)

    return data


def load_base_model_answers(model_name, task_name, ds_split, ref_model_name):
    """Multiple-choice answers of a model under results/inference_base_models, aligned to the rows returned by
    load_infer_prob_data(..., task_name, ds_split). Returns choice indices, -1 where no answer could be parsed
    (e.g. the generation hit the token limit before answering)."""
    files = sorted(glob.glob(os.path.join(base_models_dir, model_name, f"{task_name}_{ds_split}_results_*_final.csv")))
    if not files:
        raise FileNotFoundError(f"no {task_name} {ds_split} results for {model_name} under {base_models_dir}")
    src_df = pd.read_csv(files[-1])
    ref_df = pd.read_csv(os.path.join(infer_dir, task_name, ds_split, f"{ref_model_name}_output.csv"))

    ref_q = ref_df["question"].astype(str).str.strip().tolist()
    ref_a = ref_df["answer"].astype(str).tolist()
    src_q = [_question_body(p) for p in src_df["question"]]
    src_a = src_df["ground_truth"].astype(str).tolist()
    idx = _align_rows(ref_q, ref_a, src_q, src_a)

    letters = [_parse_choice(out) for out in src_df["model_output"].values[idx]]
    answers = np.array([ord(c) - ord("A") if c else -1 for c in letters])
    n_reworded = sum(ref_q[i] != src_q[j] for i, j in enumerate(idx))
    n_new_key = sum(ref_a[i] != src_a[j] for i, j in enumerate(idx))
    print(f"{model_name}: aligned {len(idx)} rows to {os.path.basename(files[-1])} ({n_reworded} reworded questions, "
          f"{n_new_key} with a different answer key), parsed {np.sum(answers >= 0)} answers")
    return answers


def _question_body(prompt):
    """question text of a prompt in results/inference_base_models, without the options and instructions"""
    body = str(prompt)
    for marker in ["\nPlease answer directly", "\n\nYou are a visual question answering assistant", "\nA. "]:
        body = body.split(marker)[0]
    return body.strip()


_answer_line = re.compile(r"^[\s*#>-]*(?:final\s+)?answer[\s*]*:[\s*]*\(?([A-I])\)?[\s*.]*$", re.IGNORECASE | re.MULTILINE)
_answer_inline = re.compile(r"Answer:\s*\**\s*\(?([A-I])\b(?![ \t]+[a-z])")


def _parse_choice(output):
    """last answer letter of a generated rationale: an 'Answer: X' line first, else e.g. 'Answer: B. 3.47'"""
    found = [c for c in _answer_line.findall(str(output)) if c.isupper()]
    if not found:
        found = _answer_inline.findall(str(output))
    return found[-1] if found else None


def _align_rows(ref_q, ref_a, src_q, src_a, min_sim=0.75, answer_bonus=0.5):
    """Order-preserving alignment of every reference row to a distinct source row; the source has extra rows
    (questions our inference filter skipped). Rows match on question text: exact matches first, fuzzy ones for
    questions reworded in a later dataset release, and agreeing answer keys break ties between repeated texts."""
    n, band = len(ref_q), len(src_q) - len(ref_q)
    if band < 0:
        raise ValueError("source has fewer rows than the reference")
    score = np.full((n, band + 1), -np.inf)
    for i in range(n):
        window = src_q[i:i + band + 1]
        exact = [off for off, text in enumerate(window) if text == ref_q[i]]
        if exact:
            candidates = [(off, 1.0) for off in exact]
        else:
            candidates = [(off, difflib.SequenceMatcher(None, ref_q[i], text).ratio()) for off, text in enumerate(window)]
        for off, sim in candidates:
            if sim >= min_sim:
                score[i, off] = sim + answer_bonus * (ref_a[i] == src_a[i + off])

    # row i -> source row i + off; the matches stay strictly increasing iff off never decreases
    best, back = score.copy(), np.zeros(score.shape, dtype=int)
    for i in range(1, n):
        arg = 0
        for off in range(band + 1):
            if best[i - 1, off] > best[i - 1, arg]:
                arg = off
            best[i, off] += best[i - 1, arg]
            back[i, off] = arg
    if not np.isfinite(best[-1].max()):
        raise ValueError("could not align the rows by question text")

    offsets = np.zeros(n, dtype=int)
    offsets[-1] = int(np.argmax(best[-1]))
    for i in range(n - 1, 0, -1):
        offsets[i - 1] = back[i, offsets[i]]
    return np.arange(n) + offsets



def load_infer_open_data(model_names, task_name, ds_split):
    model_outputs = []
    answers = []
    questions = []
    for mn in model_names:
        data_path = os.path.join(infer_dir, task_name, ds_split, f"{mn}_output.csv")
        data_df = pd.read_csv(data_path, index_col=0)
        model_outputs.append(data_df["generated_outputs"].values)
        if len(answers) == 0:
            answers = data_df["answer"].tolist()
            questions = data_df["question"].tolist()
    model_outputs = np.array(model_outputs)

    return model_outputs, questions, answers


def load_infer_mc_data(model_names, task_name, ds_split):
    ds_creator = DataCreator(task_name)
    ds_questions= []
    for ds in ds_creator.get(ds_split):
        for example in ds:
            start_chr = 'A'
            index2ans = {}
            option_txt = ""
            prediction_range = []
            if "mmmu" in task_name:
                options = eval(example["options"])
            else:
                options = example["options"]
            for option in options:
                prediction_range.append(start_chr)
                option_txt += f"({start_chr}) {option}\n"
                index2ans[start_chr] = option
                start_chr = chr(ord(start_chr) + 1)
            empty_prompt_sample_structure = prompt_formats['multi_choice_example_format']
            empty_prompt = empty_prompt_sample_structure.format(example["question"], option_txt)
            ds_questions.append(empty_prompt)

    def extract_letter(text):
        match = re.search(r"\((\w)\)", text)
        return match.group(1) if match else ""

    model_outputs, answers = [], []
    for mn in model_names:
        data_path = os.path.join(infer_dir, task_name, ds_split, f"{mn}_output.csv")
        data_df = pd.read_csv(data_path)

        arr_path = os.path.join(infer_dir, task_name, ds_split, f"{mn}_prob.npy")
        prob_arr = np.load(arr_path)

        start_chr = 'A'
        choices = []
        for i in range(prob_arr.shape[1]):
            choices.append(start_chr)
            start_chr = chr(ord(start_chr) + 1)

        generated_outputs = data_df["generated_outputs"].values
        if len(answers) == 0:
            answers = data_df["answer"].tolist()

        extracted_outputs = []
        for output in generated_outputs:
            pred_txt = str(output)[:10].strip()
            if "\n" in pred_txt:
                pred_txt = pred_txt.split("\n")[1]
            if "(" in pred_txt or ")" in pred_txt:
                pred_txt = extract_letter(pred_txt)
            extracted_outputs.append(pred_txt[:1].upper())
        extracted_outputs = np.array(extracted_outputs)

        labels = data_df["answer"].values.astype(str)
        if task_name == "mmmu_pro" and mn != "llava-v1.6-vicuna-13b-hf":
            extracted_outputs = np.delete(extracted_outputs, (1017), axis=0)
            labels = np.delete(labels, (1017), axis=0)
            ds_questions = np.delete(ds_questions, (1017), axis=0)
        model_outputs.append(extracted_outputs)
    model_outputs = np.array(model_outputs)
    ds_questions = ds_questions[:len(model_outputs[0])]
    labels = labels.tolist()
    return model_outputs, ds_questions, labels

