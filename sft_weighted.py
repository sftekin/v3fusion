import os
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import entropy
from sklearn.metrics import roc_auc_score

from configs import RESULT_DIR
from data_generator.inference_loader import load_infer_prob_data, load_base_model_answers
from ens_pruning.uncertainty_rectify import adaptive_entropy_threshold


model_mapper_dict = {
    0: "llava-v1.6-vicuna-7b-hf",
    1: "llava-v1.6-vicuna-13b-hf",
    2: "Qwen2.5-VL-7B-Instruct",
    3: "InternVL2-8B",
    4: "deepseek-vl2-tiny",
    5: "deepseek-vl2-small"
}


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.ReLU(),
            nn.Linear(hidden_dim[1], output_dim),
        )
        self.net.apply(self.init_weights)

    def forward(self, x):
        out = self.net(x)
        out = torch.softmax(out, dim=-1)
        return out

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)


def train_ensemble(model_names, train_loader, val_loader, novel_loader, n_epochs, save_dir, space_size, verbose=True,
                   rectify=False, rectify_alpha=10.0, rectify_policy="mean", router_answers=None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model = MLP(len(model_names) * space_size, [100, 100], space_size)
    model = model.to("cuda")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_val_acc, tol = (0, 0)
    for epoch in range(n_epochs):
        avg_loss = []
        for i, batch_data in enumerate(train_loader):
            in_x = batch_data[:, :-1].to("cuda").float()
            label = batch_data[:, -1].type(torch.long).to("cuda")

            optimizer.zero_grad()
            out = model(in_x)
            loss = loss_fn(out, label)

            loss.backward()
            optimizer.step()
            avg_loss.append(loss.item())

        if epoch % 10 == 0 and verbose:
            run_loss = np.mean(avg_loss)
            print(f'Epoch {epoch} | Loss {run_loss:.4f}')

        if val_loader:
            acc_mean = test_loop(model, val_loader)

            if acc_mean > best_val_acc:
                outfile = os.path.join(save_dir, f'best_model.tar')
                torch.save({'epoch': epoch,
                            'state': model.state_dict(),
                            "accuracy": acc_mean}, outfile)
                best_val_acc = acc_mean
                tol = 0
            else:
                tol += 1

            if tol > 300:
                print("No improvement in 200 epochs, breaking")
                break

    if val_loader:
        best_dict = torch.load(f"{save_dir}/best_model.tar", weights_only=False)
        model.load_state_dict(best_dict["state"])

    model.eval()
    acc_mean, logits, labels = test_loop(model, novel_loader, ret_logit=True)
    print(f'Novel Acc = {acc_mean:.4f}')
    exp_result = dict(val_acc=best_dict["accuracy"],
                      test_acc=acc_mean,
                      state=model.state_dict(),
                      model_names=model_names,
                      logits=logits,
                      labels=labels)

    if rectify:
        exp_result["rectify"] = rectify_predictions(logits, labels, len(model_names), space_size,
                                                  alpha=rectify_alpha, policy=rectify_policy,
                                                  router_answers=router_answers)

    output_path = os.path.join(save_dir, "exp_result.pth")
    torch.save(exp_result, output_path)

    return exp_result


def rectify_predictions(logits, labels, model_count, space_size, alpha=10.0, policy="mean", router_answers=None):
    """Uncertainty rectification stage from notebooks/entropy_open_ended.ipynb: epistemic uncertainty is the
    entropy of the ensemble output minus the mean base-model entropy; tau is picked by the adaptive
    Gaussian-vs-GMM threshold, and samples above tau fall back to either the average of the base models'
    probabilities (policy="mean"), their majority vote (policy="vote", ties broken by the average), the
    ensemble's own second choice (policy="second"), or the answers of a larger model (policy="router",
    router_answers holds its choice index per sample, -1 where it gave none)."""
    base_probs = logits[:, :model_count * space_size]
    ens_probs = logits[:, model_count * space_size:]

    per_model_entropy = np.mean([entropy(p, base=2, axis=1)
                                 for p in np.split(base_probs, model_count, axis=1)], axis=0)
    per_model_entropy[np.isnan(per_model_entropy)] = 0
    ens_entropy = entropy(ens_probs, base=2, axis=1)
    ens_entropy[np.isnan(ens_entropy)] = 0
    epistemic_uncertainty = ens_entropy - per_model_entropy

    tau, selected, llr, _ = adaptive_entropy_threshold(epistemic_uncertainty, alpha=alpha)
    reject_idx = epistemic_uncertainty > tau

    model_probs = np.stack(np.split(base_probs, model_count, axis=1))  # (model_count, n, space_size)
    fallback_probs = model_probs.mean(axis=0)
    if policy == "vote":
        votes = np.eye(space_size)[model_probs.argmax(axis=2)].sum(axis=0)
        # the average probabilities are < 1, so they only break ties between equal vote counts
        fallback_probs = votes + fallback_probs
    elif policy == "second":
        # zero out the ensemble's top pick so the argmax falls through to its runner-up
        fallback_probs = ens_probs.copy()
        fallback_probs[np.arange(len(ens_probs)), ens_probs.argmax(axis=1)] = 0
    elif policy == "router":
        # one-hot router answers; samples it left unanswered keep the ensemble's prediction
        answered = (router_answers >= 0) & (router_answers < space_size)
        fallback_probs = ens_probs.copy()
        fallback_probs[answered] = np.eye(space_size)[router_answers[answered]]
    elif policy != "mean":
        raise ValueError(f"unknown rectify policy: {policy}")

    rectified_probs = ens_probs.copy()
    rectified_probs[reject_idx] = fallback_probs[reject_idx]

    before_acc = np.mean(ens_probs.argmax(1) == labels) * 100
    after_acc = np.mean(rectified_probs.argmax(1) == labels) * 100
    print(f'Rectification ({policy} fallback): {selected} selected (LLR={llr:.2f}), tau={tau:.4f}, '
          f'rejected {reject_idx.mean() * 100:.2f}% of samples')
    errors = ens_probs.argmax(1) != labels
    auroc, auroc_ci = detection_auroc(epistemic_uncertainty, errors)
    rejection = rejection_metrics(reject_idx, errors)
    print(f'Detection AUROC = {auroc:.4f} (95% CI {auroc_ci[0]:.4f}-{auroc_ci[1]:.4f})')
    print(f'Rejection balanced accuracy = {rejection["balanced_accuracy"]:.4f} | precision {rejection["precision"] * 100:.2f}% '
          f'(errors among rejected, vs {errors.mean() * 100:.2f}% overall) | recall {rejection["recall"] * 100:.2f}% '
          f'(errors rejected) | {rejection["false_rejection_rate"] * 100:.2f}% of correct answers rejected')
    print(f'Novel Acc before rectification = {before_acc:.4f}')
    print(f'Novel Acc after rectification  = {after_acc:.4f}')
    router_acc = None
    if policy == "router":
        router_acc = np.mean(router_answers == labels) * 100
        rejected_acc = np.mean(router_answers[reject_idx] == labels[reject_idx]) * 100 if reject_idx.any() else float("nan")
        print(f'Router alone: acc {router_acc:.4f} on all samples, {rejected_acc:.4f} on the rejected ones; '
              f'{np.sum(reject_idx & ~answered)} rejected samples had no router answer and kept the ensemble\'s')

    return dict(tau=tau,
                selected_dist=selected,
                log_lik_ratio=llr,
                policy=policy,
                router_acc=router_acc,
                detection_auroc=auroc,
                detection_auroc_ci=auroc_ci,
                rejection=rejection,
                reject_idx=reject_idx,
                epistemic_uncertainty=epistemic_uncertainty,
                before_acc=before_acc,
                after_acc=after_acc,
                rectified_probs=rectified_probs)


def rejection_metrics(reject_idx, errors):
    """The rejection decision scored as a detector of the ensemble's errors: precision (errors among the rejected),
    recall (errors that got rejected), false rejection rate (correct answers that got rejected) and balanced
    accuracy, the mean of recall and 1 - false rejection rate (0.5 = no better than rejecting at random)."""
    caught = np.sum(reject_idx & errors)
    precision = caught / reject_idx.sum() if reject_idx.any() else float("nan")
    recall = caught / errors.sum() if errors.any() else float("nan")
    false_rejection = np.sum(reject_idx & ~errors) / np.sum(~errors) if (~errors).any() else float("nan")
    return dict(precision=precision, recall=recall, false_rejection_rate=false_rejection,
                balanced_accuracy=(recall + 1 - false_rejection) / 2)


def detection_auroc(score, errors, n_boot=1000, seed=0):
    """AUROC of an uncertainty score for flagging the ensemble's errors (0.5 = no better than picking samples at
    random), with a percentile bootstrap 95% CI over samples."""
    if errors.all() or not errors.any():
        return float("nan"), (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(score), len(score))
        if errors[idx].any() and not errors[idx].all():
            boots.append(roc_auc_score(errors[idx], score[idx]))
    return roc_auc_score(errors, score), tuple(np.percentile(boots, [2.5, 97.5]))


def test_loop(model, data_loader, ret_logit=False, device="cuda"):
    assert device in ["cuda", "cpu"]
    acc_all = []
    logits = []
    labels = []
    for i, batch_data in enumerate(data_loader):
        in_x = batch_data[:, :-1].to(device).float()
        scores = model(in_x)
        label = batch_data[:, -1].numpy()

        scores = scores.detach().cpu().numpy()
        in_x = in_x.detach().cpu().numpy()
        pred = np.argmax(scores, axis=1)
        corrects = np.sum(pred == label)
        acc_all.append(corrects / len(label) * 100)
        if ret_logit:
            logits.append(np.concatenate([in_x, scores], axis=1))
            labels.append(label)

    acc_all = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)

    if ret_logit:
        logits = np.concatenate(logits)
        labels = np.concatenate(labels)
        return acc_mean, logits, labels
    else:
        return acc_mean


def run(args):
    np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)
    model_names = [model_mapper_dict[int(i)] for i in args.model_ids]

    if args.task_name == "mmmu":
        data = load_infer_prob_data(model_names, "mmmu_pro", "test")    
        novel_split = "validation"
    elif args.task_name == "mmmu_pro":
        data = load_infer_prob_data(model_names, "mmmu", "validation") 
        novel_split = "test"
    else:
        # okvqa
        data = load_infer_prob_data(model_names, args.task_name, args.dataset_type)
        novel_split = "validation"
    test_data = load_infer_prob_data(model_names, args.task_name, novel_split)

    router_answers = None
    if args.rectify and args.rectify_policy == "router":
        router_answers = load_base_model_answers(args.router_model, args.task_name, novel_split, model_names[0])
        if len(router_answers) != len(test_data):
            raise ValueError(f"{args.router_model} answers {len(router_answers)} samples, novel set has {len(test_data)}")
    space_size = (data.shape[1] - 1) // len(model_names)
    print(f"Space Size: {space_size}")


    rand_idx = np.random.permutation(len(data))
    data = data[rand_idx]
    ds_len = len(data)
    train_size = int(ds_len * 0.75)
    print(f"Train Size: {train_size}")
    val_size = int(ds_len * 0.3)
    split = {"train": data[:train_size],
             "val": data[train_size:train_size + val_size],
             "test": test_data}

    train_loader = DataLoader(split["train"], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(split["val"], batch_size=args.batch_size, shuffle=True)
    novel_loader = DataLoader(split["test"], batch_size=args.batch_size, shuffle=False)

    train_ensemble(model_names, train_loader, val_loader, novel_loader,
                   n_epochs=args.epochs, save_dir=f"results/{args.out_root}/{args.task_name}/{args.model_ids}",
                   space_size=space_size, verbose=True,
                   rectify=args.rectify, rectify_alpha=args.rectify_alpha,
                   rectify_policy=args.rectify_policy, router_answers=router_answers)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inference scripts for the trained models')
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--task_name", type=str, default="okvqa",
                        choices=["okvqa", "mmmu", "mmmu_pro"])
    parser.add_argument('--model_ids', default="123", type=str)
    parser.add_argument('--out_root', default="ensemble", type=str,
                        help="results/<out_root>/<task_name>/<model_ids> is used as the save dir")
    parser.add_argument("--dataset_type", type= str, default="train", 
                        choices=["test", "validation", "train"])
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--rectify', action='store_true',
                        help="run the epistemic-uncertainty rectification stage on the novel set after training")
    parser.add_argument('--rectify_alpha', default=10.0, type=float,
                        help="log-likelihood-ratio margin for choosing the 2-component GMM over a single Gaussian")
    parser.add_argument('--rectify_policy', default="mean", choices=["mean", "vote", "second", "router"],
                        help="fallback for rejected samples: average of base-model probabilities, their majority vote, "
                             "the ensemble's second choice, or the answer of --router_model")
    parser.add_argument('--router_model', default="Qwen3-VL-235B-A22B-Instruct", type=str,
                        help="model under results/inference_base_models whose answers rejected samples are routed to")
    arguments = parser.parse_args()
    run(arguments)

