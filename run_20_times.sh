#!/usr/bin/env bash
set -uo pipefail

# usage: bash run_20_times.sh [task_name] [runs] [extra sft_weighted.py args, e.g. --rectify_policy vote]
TASK_NAME="${1:-mmmu}"   # mmmu, mmmu_pro or okvqa
RUNS="${2:-10}"
OUTFILE="novel_acc_results_${TASK_NAME}.txt"

: > "$OUTFILE"          # start with an empty results file
befores=()
afters=()
aucs=()
baccs=()

for i in $(seq 1 "$RUNS"); do
    echo "=== Run $i / $RUNS ==="

    # Run the command, capture stdout+stderr, keep the LAST before/after accuracy and detection AUROC lines
    output=$(python sft_weighted.py --task_name "$TASK_NAME" --model_ids 1235 --rectify "${@:3}" 2>&1)
    before=$(grep -oE 'before rectification += +[0-9]+(\.[0-9]+)?' <<< "$output" | tail -n 1)
    after=$(grep -oE 'after rectification += +[0-9]+(\.[0-9]+)?' <<< "$output" | tail -n 1)
    auc=$(grep -oE 'Detection AUROC = [0-9]+(\.[0-9]+)?' <<< "$output" | tail -n 1)
    bacc=$(grep -oE 'Rejection balanced accuracy = [0-9]+(\.[0-9]+)?' <<< "$output" | tail -n 1)

    if [[ -z "$before" || -z "$after" || -z "$auc" || -z "$bacc" ]]; then
        echo "  WARNING: no rectification accuracy lines found this run, last lines of output:"
        tail -n 5 <<< "$output" | sed 's/^/    /'
        printf 'Run %2d: (no match)\n' "$i" >> "$OUTFILE"
    else
        before="${before##* }"   # everything after the last space = the number
        after="${after##* }"
        auc="${auc##* }"
        bacc="${bacc##* }"
        echo "  before = $before | after = $after | detection AUROC = $auc | rejection balanced acc = $bacc"
        printf 'Run %2d: before = %s | after = %s | detection AUROC = %s | rejection balanced acc = %s\n' \
               "$i" "$before" "$after" "$auc" "$bacc" >> "$OUTFILE"
        befores+=("$before")
        afters+=("$after")
        aucs+=("$auc")
        baccs+=("$bacc")
    fi
done

echo
echo "Results saved to $OUTFILE"

# ---- summary across all successful runs ----
if ((${#befores[@]} > 0)); then
    paste <(printf '%s\n' "${befores[@]}") <(printf '%s\n' "${afters[@]}") <(printf '%s\n' "${aucs[@]}") \
          <(printf '%s\n' "${baccs[@]}") | awk '
        function report(name, v,    i, mean, sum, ss, sd, min, max, d) {
            sum=0; ss=0; sd=0; min=v[1]; max=v[1]
            for (i=1; i<=n; i++) { sum+=v[i]; if (v[i]<min) min=v[i]; if (v[i]>max) max=v[i] }
            mean=sum/n
            if (n>1) { for (i=1; i<=n; i++) { d=v[i]-mean; ss+=d*d } sd=sqrt(ss/(n-1)) }
            printf "%-7s Mean: %.4f | StdDev: %.4f | SEM: %.4f | Min: %.4f | Max: %.4f\n", \
                   name, mean, sd, sd/sqrt(n), min, max
        }
        { n++; b[n]=$1; a[n]=$2; delta[n]=$2-$1; auc[n]=$3; bacc[n]=$4 }
        END {
            printf "\n=== Summary (%d runs) ===\n", n
            report("Before", b)
            report("After", a)
            report("Delta", delta)
            report("AUROC", auc)
            report("RejBAcc", bacc)
        }' | tee -a "$OUTFILE"
fi
