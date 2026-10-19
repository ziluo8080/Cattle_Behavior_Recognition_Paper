"""Standard-library-only offline checks; no scientific package or model is imported."""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
import sys

ROOT = Path(__file__).resolve().parent
GENERATED = ROOT / "_verification_output"
GENERATED.mkdir(exist_ok=True)

def open_jsonl(relative):
    p = ROOT / relative
    return p.open(encoding="utf-8") if p.exists() else gzip.open(str(p)+".gz", "rt", encoding="utf-8")

CHECKS = Counter()
TOLERANCE = 1e-12


def rows(relative):
    with (ROOT / relative).open(encoding='utf-8-sig', newline='') as f:
        yield from csv.DictReader(f)


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8-sig'))


def check(condition, description):
    if not condition:
        raise AssertionError(description)
    CHECKS[description.split(':')[0]] += 1


def close(a, b, description):
    check(math.isclose(float(a), float(b), abs_tol=TOLERANCE, rel_tol=0), description)


def average(values):
    values = list(values)
    return math.fsum(values) / len(values)


def f1(matrix):
    out = []
    for i in range(7):
        denominator = sum(matrix[i]) + sum(matrix[j][i] for j in range(7))
        out.append(2 * matrix[i][i] / denominator if denominator else 0.0)
    return out


def load_confusions(relative):
    matrices, cells = {}, defaultdict(set)
    for row in rows(relative):
        key = (row['protocol'], row['fold'], row['model'], int(row['seed']))
        i, j, n = int(row['true_class']) - 1, int(row['predicted_class']) - 1, int(row['count'])
        check(0 <= i < 7 and 0 <= j < 7 and n >= 0, 'matrix_nonnegative_seven_classes')
        check((i, j) not in cells[key], 'matrix_unique_cells')
        cells[key].add((i, j))
        matrices.setdefault(key, [[0] * 7 for _ in range(7)])[i][j] = n
    for key, present in cells.items():
        check(len(present) == 49, 'matrix_complete_49_cells')
    return matrices


def export(relative, fields, records):
    with (GENERATED / relative).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)


def verify_main():
    base = 'results/all11_fast/'
    matrices = load_confusions(base + 'confusion_matrices.csv')
    calculated = {key: f1(cm) for key, cm in matrices.items()}
    check(len(matrices) == 310, 'main_310_model_protocol_fold_seed_records')
    stored = read(base + 'verified_summary.json')['protocols']
    for row in rows(base + 'seed_results.csv'):
        key = (row['protocol'], row['fold'], row['model'], int(row['seed']))
        close(average(calculated[key]), row['macro_f1'], 'seed_macro_f1:' + str(key))
    table4, table67, table8 = [], [], []
    for proto, block in stored.items():
        for model, score in block['scores'].items():
            keys = [k for k in calculated if k[0] == proto and k[2] == model]
            mean = average(average(calculated[k]) for k in keys)
            close(mean, score['mean'], 'table4_overall_mean:' + str((proto, model)))
            table4.append(dict(protocol=proto, model=model, macro_f1=mean, percent=mean * 100, model_records=len(keys)))
            for label in range(1, 8):
                value = average(calculated[k][label - 1] for k in keys)
                close(value, score['per_class_mean'][label - 1], 'table6_7_class_mean:' + str((proto, model, label)))
                table67.append(dict(table=6 if proto == 'TS' else 7, protocol=proto, model=model, class_id=label, f1=value, percent=value * 100))
            for seed, value in zip(score['seeds'], score['per_seed_fivefold_mean']):
                subset = [k for k in keys if k[3] == int(seed)]
                check(len(subset) == 5, 'five_folds_per_seed')
                close(average(average(calculated[k]) for k in subset), value, 'fivefold_seed_mean')
    for row in rows(base + 'class_results.csv'):
        subset = [k for k in calculated if k[0] == row['protocol'] and k[2] == row['model']]
        close(average(calculated[k][int(row['class_id']) - 1] for k in subset), row['f1'], 'saved_class_csv')
    for row in rows(base + 'fold_results.csv'):
        subset = [k for k in calculated if k[:3] == (row['protocol'], row['fold'], row['model'])]
        value = average(average(calculated[k]) for k in subset)
        close(value, row['macro_f1'], 'fold_mean:' + str((row['protocol'], row['fold'], row['model'])))
        check(len(subset) == int(row['seed_count']), 'fold_seed_count')
        check(all(sum(map(sum, matrices[k])) == int(row['targets']) for k in subset), 'fold_target_count')
        if row['protocol'] == 'OS':
            table8.append(dict(protocol='OS', fold=row['fold'], model=row['model'], macro_f1=value, percent=value * 100))
    export('RECOMPUTED_TABLE4.csv', ['protocol', 'model', 'macro_f1', 'percent', 'model_records'], table4)
    export('RECOMPUTED_TABLE6_7.csv', ['table', 'protocol', 'model', 'class_id', 'f1', 'percent'], table67)
    export('RECOMPUTED_TABLE8.csv', ['protocol', 'fold', 'model', 'macro_f1', 'percent'], table8)
    return len(matrices)


def verify_fixed20():
    base = 'results/fixed20/'
    matrices = load_confusions(base + 'confusion_matrices.csv')
    saved = read(base + 'verified_summary.json')['protocols']
    for proto, block in saved.items():
        for model, score in block['scores'].items():
            values = []
            for case in score['cases']:
                key = (proto, case['fold'], model, int(case['seed']))
                check(matrices[key] == case['scores']['CM'], 'fixed20_csv_matches_saved_matrix')
                value = average(f1(matrices[key]))
                close(value, case['scores']['macro_f1'], 'fixed20_case_macro')
                values.append(value)
            close(average(values), score['mean'], 'fixed20_mean')
    for row in rows(base + 'key_ablations.csv'):
        score = saved[row['protocol']]['scores'][row['method']]
        close(score['mean'], row['macro_f1'], 'fixed20_key_ablation_mean')
        close(saved[row['protocol']]['scores']['Base']['mean'] - score['mean'], row['Base_minus_variant'], 'fixed20_base_minus_variant')
    return len(matrices)


def verify_targets():
    universe = {}
    for line in open_jsonl('evidence/FINAL_46821_TARGET_KEYS.jsonl'):
        row = json.loads(line)
        key = (int(row['cow_id']), int(row['timestamp']))
        check(key not in universe, 'target_unique_key')
        check(1 <= int(row['label']) <= 7, 'target_seven_classes')
        universe[key] = int(row['label'])
    check(len(universe) == 46821, 'target_46821_count')
    cows = sorted({k[0] for k in universe})
    check(cows == list(range(1, 11)), 'target_ten_cow_ids')
    counts = Counter(universe.values())
    for row in rows('evidence/CLASS_SUPPORT.csv'):
        check(counts[int(row['label'])] == int(row['targets']), 'class_support_matches_keys')
    for row in rows('evidence/COW_CLASS_SUPPORT.csv'):
        subset = {k: v for k, v in universe.items() if k[0] == int(row['cow_id'])}
        check(len(subset) == int(row['targets']), 'cow_support_matches_keys')
        check(min(k[1] for k in subset) == int(row['first_timestamp']), 'cow_first_timestamp')
        check(max(k[1] for k in subset) == int(row['last_timestamp']), 'cow_last_timestamp')
        for label in range(1, 8):
            check(sum(v == label for v in subset.values()) == int(row['label_' + str(label)]), 'cow_class_support_matches_keys')
    seen, role_counts, role_labels, role_cows, test_coverage = defaultdict(set), Counter(), defaultdict(Counter), defaultdict(set), defaultdict(Counter)
    for line in open_jsonl('evidence/ALL_FOLD_TARGET_ROLE_INDEX.jsonl'):
        row = json.loads(line)
        key = (int(row['cow_id']), int(row['timestamp']))
        pf = (row['protocol'], row['fold'])
        role = row['selection_role']
        check(universe.get(key) == int(row['label']), 'role_key_and_label_in_universe')
        check(key not in seen[pf], 'role_unique_per_fold')
        seen[pf].add(key)
        check(role in ('train', 'val', 'test'), 'selection_role_definition')
        check(row['formal_role'] == ('TEST' if role == 'test' else 'FIT'), 'formal_fit_equals_train_plus_val')
        for r in ([role, 'fit'] if role != 'test' else [role]):
            group = pf + (r,)
            role_counts[group] += 1
            role_labels[group][int(row['label'])] += 1
            role_cows[group].add(key[0])
        if role == 'test':
            test_coverage[row['protocol']][key] += 1
    check(len(seen) == 10, 'ten_protocol_fold_combinations')
    for values in seen.values():
        check(values == universe.keys(), 'same_universe_all_ten_folds')
    for proto in ('TS', 'OS'):
        check(test_coverage[proto].keys() == universe.keys() and set(test_coverage[proto].values()) == {1}, 'each_protocol_test_covers_each_key_once')
    for row in rows('evidence/FOLD_TARGET_COUNTS.csv'):
        pf = (row['protocol'], row['fold'])
        for col, role in [('TRAIN', 'train'), ('VAL', 'val'), ('FIT_TRAIN_plus_VAL', 'fit'), ('TEST', 'test')]:
            check(role_counts[pf + (role,)] == int(row[col]), 'fold_role_count_csv')
        for col, role in [('TRAIN_cows', 'train'), ('VAL_cows', 'val'), ('TEST_cows', 'test')]:
            check(role_cows[pf + (role,)] == {int(c) for c in row[col].split(',')}, 'fold_role_cow_csv')
    for row in rows('evidence/FOLD_ROLE_CLASS_SUPPORT.csv'):
        key = (row['protocol'], row['fold'], row['role'])
        check(role_counts[key] == int(row['targets']), 'fold_role_support_targets')
        for label in range(1, 8):
            check(role_labels[key][label] == int(row['label_' + str(label)]), 'fold_role_class_support')
    fig2 = read('evidence/FIGURE2_EXACT_TARGET_KEY.json')
    check(((int(fig2['cow_id']), int(fig2['timestamp'])) in universe) == fig2['in_formal_46821'], 'figure2_membership_matches_explicit_development_role')
    check(fig2['in_formal_46821'] is False and fig2['source_role'] == 'development_TRAIN', 'figure2_is_development_illustration')
    nfig5 = 0
    for row in rows('evidence/FIGURE5_EXACT_TARGET_KEYS.csv'):
        key = (int(row['cow_id']), int(row['timestamp']))
        check(universe[key] == int(row['label']), 'figure5_key_and_label')
        plot = {(int(r['cow_id']), int(r['timestamp'])): r for r in rows('figures/plot_data/case_' + row['case'] + '.csv')}[key]
        for name in ['label', 'relative_seconds', 'current_repeat_prediction', 'history_prediction']:
            check(plot[name] == row[name], 'figure5_plot_data_exact_match')
        nfig5 += 1
    check(nfig5 == 62, 'figure5_62_targets')
    return {'unique_targets': len(universe), 'cow_ids': cows, 'class_support': dict(sorted(counts.items())),
            'protocol_fold_combinations': len(seen), 'figure5_targets': nfig5, 'figure2_is_formal_target': False}


def main():
    main_cases = verify_main()
    fixed_cases = verify_fixed20()
    targets = verify_targets()
    report = {'status': 'PASS', 'verification_scope': 'saved_confusion_matrices_and_target_definition_only',
              'tolerance': TOLERANCE, 'main_model_records': main_cases, 'fixed20_case_records': fixed_cases,
              'targets': targets, 'check_counts': dict(CHECKS),
              'training': False, 'inference': False, 'raw_source_reloaded': False,
              'bootstrap_recomputed': False,
              'bootstrap_limitation_zh': '该标准库脚本核对保存的混淆矩阵；逐目标预测与条件区间另由verify_predictions.py --bootstrap核对。',
              'mean_definition_zh': '先在每个协议/折/seed的7×7混淆矩阵上计算7类F1和宏F1，再对五折与可用seed等权平均；原论文RF每折只有seed42。不是先池化混淆矩阵再求F1。'}
    (GENERATED / 'VERIFICATION_REPORT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
