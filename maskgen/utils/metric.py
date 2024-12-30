import argparse
import json
import logging
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import chain
from typing import Any, Callable, Dict, List, Set, Tuple

import numpy as np
from sklearn.metrics import (accuracy_score, auc, average_precision_score,
                             classification_report, precision_recall_curve, roc_auc_score)
from scipy.stats import entropy

from BERT_rationale_benchmark.utils import (
    Annotation, annotations_from_jsonl, load_documents, load_flattened_documents
)

# Logging setup
logging.basicConfig(level=logging.DEBUG, format='%(relativeCreated)6d %(threadName)s %(message)s')

# Classes
@dataclass(eq=True, frozen=True)
class Rationale:
    ann_id: str
    docid: str
    start_token: int
    end_token: int

    def to_token_level(self) -> List['Rationale']:
        return [Rationale(self.ann_id, self.docid, t, t + 1) for t in range(self.start_token, self.end_token)]

    @classmethod
    def from_annotation(cls, ann: Annotation) -> List['Rationale']:
        return [
            Rationale(ann.annotation_id, ev.docid, ev.start_token, ev.end_token)
            for ev_group in ann.evidences for ev in ev_group
        ]

    @classmethod
    def from_instance(cls, inst: dict) -> List['Rationale']:
        return [
            Rationale(inst['annotation_id'], rat['docid'], pred['start_token'], pred['end_token'])
            for rat in inst['rationales']
            for pred in rat.get('hard_rationale_predictions', [])
        ]

@dataclass(eq=True, frozen=True)
class PositionScoredDocument:
    ann_id: str
    docid: str
    scores: Tuple[float]
    truths: Tuple[bool]

    @classmethod
    def from_results(
        cls, instances: List[dict], annotations: List[Annotation], docs: Dict[str, List[Any]], use_tokens: bool = True
    ) -> List['PositionScoredDocument']:
        key_to_annotation = defaultdict(lambda: [False for _ in docs[next(iter(docs))]])
        field = 'soft_rationale_predictions' if use_tokens else 'soft_sentence_predictions'

        for ann in annotations:
            for ev in chain.from_iterable(ann.evidences):
                key = (ann.annotation_id, ev.docid)
                start, end = (ev.start_token, ev.end_token) if use_tokens else (ev.start_sentence, ev.end_sentence)
                for t in range(start, end):
                    key_to_annotation[key][t] = True

        result = []
        for inst in instances:
            for rat in inst['rationales']:
                docid = rat['docid']
                scores = rat[field]
                key = (inst['annotation_id'], docid)
                if key not in key_to_annotation:
                    key_to_annotation[key] = [False] * len(docs[docid])
                result.append(PositionScoredDocument(inst['annotation_id'], docid, tuple(scores), tuple(key_to_annotation[key])))

        return result

# Utility Functions
def calculate_f1(precision, recall):
    return 0 if precision == 0 or recall == 0 else 2 * (precision * recall) / (precision + recall)

def compute_partial_match_scores(truth, pred, thresholds):
    ann_to_rat = _keyed_rationale_from_list(truth)
    pred_to_rat = _keyed_rationale_from_list(pred)
    
    ious = compute_ious(ann_to_rat, pred_to_rat)
    
    scores = []
    for threshold in thresholds:
        scores.append(compute_threshold_scores(ious, threshold, ann_to_rat, pred_to_rat))
    return scores

def _keyed_rationale_from_list(rats: List[Rationale]) -> Dict[Tuple[str, str], Rationale]:
    ret = defaultdict(set)
    for r in rats:
        ret[(r.ann_id, r.docid)].add(r)
    return ret

def compute_ious(ann_to_rat, pred_to_rat):
    ious = defaultdict(dict)
    for key in set(ann_to_rat.keys()) | set(pred_to_rat.keys()):
        for p in pred_to_rat.get(key, []):
            ious[key][p] = max(
                compute_iou(p, t)
                for t in ann_to_rat.get(key, [])
            )
    return ious

def compute_iou(p, t):
    overlap = len(set(range(p.start_token, p.end_token)) & set(range(t.start_token, t.end_token)))
    union = len(set(range(p.start_token, p.end_token)) | set(range(t.start_token, t.end_token)))
    return 0 if union == 0 else overlap / union

def compute_threshold_scores(ious, threshold, ann_to_rat, pred_to_rat):
    threshold_tps = {
        key: sum(1 for score in ious[key].values() if score >= threshold)
        for key in ious
    }
    
    total_truths = sum(len(ann_to_rat[key]) for key in ann_to_rat)
    total_preds = sum(len(pred_to_rat[key]) for key in pred_to_rat)

    micro_precision = sum(threshold_tps.values()) / total_preds if total_preds > 0 else 0
    micro_recall = sum(threshold_tps.values()) / total_truths if total_truths > 0 else 0

    return {
        "threshold": threshold,
        "micro": {
            "p": micro_precision,
            "r": micro_recall,
            "f1": calculate_f1(micro_precision, micro_recall)
        }
    }

# Main Execution
def main():
    parser = argparse.ArgumentParser(description="Compute rationale and classification scores.")
    parser.add_argument('--data_dir', required=True, help='Directory containing {train,val,test}.jsonl files')
    parser.add_argument('--split', required=True, help='Split to evaluate: {train,val,test}')
    parser.add_argument('--results', required=True, help='Results file (JSONL)')
    parser.add_argument('--score_file', help='Output file for scores')

    args = parser.parse_args()

    results = load_jsonl(args.results)
    annotations = annotations_from_jsonl(os.path.join(args.data_dir, f"{args.split}.jsonl"))
    docs = load_flattened_documents(args.data_dir, set(chain.from_iterable(res['docid'] for res in results)))

    paired_scores = PositionScoredDocument.from_results(results, annotations, docs)

    # Evaluate metrics
    metrics = score_soft_tokens(paired_scores)

    # Output results
    if args.score_file:
        with open(args.score_file, 'w') as f:
            json.dump(metrics, f, indent=4)
    else:
        print(json.dumps(metrics, indent=4))

if __name__ == "__main__":
    main()
