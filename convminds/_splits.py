from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass

from convminds.interfaces import SplitConfig, SplitPlan, StimulusSet


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


@dataclass
class _UnionFind:
    size: int

    def __post_init__(self) -> None:
        self.parent = list(range(self.size))
        self.rank = [0] * self.size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            self.parent[left_root] = right_root
            return
        if self.rank[left_root] > self.rank[right_root]:
            self.parent[right_root] = left_root
            return
        self.parent[right_root] = left_root
        self.rank[left_root] += 1


def build_group_labels(stimuli: StimulusSet, topic_splitting: bool) -> list[str]:
    uf = _UnionFind(len(stimuli))
    first_by_text: dict[str, int] = {}
    first_by_topic: dict[str, int] = {}

    for index, record in enumerate(stimuli):
        text_key = normalize_text(record.text)
        if text_key in first_by_text:
            uf.union(index, first_by_text[text_key])
        else:
            first_by_text[text_key] = index

        if topic_splitting and record.topic:
            topic_key = record.topic
            if topic_key in first_by_topic:
                uf.union(index, first_by_topic[topic_key])
            else:
                first_by_topic[topic_key] = index

    label_map: dict[int, str] = {}
    labels: list[str] = []
    next_label = 0
    for index in range(len(stimuli)):
        root = uf.find(index)
        if root not in label_map:
            label_map[root] = f"group-{next_label:04d}"
            next_label += 1
        labels.append(label_map[root])
    return labels


def _group_to_indices(group_labels: list[str]) -> list[list[int]]:
    group_order: list[str] = []
    group_indices: dict[str, list[int]] = {}
    for index, label in enumerate(group_labels):
        if label not in group_indices:
            group_indices[label] = []
            group_order.append(label)
        group_indices[label].append(index)
    return [group_indices[label] for label in group_order]


def _one_group_split(group_members: list[list[int]], train_size: float, random_state: int) -> list[tuple[list[int], list[int]]]:
    if len(group_members) < 2:
        raise ValueError("At least two split groups are required to produce a train/test split.")

    shuffled = list(range(len(group_members)))
    random.Random(random_state).shuffle(shuffled)
    train_group_count = int(math.floor(train_size * len(shuffled)))
    train_group_count = max(1, min(len(shuffled) - 1, train_group_count))
    train_group_ids = set(shuffled[:train_group_count])
    train_indices = [index for group_id in train_group_ids for index in group_members[group_id]]
    test_indices = [index for group_id in shuffled[train_group_count:] for index in group_members[group_id]]
    return [(sorted(train_indices), sorted(test_indices))]


def _kfold_group_split(group_members: list[list[int]], splits: int, random_state: int) -> list[tuple[list[int], list[int]]]:
    if splits < 2:
        raise ValueError("cv must be >= 2 for K-fold style splitting.")
    if len(group_members) < splits:
        raise ValueError("The number of split groups must be at least as large as cv.")

    shuffled = list(range(len(group_members)))
    random.Random(random_state).shuffle(shuffled)
    fold_groups = [shuffled[fold_index::splits] for fold_index in range(splits)]

    results: list[tuple[list[int], list[int]]] = []
    for fold_index in range(splits):
        test_group_ids = set(fold_groups[fold_index])
        train_indices = [index for group_id, members in enumerate(group_members) if group_id not in test_group_ids for index in members]
        test_indices = [index for group_id, members in enumerate(group_members) if group_id in test_group_ids for index in members]
        results.append((sorted(train_indices), sorted(test_indices)))
    return results


def build_split_plan(stimuli: StimulusSet, split_config: SplitConfig) -> list[SplitPlan]:
    group_labels = build_group_labels(stimuli, topic_splitting=split_config.topic_splitting)
    group_members = _group_to_indices(group_labels)

    if split_config.cv == 1:
        splits = _one_group_split(group_members, train_size=split_config.train_size, random_state=split_config.random_state)
    else:
        splits = _kfold_group_split(group_members, splits=split_config.cv, random_state=split_config.random_state)

    plans: list[SplitPlan] = []
    stimulus_ids = stimuli.ids()
    for index, (train_indices, test_indices) in enumerate(splits):
        plans.append(
            SplitPlan(
                index=index,
                train_indices=tuple(train_indices),
                test_indices=tuple(test_indices),
                train_stimulus_ids=tuple(stimulus_ids[position] for position in train_indices),
                test_stimulus_ids=tuple(stimulus_ids[position] for position in test_indices),
            )
        )
    return plans
