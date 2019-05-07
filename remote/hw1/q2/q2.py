import sys
import itertools

input_path = sys.argv[1]

item_counts = {}
MIN_SUPPORT = 100

data = None
with open(input_path) as f:
    data = [line.strip().split(' ') for line in f.readlines()]

frequent_sets = [[]]
counts = [[]]
# First pass
for row in data:
    for item in row:
        if item in item_counts:
            item_counts[item] += 1
        else:
            item_counts[item] = 1
frequent_items = [item for item, cnt in item_counts.items() if cnt >= MIN_SUPPORT]
print(f"Frequent items after 1st pass: {len(frequent_items)}")
counts.append(item_counts)
frequent_sets.append(frequent_items)

for pass_index in range(2, 4):
    print(f"Pass #{pass_index}")
    counts_for_pass = {}
    for row in data:
        candidate_items = sorted([item for item in row if item in frequent_items])
        for comb in itertools.combinations(candidate_items, pass_index):
            if comb in counts_for_pass:
                counts_for_pass[comb] += 1
            else:
                counts_for_pass[comb] = 1
    frequent_sets_for_pass = [comb for comb, cnt in counts_for_pass.items() if cnt >= MIN_SUPPORT]
    print(f"Frequent sets after pass {pass_index}: {len(frequent_sets_for_pass)}")
    frequent_sets.append(frequent_sets_for_pass)
    counts.append(counts_for_pass)

    association_confidences = {}
    for frequent_set in frequent_sets_for_pass:
        for right in frequent_set:
            left = tuple([item for item in list(frequent_set) if item != right])
            if pass_index == 2:
                left = left[0]
            count_total = counts[pass_index][frequent_set]
            count_left = counts[pass_index - 1][left]
            association_confidences[(left, right)] = 1.0 * count_total / count_left
    best_associations = sorted(association_confidences.items(), key=lambda kv: (-kv[1], kv[0]))[:20]
    best_associations = [f"{a[0][0]}->{a[0][1]}, {a[1]}" for a in best_associations]
    print(f"Best associations: {best_associations}")
