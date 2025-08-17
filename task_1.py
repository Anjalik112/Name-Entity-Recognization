import os
import re

ANNOTATIONS_DIR = '/Users/anjalikulkarni/Desktop/Assignment1/CADEC-lPWNPfjE-/data/cadec/original'

LABELS = ("ADR", "Drug", "Disease", "Symptom")
label_entities = {k: set() for k in LABELS}

def normalize(text: str) -> str:
    # normalize entity surface form a bit
    t = text.strip().lower()
    t = re.sub(r'\s+', ' ', t)           # collapse whitespace
    t = t.strip(' .,:;!?()[]{}"\'')      # trim punctuation at ends
    return t

def collect_entities(dirpath: str):
    files = [f for f in os.listdir(dirpath) if f.endswith('.ann')]
    if not files:
        print("No .ann files found. Is the path correct?")
        return

    for fname in files:
        fpath = os.path.join(dirpath, fname)
        with open(fpath, 'r', encoding='utf-8') as fh:
            for line in fh:
                line = line.rstrip('\n')
                if not line or line[0] in {'#', 'A', 'R'}:
                    # comments / annotator notes / attributes / relations → skip
                    continue
                if not line.startswith('T'):
                    continue  # only entity lines carry text spans

                # BRAT format: ID \t TYPE + SPAN(S) \t TEXT
                parts = line.split('\t')
                if len(parts) < 3:
                    continue  # malformed

                meta = parts[1]              # e.g. "ADR 9 19" or "ADR 9 19;25 30"
                text = parts[2]              # entity surface text from the post
                label = meta.split()[0]      # first token is the label

                if label in label_entities:
                    label_entities[label].add(normalize(text))

collect_entities(ANNOTATIONS_DIR)

for label, ents in label_entities.items():
    print(f"Label: {label}")
    print(f"Distinct Entities: {len(ents)}")
    print(f"Sample (up to 10): {sorted(list(ents))[:10]}")
    print("=" * 60)

import os
from collections import defaultdict

# Path to the 'original' annotation files
directory = '/Users/anjalikulkarni/Desktop/Assignment1/CADEC-lPWNPfjE-/data/cadec/original'


# Dictionary to store unique entities for each label
distinct_entities = defaultdict(set)

# Iterate through all annotation files in the directory
for filename in os.listdir(directory):
    if not filename.endswith('.ann'):
        continue
    filepath = os.path.join(directory, filename)
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('#'):
                continue  # Skip comments and empty lines
            parts = line.split('\t')
            if len(parts) < 3:
                continue  # Skip malformed lines
            tag, label_ranges, entity_text = parts[0], parts[1], parts[2]
            label = label_ranges.split(' ')[0]  # Label is the first word in the second column
            distinct_entities[label].add(entity_text)

# Print the results
for label in ['ADR', 'Drug', 'Disease', 'Symptom']:
    entities = distinct_entities[label]
    print(f'Label: {label}')
    print(f'Total unique entities: {len(entities)}')
    print(f'Example entities: {list(entities)[:10]}')
    print('-' * 40) 