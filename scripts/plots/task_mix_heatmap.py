#!/usr/bin/env python3
import argparse
import csv
import os
from collections import Counter, defaultdict
from typing import List, Dict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_counts(tsv_path: str, labels_keep: set[str], top_uses: int, keep_modalities: List[str]) -> tuple[List[str], List[str], np.ndarray]:
	counts = Counter()
	use_tot = Counter()
	mod_tot = Counter()
	with open(tsv_path, 'r', encoding='utf-8') as f:
		r = csv.DictReader(f, delimiter='\t')
		for row in r:
			label = (row.get('Label') or '').strip()
			if label not in labels_keep:
				continue
			m = (row.get('modality') or '').strip()
			u = (row.get('use') or '').strip()
			if not m or m.lower() == 'unknown' or m == '(unknown)':
				continue
			if not u or u.lower() == 'unknown' or u == '(unknown)':
				continue
			m2 = m if m in keep_modalities else 'Other'
			counts[(m2, u)] += 1
			use_tot[u] += 1
			mod_tot[m2] += 1

	# Select top uses by total frequency
	uses_sorted = [u for u, _ in use_tot.most_common(top_uses)]
	modalities = keep_modalities + (['Other'] if any(m not in keep_modalities for m in mod_tot) else [])

	M = np.zeros((len(uses_sorted), len(modalities)), dtype=int)
	for i, u in enumerate(uses_sorted):
		for j, m in enumerate(modalities):
			M[i, j] = counts.get((m, u), 0)

	return uses_sorted, modalities, M


def save_matrix_tsv(out_tsv: str, uses: List[str], modalities: List[str], M: np.ndarray) -> None:
	with open(out_tsv, 'w', encoding='utf-8', newline='') as f:
		writer = csv.writer(f, delimiter='\t')
		writer.writerow(['Use'] + modalities + ['Total'])
		for i, u in enumerate(uses):
			row = [u] + [str(int(x)) for x in M[i]] + [str(int(M[i].sum()))]
			writer.writerow(row)
		writer.writerow(['Total'] + [str(int(M[:, j].sum())) for j in range(M.shape[1])] + [str(int(M.sum()))])


def plot_heatmap(out_png: str, uses: List[str], modalities: List[str], M: np.ndarray, title: str) -> None:
	fig, ax = plt.subplots(figsize=(max(6, len(modalities)*1.2), max(6, len(uses)*0.35)))
	im = ax.imshow(M, cmap='Blues', aspect='auto')
	# Labels
	ax.set_xticks(np.arange(len(modalities)))
	ax.set_xticklabels(modalities, rotation=45, ha='right')
	ax.set_yticks(np.arange(len(uses)))
	ax.set_yticklabels(uses)
	# Annotate
	for i in range(M.shape[0]):
		for j in range(M.shape[1]):
			v = int(M[i, j])
			if v > 0:
				ax.text(j, i, str(v), va='center', ha='center', fontsize=8, color='#0a2540')
	plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Count')
	ax.set_xlabel('Modality')
	ax.set_ylabel('Use (Top)')
	ax.set_title(title)
	plt.tight_layout()
	plt.savefig(out_png, dpi=200)
	plt.close(fig)


def main() -> None:
	parser = argparse.ArgumentParser(description='Plot modality-use heatmap (hide unknown), labels in {1,3,4}')
	parser.add_argument('--input', default=os.path.abspath(os.path.join(os.path.dirname(__file__), 'checked_modality_use.tsv')))
	parser.add_argument('--output-png', default=os.path.abspath(os.path.join(os.path.dirname(__file__), 'plots', 'task_mix_heatmap.png')))
	parser.add_argument('--output-tsv', default=os.path.abspath(os.path.join(os.path.dirname(__file__), 'plots', 'task_mix_heatmap.tsv')))
	parser.add_argument('--top-uses', type=int, default=12)
	parser.add_argument('--modalities', nargs='*', default=['Written','Speech','Multimodal/Multimedia'])
	args = parser.parse_args()

	labels_keep = {'1','3','4'}
	uses, modalities, M = load_counts(args.input, labels_keep, args.top_uses, args.modalities)
	os.makedirs(os.path.dirname(args.output_png), exist_ok=True)
	save_matrix_tsv(args.output_tsv, uses, modalities, M)
	plot_heatmap(args.output_png, uses, modalities, M, title='Task Mix by Modality (Labels 1/3/4, unknown hidden)')


if __name__ == '__main__':
	main()


