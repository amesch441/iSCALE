import sys
from utils import load_tsv, write_lines

prefix = sys.argv[1]  # e.g. 'data/markers/lymphoid/'

df = load_tsv(prefix+'markers.tsv')
df = df.astype(bool)
genes = df.index.to_numpy()

for structure, isin in df.items():
    ges = genes[isin]
    write_lines(ges, f'{prefix}{structure}/gene-names.txt')

write_lines(df.columns.to_list(), f'{prefix}structures.txt')
write_lines(df.index.to_list(), f'{prefix}genes.txt')
