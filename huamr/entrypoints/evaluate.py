import os

import pandas as pd
import click
import penman
from smatchpp import Smatchpp, solvers
from tqdm import tqdm


tqdm.pandas()

ilp = solvers.ILP()
measure = Smatchpp(alignmentsolver=ilp)


def remove_wiki_from_graph(graph: penman.Graph) -> penman.Graph:
    # https://github.com/BramVanroy/multilingual-text-to-amr/blob/efc1f7249bda34cd01dbe3ced2deaa5edeff84b8/src/multi_amr/utils/__init__.py#L79
    # modified from SPRING
    triples = []
    for t in graph.triples:
        v1, rel, v2 = t
        if rel == ":wiki":
            t = penman.Triple(v1, rel, "+")
        triples.append(t)

    return penman.Graph(triples, metadata=graph.metadata)


def score(row, ref_column, pred_column):
    try:
        ref_graph = penman.decode(row[ref_column].replace('\n', ' '))
        pred_graph = penman.decode(row[pred_column].replace('\n', ' '))

        row[ref_column] = penman.encode(remove_wiki_from_graph(ref_graph))
        row[pred_column] = penman.encode(remove_wiki_from_graph(pred_graph))

        smatch_score = measure.score_pair(row[ref_column], row[pred_column])

        row['smatchpp_f1'] = smatch_score['main']['F1']
        row['smatchpp_prec'] = smatch_score['main']['Precision']
        row['smatchpp_rec'] = smatch_score['main']['Recall']

    except Exception as e:
        row['smatchpp_f1'] = None
        row['smatchpp_prec'] = None
        row['smatchpp_rec'] = None
    return row

@click.command()
@click.argument('data_path')
@click.argument('ref_column')
@click.argument('pred_column')
@click.argument('out_path')
def main(data_path, ref_column, pred_column, out_path):
    df = pd.read_csv(data_path)
    df = df.progress_apply(score, axis=1, ref_column=ref_column, pred_column=pred_column)

    df.to_csv(os.path.join(out_path, 'evaluated.csv'), index=False)

    print('Smatchpp F1 mean:', df['smatchpp_f1'].mean())
    print('Smatchpp Precision mean:', df['smatchpp_prec'].mean())
    print('Smatchpp Recall mean:', df['smatchpp_rec'].mean())
    print('Unparsable AMR graphs:', df['smatchpp_f1'].isna().sum())


if __name__ == "__main__":
    main()
