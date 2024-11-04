import penman
from amr import AMR


def remove_wiki_from_graph(graph: penman.Graph) -> penman.Graph:
    # modified from SPRING
    triples = []
    for t in graph.triples:
        v1, rel, v2 = t
        if rel == ":wiki":
            continue
        triples.append(t)

    return penman.Graph(triples, metadata=graph.metadata)


def is_amr_valid(amr):
    try:
        amr = amr.replace('\n', ' ')

        if amr.count('(') == 0:
            return False
        if amr.count(')') == 0:
            return False

        graph = penman.decode(amr)
        penman.encode(remove_wiki_from_graph(graph))

        theamr = AMR.parse_AMR_line(amr)
        if theamr is None:
            return False
        
        return True
    except Exception:
        return False


def filter_valid_amrs(df, amr_column):
    filtered_df = df[df[amr_column].apply(is_amr_valid)]
    print(f"Filtered out {len(df) - len(filtered_df)} non-valid AMRs.")
    return filtered_df
