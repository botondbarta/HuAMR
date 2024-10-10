import penman
from amr import AMR


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


def is_amr_valid(amr):
    try:
        amr = amr.replace('\n', ' ')

        if amr.count('(') == 0:
            return False
        if amr.count(')') == 0:
            return False
        # this line fires for emojis in strings and valid amrs but more closing parentheses
        # if amr.count('(') != amr.count(')'):
        #    return False

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
