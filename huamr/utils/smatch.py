import penman
from smatchpp import Smatchpp, solvers


def calculate_smatch(ref_graphs: list[penman.Graph], pred_graphs: list[penman.Graph]):
    # https://github.com/BramVanroy/multilingual-text-to-amr/blob/main/src/multi_amr/run_amr_generation.py#L268
    ilp = solvers.ILP()
    smatch_metric = Smatchpp(alignmentsolver=ilp)

    filtered_refs = []
    filtered_preds = []
    for ref_graph, pred_graph in zip(ref_graphs, pred_graphs):
        try:
            ref = penman.encode(ref_graph)
            pred = penman.encode(pred_graph)
        except Exception as e:
            continue
        else:
            try:
                # To test that we can parse them with smatchpp (not necessarily compatible with penman!)
                _ = smatch_metric.score_pair(ref, pred)
            except Exception as exc:
                print(ref)
                print(pred)
                print(exc)
                print()
                continue
            else:
                filtered_refs.append(ref)
                filtered_preds.append(pred)

    if not filtered_refs or not filtered_preds:
        return {
            "smatch_precision": 0.0,
            "smatch_recall": 0.0,
            "smatch_fscore": 0.0,
            "smatch_unparsable": len(ref_graphs),
        }

    score, optimization_status = smatch_metric.score_corpus(filtered_refs, filtered_preds)

    try:
        score = score["main"]
    except KeyError:
        pass

    return {
        "smatch_precision": score["Precision"]["result"],
        "smatch_recall": score["Recall"]["result"],
        "smatch_fscore": score["F1"]["result"],
        "smatch_unparsable": len(ref_graphs) - len(filtered_refs),
    }
