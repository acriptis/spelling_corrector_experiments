
from dp_components.levenshtein_searcher import SegmentTransducer
from utilities.recursive_dict_merge import recursive_dict_merge
from utilities.karta_slov_helper_fns import generate_karta_slov_costs_dict


def generate_operation_costs_dict(alphabet):
    ops_costs = SegmentTransducer.make_default_operation_costs(alphabet)
    distant_substitutions_costs = {
        "ться": {
            "цца": 1.1,
            "ца": 1.2},
        "тся": {
            "цца": 1.1,
            "ца": 1.2},
        "нибудь": {
            "нить": 1.0,
        },
        "а": {
            "aa": 1.0,
            "aaa": 1.1,
            "aaaа": 1.1,
        },
        "о": {
            "оо": 1.0,
            "ооо": 1.1,
            "оооо": 1.1,
        },
        "ч": {
            "чч": 1.0,
            "ччч": 1.1,
            "чччч": 1.1,
        }

    }

    merged_costs = recursive_dict_merge(ops_costs, distant_substitutions_costs)

    karta_slov_costs_dict = generate_karta_slov_costs_dict()

    operation_costs = recursive_dict_merge(merged_costs, karta_slov_costs_dict)
    return operation_costs
