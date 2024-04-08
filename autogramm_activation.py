import argparse
from contextlib import ExitStack

import numpy as np
import sys

from pyautogramm.activation import feature_activation_rule_extractor


if __name__ == "__main__":
    cmd = argparse.ArgumentParser()
    cmd.add_argument("--treebank", type=str, required=True)
    cmd.add_argument("--json", type=str, required=True)
    cmd.add_argument("--error", type=str, required=False)
    cmd.add_argument("--feature-name", type=str, required=True)
    cmd.add_argument("--feature-value", type=str, required=True)
    cmd.add_argument("--dep-filter", type=str, default="", required=False)
    cmd.add_argument("--feature-filter", type=str, default="", required=False)
    cmd.add_argument("--treebank-filter", type=str, default="")
    cmd.add_argument("--alpha-start", type=float, default=0.1)
    cmd.add_argument("--alpha-end", type=float, default=0.001)
    cmd.add_argument("--alpha-num", type=int, default=100)
    args = cmd.parse_args()

    # constraint on filters
    # there are hard constraints, i.e. like head_upos=VERB
    # if the feature is a set, it will be interpreted has mod_children_upos must contains VERB
    if len(args.dep_filter) > 0:
        dep_filters = [a.split("=") for a in args.dep_filter.split(",")]
        assert len(dep_filters) > 0
        assert all(len(f) == 2 for f in dep_filters)
    else:
        dep_filters = []

    # feature names that include these strings will be removed,
    # used for features that can spoilt the prediction.
    # e.g. in case of number agreement,
    # we want to remove all features containing "number" and "person"
    if len(args.feature_filter) == 0:
        feature_filter = []
    else:
        feature_filter = args.feature_filter.split(",")

    with ExitStack() as stack:
        if len(args.error) > 0:
            error_stream = stack.enter_context(open(args.error, "w"))
        else:
            error_stream = sys.stderr

        feature_activation_rule_extractor(
            args.treebank,
            args.json,
            # dependency filter
            lambda dep: (
                dep["gov.rel_synt"] not in ["orphan", "goeswith", "reparandum"]
                and all(
                        False
                        if k not in dep
                        else (
                            dep[k] == v if dep[k] == str else v in dep[k]
                        )
                        for k, v in dep_filters
                )
            ),
            # feature filter
            lambda degree, name: (
                # if we filter by POS, we need to remove them
                all(name != k for k, _ in dep_filters)
                and all(name.lower().find(f) < 0 for f in feature_filter)
                # use endswith because we don't want to match patterns of the form lemma_UPOS or lemmas_UPOS
                and (not name.lower().endswith("lemma"))
                and (not name.lower().endswith("lemmas"))
            ),
            feature_name=args.feature_name,
            feature_value=args.feature_value,
            alphas=np.linspace(args.alpha_start, args.alpha_end, args.alpha_num),
            treebank_filters=None if len(args.treebank_filter) == 0 else args.treebank_filter.split(","),
            error_stream=error_stream
        )
