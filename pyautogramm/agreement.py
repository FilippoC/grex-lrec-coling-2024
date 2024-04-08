import collections
import os
import sys
import glob
import json

import numpy as np
from scipy.stats import chisquare
import scipy
import skglm

import pyximport
pyximport.install()
import pyautogramm.features

import pyautogramm.data
import time


# stolen form here: https://github.com/Aditi138/LASE-Agreement/blob/main/utils.py#L209
# and slightly modified
def is_agreement(chance_agreement, agree, disagree, p_value_threshold=0.01, effect_size_threshold=0.5):
    if agree < disagree:
        return False
    leaftotal = agree + disagree

    empirical_distr = [chance_agreement, 1 - chance_agreement]

    expected_agree = empirical_distr[0] * leaftotal
    expected_disagree = empirical_distr[1] * leaftotal

    if min(expected_disagree, expected_agree) < 5:  #cannot apply the chi-squared test, return chance agreement
        return False

    T, p = chisquare([disagree, agree], [expected_disagree, expected_agree])
    w = np.sqrt(T / leaftotal)

    if p < p_value_threshold and w > effect_size_threshold :  # reject the null
        return True
    else:
        return False


def morphological_agreement_rule_extractor(
        sud_path,
        output_path,
        dependency_predicate,
        feature_predicate,
        feature_1_name,
        feature_2_name,
        alphas,
        max_degree=2,
        min_feature_occurence=5,
        treebank_filters=None,
        error_stream=sys.stderr,
        p_value_threshold=0.01,
        effect_size_threshold=0.5
):
    treebank_paths = glob.glob(os.path.join(sud_path, "*"))

    # filter
    if treebank_filters is not None:
        treebank_paths = [path for path in treebank_paths if any(path.find(f) > 0 for f in treebank_filters)]

    extracted_data = dict()
    for i, treebank_path in enumerate(treebank_paths):
        treebank_name = os.path.basename(treebank_path)

        # find all conllu files for treebank
        conllu_paths = glob.glob(os.path.join(treebank_path, "*.conllu"))
        if len(conllu_paths) == 0:
            print("Skipping treebank %s because there is no conllu file!" % treebank_name, file=error_stream, flush=True)
            continue

        output_pre = "%s\t(%i / %i):\t" % (treebank_name, i+1, len(treebank_paths))

        # Read data
        print("%s%s" % (output_pre, "reading data"), flush=True)
        deps = list()
        for conllu_path in conllu_paths:
            data = pyautogramm.data.read(conllu_path)
            deps.extend(
                pyautogramm.data.extract_dependencies(
                    data,
                    split_head_rel=True,
                    add_closed_pos_tags_lemma=True,
                    add_similar_pos_tags=True
                )
            )

        # filter deps
        print("%s%s" % (output_pre, "filtering dependencies"), flush=True)
        filtered_deps = list()
        for dep in deps:
            if dependency_predicate(dep) and feature_1_name in dep and feature_2_name in dep:
                filtered_deps.append(dep)

        if len(filtered_deps) == 0:
            print("Skipping treebank %s because there is no dependency to analyse!" % treebank_name, file=error_stream, flush=True)
            continue

        print("%s%s" % (output_pre, "Number of dependencies after filtering: %i / %i" % (len(filtered_deps), len(deps))), flush=True)

        # extract features
        print("%s%s" % (output_pre, "extracting features"), flush=True)
        feature_set = pyautogramm.features.FeatureSet()

        feature_set.add_feature(pyautogramm.features.AllSingletonFeatures(
            predicate=lambda name: (feature_predicate(1, name) and name != feature_1_name and name != feature_2_name)
        ))
        for degree in range(2, max_degree + 1):
            feature_set.add_feature(pyautogramm.features.AllProductFeatures(
                degree=degree,
                min_occurences=min_feature_occurence,
                predicate=lambda name, degree=degree: (feature_predicate(degree, name) and name != feature_1_name and name != feature_2_name)
            ))

        try:
            feature_set.init_from_data(filtered_deps)
            X = feature_set.build_features(filtered_deps, sparse=True)
            if X.shape[1] == 0:
                print("Skipping treebank %s because there is no extracted feature!" % treebank_name, file=error_stream, flush=True)
                continue
        except RuntimeError:
            print("Skipping treebank %s because there is no extracted feature!" % treebank_name, file=error_stream, flush=True)
            continue

        # build targets
        y = np.empty((len(filtered_deps),))
        for i, dep in enumerate(filtered_deps):
            assert type(dep[feature_1_name]) == str
            assert type(dep[feature_2_name]) == str
            y[i] = 1 if dep[feature_1_name] == dep[feature_2_name] else 0


        filtered_deps_len = len(filtered_deps)
        n_yes = int(y.sum())

        extracted_data[treebank_name] = dict()
        extracted_data[treebank_name]["filtered_deps_len"] = len(filtered_deps)
        extracted_data[treebank_name]["n_yes"] = int(y.sum())
        extracted_data[treebank_name]["intercepts"] = list()

        # extract rules
        all_rules = set()
        ordered_rules = list()

        # To compute the chi-square test, we need a base distribution
        # we assume our base hypothesis is that there is chance agreement.
        # For example, for the number feature we can have either singular or plural has a value.
        # we can estimate, from the dataset, the probability p(singular) and p(plural).
        # Then, for a given set of dependencies,
        # the probability of "chance agreement" is: p(singular) * p(singular) + p(plural) * p(plural).
        # Note that this extends to non-binary features easily.
        # unary_feature_counter = collections.Counter()
        # for dep in filtered_deps:
        #     unary_feature_counter[dep[feature_1_name]] += 1
        #     unary_feature_counter[dep[feature_2_name]] += 1
        # unary_feature_sum = sum(unary_feature_counter.values())
        # base_p_chance_agreement = sum(
        #     (v / unary_feature_sum) ** 2
        #     for v in unary_feature_counter.values()
        # )

        for j, alpha in enumerate(alphas):
            print("%s%s" % (output_pre, "extracting rules (%i / %i)" % (j+1, len(alphas))), flush=True)
            model = skglm.SparseLogisticRegression(
                alpha=alpha,
                fit_intercept=True,
                max_iter=20,
                max_epochs=1000,
            )
            model.fit(X, y)
            extracted_data[treebank_name]["intercepts"].append((alpha, model.intercept_))

            for name, (value, idx) in feature_set.feature_weights(model.coef_[0]).items():
                if name not in all_rules:
                    all_rules.add(name)
                    col = np.asarray(X[:, idx].todense())
                    idx_col = col.squeeze(1)

                    with_feature_selector = idx_col > 0
                    without_feature_selector = np.logical_not(with_feature_selector)

                    matched = y[with_feature_selector]
                    n_matched = len(matched)
                    n_pattern_positive_occurence = matched.sum()
                    n_pattern_negative_occurence = n_matched - n_pattern_positive_occurence

                    # is_agreement_rule = is_agreement(
                    #     base_p_chance_agreement,
                    #     n_pattern_positive_occurence,
                    #     n_pattern_negative_occurence,
                    #     p_value_threshold=p_value_threshold,
                    #     effect_size_threshold=effect_size_threshold
                    # )

                    # Fisher exact test,
                    # we don't use this anymore
                    """
                    if decision == "yes":
                        table = np.array([
                            [n_pattern_positive_occurence, y[without_feature_selector].sum()],
                            [n_pattern_negative_occurence, (1 - y[without_feature_selector]).sum()]
                        ])
                    else:
                        # the two lines are swapped compared to the yes case, not sure that this is the right thing to do
                        table = np.array([
                            [n_pattern_negative_occurence, (1 - y[without_feature_selector]).sum()],
                            [n_pattern_positive_occurence, y[without_feature_selector].sum()]
                        ])
                    p_value = scipy.stats.fisher_exact(table)[1]
                    p_value_greater = scipy.stats.fisher_exact(table, "greater")[1]
                    p_value_less = scipy.stats.fisher_exact(table, "less")[1]
                    """
                    mu = (n_yes/filtered_deps_len)
                    a = (n_pattern_positive_occurence/n_matched)
                    gstat =  2 * n_matched * (
                            ( (a * np.log(a)) if a > 0 else 0) - a * np.log(mu)
                            + ( ((1 - a) * np.log(1 - a)) if (1 - a) > 0 else 0) - (1 - a) * np.log(1 - mu)
                            )
                    p_value = 1 - scipy.stats.chi2.cdf(gstat,1)
                    cramers_phi = np.sqrt((gstat/n_matched))
                    
                    expected = (n_matched*n_yes) / filtered_deps_len
                    delta_observed_expected = n_pattern_positive_occurence - expected
                    
                    if p_value < 0.01 and delta_observed_expected > 0:
                        decision = 'yes'
                        coverage = (n_pattern_positive_occurence/n_yes)*100
                        presicion = (n_pattern_positive_occurence/n_matched)*100
                    else:
                        decision = 'no'
                        coverage = (n_pattern_negative_occurence/(filtered_deps_len - n_yes))*100
                        presicion = (n_pattern_negative_occurence/n_matched)*100

                    ordered_rules.append({
                        "pattern": name,
                        "n_pattern_occurence": idx_col.sum(),
                        "n_pattern_positive_occurence": n_pattern_positive_occurence,
                        "decision": decision,
                        "alpha": alpha,
                        "value": value,
                        "coverage": coverage,
                        "precision": presicion,
                        "delta": delta_observed_expected,
                        "g-statistic": gstat,
                        "p-value": p_value,
                        "cramers_phi": cramers_phi
                    })

        extracted_data[treebank_name]["rules"] = ordered_rules

        #if len(extracted_data) == 3:
        #    break

    print("Done.", flush=True)
    with open(output_path, 'w', encoding="utf-8") as out_stream:
        json.dump(extracted_data, out_stream)