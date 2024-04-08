import itertools
import collections
import numpy as np
import scipy.sparse

from pyautogramm.utils import Dict

# Should not be used,
# I implemented this because unregularized intercept term
# was not implemented in celer. However, as I now use skglm,
# this is useless...
class InterceptFeature:
    def __init__(self):
        self.initialized = True

    def init_from_data(self, data):
        pass

    def build_features(self, X, data, offset):
        X[:, offset] = 1

    def get_all_names(self):
        return ["intercept"]

    def __len__(self):
        return 1


class ClassFeature:
    def __init__(self, name, neg=False):
        self.name = name
        self.initialized = False
        self.neg = neg

    def init_from_data(self, data):
        values = set()
        for dep in data:
            if self.name in dep:
                v = dep[self.name]
                assert type(v) == str
                values.add(v)
        if len(values) == 0:
            raise RuntimeError("No value found for feature")
        self.dict = Dict(values)
        self.initialized = True

    def build_features(self, X, data, offset):
        for i, dep in enumerate(data):
            if self.name in dep:
                value = dep[self.name]
                value_id = self.dict.str_to_id(value)
                if self.neg:
                    for other_value_id in range(len(self)):
                        if other_value_id != value_id:
                            X[i, offset + other_value_id] = 1
                else:
                    X[i, offset + value_id] = 1

    def get_all_names(self):
        if self.neg:
            return ["%s!=%s" % (self.name, v) for v in self.dict._id_to_str]
        else:
            return ["%s=%s" % (self.name, v) for v in self.dict._id_to_str]

    def __len__(self):
        if not self.initialized:
            raise RuntimeError("Feature not initialized")
        else:
            return len(self.dict)


class IndicatorFeature:
    def __init__(self, name, neg=False):
        self.name = name
        self.initialized = False
        self.neg = neg

    def init_from_data(self, data):
        values = set()
        for dep in data:
            if self.name in dep:
                v = dep[self.name]
                assert type(v) == set
                values.update(v)
        if len(values) == 0:
            raise RuntimeError("No value found for feature")
        self.dict = Dict(values)
        self.initialized = True

    def build_features(self, X, data, offset):
        for i, dep in enumerate(data):
            if self.name in dep:
                if self.neg:
                    for value, value_id in self.dict._str_to_id.items():
                        if value not in dep[self.name]:
                            X[i, offset + value_id] = 1
                else:
                    for value in dep[self.name]:
                        value_id = self.dict.str_to_id(value)
                        X[i, offset + value_id] = 1

    def get_all_names(self):
        if self.neg:
            return ["%s!=%s" % (self.name, v) for v in self.dict._id_to_str]
        else:
            return ["%s=%s" % (self.name, v) for v in self.dict._id_to_str]

    def __len__(self):
        if not self.initialized:
            raise RuntimeError("Feature not initialized")
        else:
            # it always takes a single column
            return len(self.dict)


class AllFeatures:
    def __init__(self, predicate=None, include_neg=False):
        self.predicate = predicate
        self.initialized = False
        self.include_neg = include_neg

    def init_from_data(self, data):
        class_feature_names = set()
        indicator_feature_names = set()
        for dep in data:
            for k, v in dep.items():
                if self.predicate is not None and not self.predicate(k):
                    continue
                if type(v) == str:
                    class_feature_names.add(k)
                elif type(v) == set:
                    indicator_feature_names.add(k)
                else:
                    raise RuntimeError("Unusable data type for feature %s: %s" % (k, type(v)))

        if len(class_feature_names.intersection(indicator_feature_names)) != 0:
            raise RuntimeError("Error in feature types")

        self.features = list()
        self.len_ = 0
        negs = [False, True] if self.include_neg else [False]
        for name in class_feature_names:
            for neg in negs:
                feature = ClassFeature(name, neg=neg)
                feature.init_from_data(data)
                self.len_ += len(feature)
                self.features.append(feature)

        for name in indicator_feature_names:
            for neg in negs:
                feature = IndicatorFeature(name, neg=neg)
                feature.init_from_data(data)
                self.len_ += len(feature)
                self.features.append(feature)
        self.initialized = True

    def build_features(self, X, data, offset):
        offset2 = 0
        for feature in self.features:
            feature.build_features(X, data, offset + offset2)
            offset2 += len(feature)

    def get_all_names(self):
        return itertools.chain(*[feature.get_all_names() for feature in self.features])

    def __len__(self):
        if not self.initialized:
            raise RuntimeError("Feature not initialized")
        else:
            # it always takes a single column
            return self.len_


class ClassProductFeature:
    def __init__(self, degree=2, min_occurences=1, predicate=None, include_neg=False):
        self.predicate = predicate
        self.initialized = False
        self.degree = degree
        self.min_occurences = min_occurences
        self.include_neg = include_neg

    def init_from_data(self, data):
        class_feature_names = set()
        indicator_feature_names = set()
        key_values = collections.defaultdict(lambda:set())  # only use in case of negative features
        for dep in data:
            for k, v in dep.items():
                if self.predicate is not None and not self.predicate(k):
                    continue
                if type(v) == str:
                    class_feature_names.add(k)
                    if self.include_neg:
                        key_values[k].add(v)
                elif type(v) == set:
                    indicator_feature_names.add(k)
                    if self.include_neg:
                        key_values[k].update(v)
                else:
                    raise RuntimeError("Unusable data type for feature %s: %s" % (k, type(v)))

        if len(class_feature_names.intersection(indicator_feature_names)) != 0:
            raise RuntimeError("Error in feature types")

        merged_features = {(False, k) for k in class_feature_names}.union((True, k) for k in indicator_feature_names)

        feature_count = collections.defaultdict(lambda: 0)
        for dep in data:
            for names in itertools.combinations(merged_features, self.degree):
                for negs in list(itertools.product([False, True] if self.include_neg else [False], repeat=self.degree)):
                    all_keys = []
                    for is_neg, (is_set_feature, name) in zip(negs, names):
                        if name not in dep:
                            break  # cannot build feature for this dep

                        if is_neg:
                            if is_set_feature:
                                non_matched_values = key_values[name].difference(dep[name])
                            else:
                                non_matched_values = key_values[name].difference({dep[name]})
                            if len(all_keys) == 0:
                                all_keys = [[(name, True, v)] for v in non_matched_values]
                            else:
                                all_keys = [
                                    k + [(name, True, v)]
                                    for k in all_keys
                                    for v in non_matched_values
                                ]
                        else:
                            if is_set_feature:
                                if len(all_keys) == 0:
                                    all_keys = [[(name, False, v)] for v in dep[name]]
                                else:
                                    all_keys = [
                                        k + [(name, False, v)]
                                        for k in all_keys
                                        for v in dep[name]
                                    ]
                            else:
                                if len(all_keys) == 0:
                                    all_keys = [[(name, False, dep[name])]]
                                else:
                                    all_keys = [k + [(name, False, dep[name])] for k in all_keys]
                    else:
                        # exit without break
                        for key in all_keys:
                            key = tuple(key)
                            feature_count[key] += 1

        # filter on number of occurences
        if self.min_occurences > 1:
            filtered_features = list(k for k, v in feature_count.items() if v >= self.min_occurences)
        else:
            filtered_features = list(feature_count.keys())

        self.initialized = True
        self.valid_features = filtered_features
        self.n_features = len(filtered_features)

    def build_features(self, X, data, int offset):
        # Compliqué à adapter avec les features negatives
        cdef int i, j

        for i, dep in enumerate(data):
            for j, feature in enumerate(self.valid_features):
                match = True
                for k, n, v in feature:
                    if k not in dep:
                        match = False
                        break
                    if type(dep[k]) == str:
                        if n and dep[k] == v:
                            match = False
                            break
                        if (not n) and dep[k] != v:
                            match = False
                            break
                    elif type(dep[k]) == set:
                        if n and v in dep[k]:
                            match = False
                            break
                        if (not n) and v not in dep[k]:
                            match = False
                            break
                    else:
                        raise RuntimeError()
                if match:
                    X[i, offset + j] = 1

    def get_all_names(self):
        return [",".join("%s%s%s" % (k, "!=" if n else "=", v) for k, n, v in feature) for feature in self.valid_features]

    def __len__(self):
        if not self.initialized:
            raise RuntimeError("Feature not initialized")
        else:
            # it always takes a single column
            return self.n_features


class FeatureSet:
    def __init__(self):
        self.features = list()

    def add_feature(self, feature):
        self.features.append(feature)

    def init_from_data(self, data):
        for feature in self.features:
            feature.init_from_data(data)

    def build_features(self, data, sparse=True):
        n_columns = sum(len(f) for f in self.features)
        if sparse:
            # TODO: check in celer what kind of sparse matrix it can use
            X = scipy.sparse.lil_matrix((len(data), n_columns))
        else:
            X = np.zeros((len(data), n_columns))

        offset = 0
        for feature in self.features:
            feature.build_features(X, data, offset)
            offset += len(feature)

        if sparse:
            X = scipy.sparse.csc_matrix(X)
        return X

    def feature_weights(self, weights, ignore_zeros=True):
        ret = dict()
        offset = 0
        for feature in self.features:
            names = feature.get_all_names()
            for i, name in enumerate(names):
                if not ignore_zeros or not np.isclose(weights[offset], 0):
                    ret[name] = (weights[offset], offset)
                offset += 1
        return ret

    def print_weights(self, weights, ignore_zeros=True):
        for n, (v, _) in self.feature_weights(weights, ignore_zeros=ignore_zeros).items():
            print("%s:\t%.4f" % (n, v))
