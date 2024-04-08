import collections


CLOSED_POS_TAGS = {
    "AUX",
    "ADP",
    "PRON",
    "DET",
    "SCONJ",
    "CCONJ",
    "PART"
}

# le terme "similar" n'est pas du tout adapté,
# mais c'est en gros les POS tags qu'on veut merger ensemble
# pour ajouter de nouvelles features
# càd que plutôt qu'avoir seulement "POS=NOUN",
# on veut aussi avoir "POS=NOUN|PROPN"
SIMILAR_POS_TAGS = [
    # use list to fix order, probably unecessary,
    # but just in case
    [
        "NOUN", "PROPN"
    ],
    [
        "NOUN", "PROPN", "PRON"
    ],
    [
        "AUX", "VERB"
    ],
    [
        "DET", "NUM"
    ]
]


def read(path):
    data = list()
    with open(path) as istream:
        need_new = True
        for line in istream:
            line = line.strip()
            if len(line) == 0:
                need_new = True
                continue
            if line[0] == "#":
                continue

            line = line.split("\t")
            if line[0].find(".") >= 0 or line[0].find("-") >= 0:
                continue

            if need_new:
                data.append(list())
                need_new = False

            if line[5] != "_":
                feats = {
                    k: v for k, v in [m.split("=") for m in line[5].split("|")]
                }
            else:
                feats = dict()

            data[-1].append({
                "idx": len(data[-1]) + 1,
                "lemma": line[2],
                "upos": line[3],
                "head": int(line[6]),
                "dep.rel": line[7],
                "feats": feats
            })

    return data


# split head rel 1:2@3 in two different case:
# _shallow: 1:2
# _deep: 1:2@3
def do_split_head_rel(head_rel, split_head_rel=True):
    if split_head_rel:
        if head_rel.find("@") >= 0:
            ret = {
                "_synt": head_rel.split("@")[0],
                "_deep": head_rel
            }
        else:
            ret = {
                "_synt": head_rel
            }
    else:
        ret = {
            "": head_rel
        }
    return ret


def extract_dependencies(data, split_head_rel=True, add_closed_pos_tags_lemma=False, add_similar_pos_tags=False):
    dependencies = list()
    for sentence in data:
        for mod in sentence:
            # not sure if this is a good idea...
            if mod["head"] == 0:
                continue

            mod_idx = mod["idx"]
            mod_head = mod["head"]
            mod_children = [w for w in sentence if w["head"] == mod_idx]
            head = sentence[mod_head - 1]
            head_children = [w for w in sentence if w["head"] == mod_head and w["idx"] != mod_idx]

            dep = {
                "dep.lemma": mod["lemma"],
                "dep.upos": mod["upos"],
                "gov.lemma": head["lemma"],
                "gov.upos": head["upos"],
                "gov.position": "before_dep" if head["idx"] < mod_idx else "after_dep",
            }
            
            if head["head"] != 0:
                dep['grandparent.position'] = "before_gov" if sentence[head["head"] - 1]["head"] < head["idx"] else "after_gov"

            if add_closed_pos_tags_lemma:
                if mod["upos"] in CLOSED_POS_TAGS:
                    dep["dep.lemma_" + mod["upos"]] = mod["lemma"]
                if head["upos"] in CLOSED_POS_TAGS:
                    dep["gov.lemma_" + head["upos"]] = head["lemma"]
            if add_similar_pos_tags:
                for tags in SIMILAR_POS_TAGS:
                    tags_str = "|".join(tags)
                    if mod["upos"] in tags:
                        dep["dep.in_upos"] = tags_str
                    if head["upos"] in tags:
                        dep["gov.in_upos"] = tags_str

            for k, v in do_split_head_rel(mod["dep.rel"], split_head_rel=split_head_rel).items():
                dep["dep.rel" + k] = v
            for k, v in do_split_head_rel(head["dep.rel"], split_head_rel=split_head_rel).items():
                dep["gov.rel" + k] = v

            for k, v in mod["feats"].items():
                dep["dep.%s" % k] = v
            for k, v in head["feats"].items():
                dep["gov.%s" % k] = v

            if head["head"] != 0:
                # dep["head_is_root"] = "false"
                gp = sentence[head["head"] - 1]
                dep["grandparent.lemma"] = gp["lemma"]
                dep["grandparent.upos"] = gp["upos"]
                if add_closed_pos_tags_lemma:
                    if gp["upos"] in CLOSED_POS_TAGS:
                        dep["grandparent.lemma_" + gp["upos"]] = gp["lemma"]
                for k, v in gp["feats"].items():
                    dep["grandparent.%s" % k] = v
            else:
                # This information is in gp_rel
                # dep["head_is_root"] = "true"
                pass

            # children of the modifier
            dep["grandchildren.lemmas"] = set(w["lemma"] for w in mod_children)
            mod_children_upos = set(w["upos"] for w in mod_children)
            dep["grandchildren.upos"] = mod_children_upos
            if add_closed_pos_tags_lemma:
                for upos in CLOSED_POS_TAGS:
                    if upos in mod_children_upos:
                        lemmas = set(w["lemma"] for w in mod_children if w["upos"] == upos)
                        assert len(lemmas) > 0
                        dep["grandchildren.lemmas_" + upos] = lemmas
            if add_similar_pos_tags:
                for tags in SIMILAR_POS_TAGS:
                    if len(mod_children_upos.intersection(tags)) > 1:
                        tags_str = "|".join(tags)
                        dep["grandchildren.in_upos"] = tags_str
            mod_children_rels = collections.defaultdict(lambda: set())
            for w in mod_children:
                for k, v in do_split_head_rel(w["dep.rel"], split_head_rel=split_head_rel).items():
                    mod_children_rels[k].add(v)
            for k, v in mod_children_rels.items():
                dep["grandchildren.rels" + k] = v
            feats = collections.defaultdict(lambda: set())
            for child in mod_children:
                for k, v in child["feats"].items():
                    feats[k].add(v)
            for k, v in feats.items():
                dep["grandchildren.%s" % k] = v

            # children of the head
            dep["siblings.lemmas"] = set(w["lemma"] for w in head_children if w["idx"] != mod_idx)
            head_children_upos = set(w["upos"] for w in head_children if w["idx"] != mod_idx)
            dep["siblings.upos"] = head_children_upos
            if add_closed_pos_tags_lemma:
                for upos in CLOSED_POS_TAGS:
                    if upos in head_children_upos:
                        lemmas = set(w["lemma"] for w in head_children if w["idx"] != mod_idx and w["upos"] == upos)
                        assert len(lemmas) > 0
                        dep["siblings.lemmas_" + upos] = lemmas
            if add_similar_pos_tags:
                for tags in SIMILAR_POS_TAGS:
                    if len(head_children_upos.intersection(tags)) > 1:
                        tags_str = "|".join(tags)
                        dep["siblings.in_upos"] = tags_str
            head_children_rels = collections.defaultdict(lambda: set())
            for w in head_children:
                for k, v in do_split_head_rel(w["dep.rel"], split_head_rel=split_head_rel).items():
                    head_children_rels[k].add(v)
            for k, v in head_children_rels.items():
                dep["siblings.rels" + k] = v
            feats = collections.defaultdict(lambda: set())
            for child in head_children:
                if child["idx"] != mod_idx:
                    for k, v in child["feats"].items():
                        feats[k].add(v)
            for k, v in feats.items():
                dep["siblings.%s" % k] = v

            dependencies.append(dep)

    return dependencies
