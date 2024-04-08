# Grex: automatic grammar extraction from treebank

Grex is a tool for automatic grammar rule extraction from treebanks created by Santiago Herrera, Caio Corro and Sylvain Kahane. A full description of the method can be found in the paper:
https://arxiv.org/abs/2403.17534

If you use this software, please cite the following work:
<pre>@inproceedings{herrera2024grex,
    title = "Sparse Logistic Regression with High-order Features for Automatic Grammar Rule Extraction from Treebanks",
    author = "Herrera, Santiago and Corro, Caio and Kahane, Sylvain",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    url = "https://arxiv.org/abs/2403.17534",
}</pre>


## Usage

You first need to install the Skglm library: https://contrib.scikit-learn.org/skglm/auto_examples/index.html

At the moment, there are only two scripts:

- ``autogramm_agreement.py``: search rules of morphological agreements (e.g. when is there a number agreement between a word and its head?)
- ``autogramm_activation.py``: search rules that activate a feature (e.g. when is the subject after the verb?)

Examples are given in ``run.sh``.
The extracted rules are exported in json format.

## Common arguments

- ``--treebank``: directory where treebanks are stored
- ``--treebank-filter``: list of treebanks to use (partial match will be used)
- ``--feature-filter``: features to remove (must be lowercased + partial match will be used)
- ``--dep-filter``: filter the dataset. For example, ``--dep-filter=head_upos=VERB,mod_upos=NOUN`` will check only dependencies between a VERB and a NOUN
- ``--json``: output file
- ``--error``: error file

## autogramm_agreement.py

Search rules that forces two features to agree.

- ``--feature1``: first feature
- ``--feature2``: second feature

## autogramm_activation.py

- ``--feature-name``: name of the feature
- ``--feature-value``: value of the feature we are interested in
