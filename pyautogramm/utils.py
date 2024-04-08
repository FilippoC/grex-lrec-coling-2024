class Dict:
    def __init__(self, values):
        values = set(values)
        self._id_to_str = list()
        self._str_to_id = dict()

        for v in values:
            self._str_to_id[v] = len(self._id_to_str)
            self._id_to_str.append(v)

    def str_to_id(self, v):
        return self._str_to_id[v]

    def id_to_str(self, v):
        return self._id_to_str[v]

    def __len__(self):
        return len(self._id_to_str)