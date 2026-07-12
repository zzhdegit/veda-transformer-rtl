"""Stage 4 token-major KV cache reference model."""


class KVCacheError(ValueError):
    pass


def linear_address(token_index, dimension, d_head, max_seq_len):
    if token_index < 0 or token_index >= max_seq_len:
        raise KVCacheError("token index out of range")
    if dimension < 0 or dimension >= d_head:
        raise KVCacheError("dimension out of range")
    return token_index * d_head + dimension


class KVCacheReference(object):
    def __init__(self, d_head, max_seq_len):
        if d_head <= 0 or max_seq_len <= 0:
            raise ValueError("d_head and max_seq_len must be positive")
        self.d_head = d_head
        self.max_seq_len = max_seq_len
        self.k_cache = []
        self.v_cache = []
        self._append_k = None
        self._append_v = None
        self._append_token = None
        self._expected_dim = 0
        self._provisional_complete = False

    @property
    def valid_seq_len(self):
        return len(self.k_cache)

    @property
    def append_incomplete(self):
        return self._append_k is not None and not self._provisional_complete

    @property
    def provisional_valid(self):
        return self._provisional_complete

    @property
    def provisional_token_index(self):
        return self._append_token

    @property
    def cache_full(self):
        return self.valid_seq_len >= self.max_seq_len

    def reset(self):
        self.k_cache = []
        self.v_cache = []
        self.abort_provisional()

    def read(self, token_index, dimension, include_provisional=False):
        linear_address(token_index, dimension, self.d_head, self.max_seq_len)
        if token_index < self.valid_seq_len:
            return self.k_cache[token_index][dimension], self.v_cache[token_index][dimension]
        if (
            include_provisional
            and self._provisional_complete
            and token_index == self.valid_seq_len
        ):
            return self._append_k[dimension], self._append_v[dimension]
        raise KVCacheError("read beyond visible sequence length")

    def append_dim(self, token_index, dimension, k_fp16, v_fp16, last_dim, complete):
        if self.cache_full:
            raise KVCacheError("cache full")
        if self._provisional_complete:
            raise KVCacheError("provisional token already complete")
        linear_address(token_index, dimension, self.d_head, self.max_seq_len)
        if token_index != self.valid_seq_len:
            self.abort_provisional()
            raise KVCacheError("append token would overwrite or skip valid token")
        if dimension != self._expected_dim:
            self.abort_provisional()
            raise KVCacheError("append dimension order violation")

        final_dim = dimension == self.d_head - 1
        if final_dim:
            if not last_dim or not complete:
                self.abort_provisional()
                raise KVCacheError("final append dimension must complete provisional token")
        elif last_dim or complete:
            self.abort_provisional()
            raise KVCacheError("non-final append dimension cannot complete provisional token")

        if self._append_k is None:
            self._append_k = [0 for _ in range(self.d_head)]
            self._append_v = [0 for _ in range(self.d_head)]
            self._append_token = token_index

        self._append_k[dimension] = k_fp16 & 0xFFFF
        self._append_v[dimension] = v_fp16 & 0xFFFF

        if final_dim:
            self._provisional_complete = True
            self._expected_dim = 0
        else:
            self._expected_dim += 1

    def append_provisional_token(self, k_row, v_row):
        if len(k_row) != self.d_head or len(v_row) != self.d_head:
            raise ValueError("K/V row dimension mismatch")
        token_index = self.valid_seq_len
        for dim in range(self.d_head):
            self.append_dim(
                token_index,
                dim,
                k_row[dim],
                v_row[dim],
                dim == self.d_head - 1,
                dim == self.d_head - 1,
            )

    def commit_provisional(self):
        if not self._provisional_complete:
            raise KVCacheError("no complete provisional token to commit")
        if self._append_token != self.valid_seq_len:
            self.abort_provisional()
            raise KVCacheError("provisional token index mismatch")
        if self.cache_full:
            self.abort_provisional()
            raise KVCacheError("cache full")
        self.k_cache.append(list(self._append_k))
        self.v_cache.append(list(self._append_v))
        self.abort_provisional()

    def abort_provisional(self):
        self._append_k = None
        self._append_v = None
        self._append_token = None
        self._expected_dim = 0
        self._provisional_complete = False

    def append_token(self, k_row, v_row):
        self.append_provisional_token(k_row, v_row)
        self.commit_provisional()

    def visible_snapshot(self, include_provisional=False):
        k_rows = [list(row) for row in self.k_cache]
        v_rows = [list(row) for row in self.v_cache]
        if include_provisional:
            if not self._provisional_complete:
                raise KVCacheError("provisional token is incomplete")
            k_rows.append(list(self._append_k))
            v_rows.append(list(self._append_v))
        return k_rows, v_rows

    def snapshot(self):
        return self.visible_snapshot(include_provisional=False)
