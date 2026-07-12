"""Stage 5 head-banked token-major KV cache reference model."""


class MultiHeadKVCacheError(ValueError):
    pass


def multi_head_linear_address(head, token_index, dimension, n_head, d_head, max_seq_len):
    if head < 0 or head >= n_head:
        raise MultiHeadKVCacheError("head index out of range")
    if token_index < 0 or token_index >= max_seq_len:
        raise MultiHeadKVCacheError("token index out of range")
    if dimension < 0 or dimension >= d_head:
        raise MultiHeadKVCacheError("dimension out of range")
    return ((head * max_seq_len) + token_index) * d_head + dimension


class MultiHeadKVCacheReference(object):
    def __init__(self, n_head, d_head, max_seq_len):
        if n_head <= 0 or d_head <= 0 or max_seq_len <= 0:
            raise ValueError("n_head, d_head, and max_seq_len must be positive")
        self.n_head = n_head
        self.d_head = d_head
        self.max_seq_len = max_seq_len
        self.k_cache = [[] for _ in range(n_head)]
        self.v_cache = [[] for _ in range(n_head)]
        self._append_k = None
        self._append_v = None
        self._append_token = None
        self._expected_head = 0
        self._expected_dim = 0
        self._head_complete = [False for _ in range(n_head)]

    @property
    def valid_seq_len(self):
        return len(self.k_cache[0])

    @property
    def append_incomplete(self):
        return self._append_k is not None and not self.provisional_valid

    @property
    def provisional_valid(self):
        return all(self._head_complete)

    @property
    def provisional_head_valid(self):
        return list(self._head_complete)

    @property
    def provisional_token_index(self):
        return self._append_token

    @property
    def cache_full(self):
        return self.valid_seq_len >= self.max_seq_len

    def reset(self):
        self.k_cache = [[] for _ in range(self.n_head)]
        self.v_cache = [[] for _ in range(self.n_head)]
        self.abort_provisional()

    def _check_shared_lengths(self):
        lengths = [len(rows) for rows in self.k_cache] + [len(rows) for rows in self.v_cache]
        if any(length != lengths[0] for length in lengths):
            raise MultiHeadKVCacheError("per-head committed lengths diverged")

    def read(self, head, token_index, dimension, include_provisional=False):
        multi_head_linear_address(head, token_index, dimension, self.n_head, self.d_head, self.max_seq_len)
        if token_index < self.valid_seq_len:
            return self.k_cache[head][token_index][dimension], self.v_cache[head][token_index][dimension]
        if (
            include_provisional
            and self._append_k is not None
            and self._head_complete[head]
            and token_index == self.valid_seq_len
        ):
            return self._append_k[head][dimension], self._append_v[head][dimension]
        raise MultiHeadKVCacheError("read beyond visible sequence length")

    def append_dim(
        self,
        head,
        token_index,
        dimension,
        k_fp16,
        v_fp16,
        last_dim,
        last_head,
        complete,
    ):
        if self.cache_full:
            raise MultiHeadKVCacheError("cache full")
        if self.provisional_valid:
            raise MultiHeadKVCacheError("provisional token already complete")
        multi_head_linear_address(head, token_index, dimension, self.n_head, self.d_head, self.max_seq_len)
        if token_index != self.valid_seq_len:
            self.abort_provisional()
            raise MultiHeadKVCacheError("append token would overwrite or skip valid token")
        if head != self._expected_head or dimension != self._expected_dim:
            self.abort_provisional()
            raise MultiHeadKVCacheError("append head/dimension order violation")

        final_dim = dimension == self.d_head - 1
        final_head = head == self.n_head - 1
        if final_dim and final_head:
            if not last_dim or not last_head or not complete:
                self.abort_provisional()
                raise MultiHeadKVCacheError("final head/dimension must complete provisional token")
        elif final_dim:
            if not last_dim or last_head or complete:
                self.abort_provisional()
                raise MultiHeadKVCacheError("non-final head cannot complete provisional token")
        elif last_dim or last_head or complete:
            self.abort_provisional()
            raise MultiHeadKVCacheError("non-final dimension cannot complete provisional token")

        if self._append_k is None:
            self._append_k = [[0 for _ in range(self.d_head)] for _ in range(self.n_head)]
            self._append_v = [[0 for _ in range(self.d_head)] for _ in range(self.n_head)]
            self._append_token = token_index

        self._append_k[head][dimension] = k_fp16 & 0xFFFF
        self._append_v[head][dimension] = v_fp16 & 0xFFFF

        if final_dim:
            self._head_complete[head] = True
            self._expected_dim = 0
            if final_head:
                self._expected_head = 0
            else:
                self._expected_head += 1
        else:
            self._expected_dim += 1

    def append_provisional_token(self, k_heads, v_heads):
        if len(k_heads) != self.n_head or len(v_heads) != self.n_head:
            raise ValueError("K/V head count mismatch")
        token_index = self.valid_seq_len
        for head in range(self.n_head):
            if len(k_heads[head]) != self.d_head or len(v_heads[head]) != self.d_head:
                raise ValueError("K/V row dimension mismatch")
            for dim in range(self.d_head):
                final_dim = dim == self.d_head - 1
                final_head = head == self.n_head - 1 and final_dim
                self.append_dim(
                    head,
                    token_index,
                    dim,
                    k_heads[head][dim],
                    v_heads[head][dim],
                    final_dim,
                    final_head,
                    final_head,
                )

    def commit_provisional(self):
        if not self.provisional_valid:
            raise MultiHeadKVCacheError("no complete all-head provisional token to commit")
        if self._append_token != self.valid_seq_len:
            self.abort_provisional()
            raise MultiHeadKVCacheError("provisional token index mismatch")
        if self.cache_full:
            self.abort_provisional()
            raise MultiHeadKVCacheError("cache full")
        for head in range(self.n_head):
            self.k_cache[head].append(list(self._append_k[head]))
            self.v_cache[head].append(list(self._append_v[head]))
        self.abort_provisional()
        self._check_shared_lengths()

    def abort_provisional(self):
        self._append_k = None
        self._append_v = None
        self._append_token = None
        self._expected_head = 0
        self._expected_dim = 0
        self._head_complete = [False for _ in range(self.n_head)]

    def append_token(self, k_heads, v_heads):
        self.append_provisional_token(k_heads, v_heads)
        self.commit_provisional()

    def visible_snapshot(self, head, include_provisional=False):
        if head < 0 or head >= self.n_head:
            raise MultiHeadKVCacheError("head index out of range")
        k_rows = [list(row) for row in self.k_cache[head]]
        v_rows = [list(row) for row in self.v_cache[head]]
        if include_provisional:
            if self._append_k is None or not self._head_complete[head]:
                raise MultiHeadKVCacheError("provisional token is incomplete for requested head")
            k_rows.append(list(self._append_k[head]))
            v_rows.append(list(self._append_v[head]))
        return k_rows, v_rows

    def snapshot(self):
        return (
            [[list(row) for row in head_rows] for head_rows in self.k_cache],
            [[list(row) for row in head_rows] for head_rows in self.v_cache],
        )
