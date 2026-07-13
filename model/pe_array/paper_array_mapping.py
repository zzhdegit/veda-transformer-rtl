"""Mapping helpers for the Stage 8 paper-structured PE array."""

ARRAY_ROWS = 8
ARRAY_COLS = 8
ARRAY_GROUPS = 2
PE_CELLS = ARRAY_ROWS * ARRAY_COLS * ARRAY_GROUPS
GROUP_CELLS = ARRAY_ROWS * ARRAY_COLS

MODE_INNER_PRODUCT = 1
MODE_OUTER_PRODUCT = 2

PE_TYPE_A = "A"
PE_TYPE_B = "B"


def pe_type_for_column(column):
    if column < 0 or column >= ARRAY_COLS:
        raise ValueError("column out of range")
    return PE_TYPE_A if (column % 2) == 0 else PE_TYPE_B


def cell_linear_index(group, row, column):
    if group < 0 or group >= ARRAY_GROUPS:
        raise ValueError("group out of range")
    if row < 0 or row >= ARRAY_ROWS:
        raise ValueError("row out of range")
    if column < 0 or column >= ARRAY_COLS:
        raise ValueError("column out of range")
    return group * GROUP_CELLS + row * ARRAY_COLS + column


def decode_cell_index(index):
    if index < 0 or index >= PE_CELLS:
        raise ValueError("cell index out of range")
    group = index // GROUP_CELLS
    rem = index % GROUP_CELLS
    row = rem // ARRAY_COLS
    column = rem % ARRAY_COLS
    return group, row, column


def active_mask_for_count(count, width):
    if count < 0 or count > width:
        raise ValueError("active count out of range")
    return (1 << count) - 1 if count else 0


def mask_bit(mask, index):
    return bool((mask >> index) & 1)


def default_group_mask():
    return active_mask_for_count(ARRAY_GROUPS, ARRAY_GROUPS)


def default_row_mask():
    return active_mask_for_count(ARRAY_ROWS, ARRAY_ROWS)


def default_column_mask():
    return active_mask_for_count(ARRAY_COLS, ARRAY_COLS)


def tile_cell_active(vector_length, tile_base, group, row, column, group_mask=None, row_mask=None, column_mask=None):
    if group_mask is None:
        group_mask = default_group_mask()
    if row_mask is None:
        row_mask = default_row_mask()
    if column_mask is None:
        column_mask = default_column_mask()
    cell_offset = cell_linear_index(group, row, column)
    dim = tile_base + cell_offset
    return (
        dim < vector_length
        and mask_bit(group_mask, group)
        and mask_bit(row_mask, row)
        and mask_bit(column_mask, column)
    )


def iter_cells():
    for group in range(ARRAY_GROUPS):
        for row in range(ARRAY_ROWS):
            for column in range(ARRAY_COLS):
                yield group, row, column


def tile_bases(length, tile_width=PE_CELLS):
    if length < 0:
        raise ValueError("length must be non-negative")
    if tile_width <= 0:
        raise ValueError("tile width must be positive")
    base = 0
    while base < length:
        yield base
        base += tile_width

