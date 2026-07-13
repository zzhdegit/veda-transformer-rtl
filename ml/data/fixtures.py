"""Small built-in text fixtures for offline ML-M2 tests."""

SMOKE_STORIES = [
    "Lina had a red kite. The kite flew high, and Lina smiled.",
    "Tom found a little box under the bed. Inside was a blue button.",
    "Mia and Sam made soup. They shared it with Dad and felt proud.",
    "The small dog sat by the door. It barked once and wagged its tail.",
    "A yellow bird sang in the tree. The children listened quietly.",
    "Nora lost her toy train. She looked under the chair and found it.",
    "Ben planted one seed. Rain came, sun came, and a green leaf grew.",
    "The moon was bright. Ella counted stars before she went to sleep.",
]

SMOKE_TEST_PROMPTS = [
    "Lina had",
    "The small dog",
    "A yellow bird",
]


def fixture_text() -> str:
    return "\n".join(SMOKE_STORIES)

