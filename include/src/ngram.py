#!/usr/bin/env python3
"""N-gram Counting and Generation (Basic) — Online Judge submission framework.

A language model can be nothing more than a big counting table: "after this
context of N-1 tokens, which token followed, and how many times?" In this
problem you build such a table from a training corpus (COUNT queries) and
then use it to generate new text by weighted random sampling (GEN queries).

Read the problem statement (Input/Output format) carefully before you start.

Only fill in the three TODO functions below. The I/O parsing and main() are
already provided — do NOT modify them.

Run locally:  python3 ngram.py < 1.in
"""
import random
import sys
from collections import Counter, defaultdict


START = "<START>"
END = "<END>"


def build_ngram_model(tokens, n):
    padded = ["<START>"] * (n - 1) + tokens + ["<END>"]
    model = defaultdict(Counter)
    for i in range(0,len(padded) - n + 1, 1):
        context = tuple(padded[i: i + n - 1])
        target = padded[i + n - 1] 
        model[context][target] += 1
    return model

def count_ngram(model, context, w):
    if context not in model:
        return 0
    Counter = model[context]
    if w not in Counter:
        return 0
    return Counter[w]


def generate_text(model, n, length, seed):
    """TODO 3: Generate up to `length` tokens by weighted random sampling.

    Args:
        model:  the dict returned by build_ngram_model.
        n:      the N the model was built with.
        length: maximum number of tokens to generate (>= 1).
        seed:   integer seed for the random number generator.

    Returns:
        The list of generated tokens, NOT including START or END. It may be
        empty (if END is sampled on the very first step).

    The sampling rule below is part of the problem statement — follow it
    exactly, otherwise your output will not be reproducible:

      1. Create a private generator:  rng = random.Random(seed).
         (Do NOT touch the global random module state.)
      2. Start from context = (START,) * (n - 1).
      3. At each step, look up the current context in the model.
         - The candidate list is ALL successor tokens of this context,
           sorted by Python's default string ordering (i.e. by Unicode code
           points; sorted(counter) does this).
         - The weight of each candidate is its integer count, in the same
           order as the sorted candidate list.
         - Draw the next token with exactly:
               rng.choices(candidates, weights=counts, k=1)[0]
      4. If the drawn token is END, stop immediately (END is not emitted).
      5. Otherwise append the token to the result, slide the window forward
         (the new context is the last n-1 tokens of old context + new token),
         and continue.
      6. Stop after `length` emitted tokens at the latest. If the current
         context is absent from the model (should not normally happen),
         stop as well.

    Hints:
      - random.Random / random.choices documentation:
        https://docs.python.org/3/library/random.html
      - Debug with a deterministic corpus first: with tokens = ["x", "y", "z"]
        and n = 2 every context has exactly one successor, so any seed must
        generate exactly ["x", "y", "z"] (then stop at END).
      - Same seed + same model must always give the same output. If yours
        doesn't, check the sorting of the candidate list — dict iteration
        order is NOT what the rule asks for.
    """
    rng = random.Random(seed)
    context = (START,) * (n - 1)
    text = []
    for i in range(length):
        if context not in model:
            break
        Counter = model[context]
        candidates = sorted(Counter.keys())
        counts = [Counter[c] for c in candidates]
        next_token = rng.choices(candidates, weights=counts, k=1)[0]
        if next_token == END :
            break
        if n == 1:
            context = ()
        else:
            context = (context + (next_token,))[-(n - 1):]
        text.append(next_token)
    return text


# ---------------- provided I/O below — do not modify ----------------

def main():
    readline = sys.stdin.readline

    first = readline().split()
    n, q = int(first[0]), int(first[1])
    tokens = readline().split()  # may be an empty corpus (empty line)

    model = build_ngram_model(tokens, n)

    out_lines = []
    for _ in range(q):
        parts = readline().split()
        if parts[0] == "COUNT":
            # COUNT c1 ... c_{n-1} w   (COUNT w when n == 1)
            context = tuple(parts[1:n])
            w = parts[n]
            out_lines.append(str(count_ngram(model, context, w)))
        else:  # GEN L seed
            length, seed = int(parts[1]), int(parts[2])
            out_lines.append(" ".join(generate_text(model, n, length, seed)))

    sys.stdout.write("\n".join(out_lines) + "\n")


if __name__ == "__main__":
    main()
