import torch
import argparse

# Allow loading pickled argparse.Namespace objects (needed by Fairseq models)
torch.serialization.add_safe_globals([argparse.Namespace])

from ai4bharat.transliteration import XlitEngine

# Initialize the transliteration engine for Hindi
e = XlitEngine("hi", beam_width=10, rescore=True)

# Transliterate the word "namasthe" and get top 5 candidates
out = e.translit_word("namasthe", topk=5)

# Print the results
print(out)
