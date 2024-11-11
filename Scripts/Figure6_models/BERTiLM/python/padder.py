"""
Pads sequences to be max_seq_len tokens long.
REQUIRES: pad_token_id - an integer; the id number of the padding token
          max_seq_len  - an integer: the max number of tokens in a sample/sentence/example
MODIFIES: none
RETURNS:  data - the padded data
"""

## Code adapted from/inspired by STAPLER transforms.py

import torch

class Padder():
    def __init__(self, pad_token_id: int, max_seq_len: int) -> None:
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        # Pad the sequences to the max length
        data = torch.nn.functional.pad(data, (0, self.max_seq_len - data.shape[0]), "constant", self.pad_token_id)
        return data