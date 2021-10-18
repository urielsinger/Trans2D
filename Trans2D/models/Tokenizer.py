import torch
from transformers import BertTokenizer

class TensorTokenizer(BertTokenizer):
    """
    Tensor tokenizer, based on BertTokenizer only can handle tokenizing and decoding tensors as input and not text.
    """
    def __init__(self, vocab_file, do_lower_case=False, do_basic_tokenize=False, tokenize_chinese_chars=False, **kwargs):
        super().__init__(vocab_file, do_lower_case=do_lower_case, do_basic_tokenize=do_basic_tokenize, tokenize_chinese_chars=tokenize_chinese_chars, **kwargs)

    def tokenize(self, tensor: torch.Tensor):
        if tensor.dim() == 0:
            token = str(tensor.tolist())
            id = self.vocab[token] if token in self.vocab else self.vocab[self.unk_token]
            return torch.tensor(id, device=tensor.device)
        return torch.stack([self.tokenize(cur_tensor) for cur_tensor in tensor], dim=0)

    def decode(self, tensor: torch.Tensor):
        if tensor.dim() == 0:
            id = tensor.tolist()
            token = self.ids_to_tokens[id] if id in self.ids_to_tokens else self.unk_token
            return torch.tensor(token, device=tensor.device)
        return torch.stack([self.decode(cur_tensor) for cur_tensor in tensor], dim=0)
