import torch
from typing import Dict, List, Optional, Tuple, Union

class TokenTracker:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.total_tokens = 0
        self.messages_seen = set()
        
    def add_generate_output(self, output_ids):
        """Track only output tokens from a model.generate() call"""
        # Count output tokens (excluding padding)
        if isinstance(output_ids, torch.Tensor):
            output_ids = output_ids.cpu().tolist()
        if isinstance(output_ids[0], list):
            for seq in output_ids:
                self.total_tokens += sum(1 for id in seq if id != self.tokenizer.pad_token_id)
        else:
            self.total_tokens += sum(1 for id in output_ids if id != self.tokenizer.pad_token_id)
    
    def get_total_tokens(self):
        """Get total tokens tracked"""
        return self.total_tokens