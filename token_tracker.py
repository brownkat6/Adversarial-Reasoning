import torch
from typing import Dict, List, Optional, Tuple, Union

class TokenTracker:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.total_tokens = 0
        self.messages_seen = set()  # Track unique messages to avoid double counting
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in a piece of text."""
        return len(self.tokenizer.encode(text))
    
    def add_message(self, message: str) -> None:
        """Add a message and count its tokens if not seen before."""
        if message not in self.messages_seen:
            self.total_tokens += self.count_tokens(message)
            self.messages_seen.add(message)
    
    def add_messages(self, messages: List[str]) -> None:
        """Add multiple messages and count their tokens."""
        for message in messages:
            self.add_message(message)
    
    def get_total_tokens(self) -> int:
        """Get the total number of tokens counted so far."""
        return self.total_tokens
    
    def reset(self) -> None:
        """Reset the token counter and message tracking."""
        self.total_tokens = 0
        self.messages_seen.clear() 