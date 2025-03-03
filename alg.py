import torch
from token_tracker import TokenTracker

class GWW():
    def __init__(self, num_prompts):
        self.num_prompts = num_prompts
        self.content = []
        self.token_tracker = None  # Will be set by GWW_dfs_min
    
    def add_prompt(self, init_prompt, losses, messages):
        self.content.append({"init_prompt": init_prompt, 
                           "mean_loss": torch.mean(losses),
                           "losses": losses,
                           "messages": messages})
        
        # Track tokens if token_tracker is available
        #if self.token_tracker is not None:
        #    self.token_tracker.add_message(init_prompt)
        #    self.token_tracker.add_messages(messages)
        
        # self.sort_prompts()
        
        if len(self.content) > self.num_prompts:
            self.content = self.content[:self.num_prompts]
    
    def sort_prompts(self):
        self.content = sorted(self.content, key=lambda x: x["mean_loss"])
        
    def get_prompt(self):
        return self.content[0]["init_prompt"], self.content[0]["losses"], self.content[0]["messages"]
    
    def get_total_tokens(self):
        """Get total tokens used if tracking is enabled."""
        return self.token_tracker.get_total_tokens() if self.token_tracker is not None else 0


class GWW_dfs_min(GWW):
    def __init__(self, model_lam, tokenizer_lam, init_msg, num_iters, num_branches, memory, K, device):
        super().__init__(memory)
        self.token_tracker = TokenTracker(tokenizer_lam)
        self.model = model_lam
        self.tokenizer = tokenizer_lam
        self.init_msg = init_msg
        self.num_iters = num_iters
        self.num_branches = num_branches
        self.K = K
        self.device = device
    
    def sort_prompts(self):
        # self.content = sorted(self.content, key=lambda x: torch.mean(x["losses"]) - torch.sqrt(torch.var(x["losses"])))
        self.content = sorted(self.content, key=lambda x: torch.min(x["losses"]))
        
    def get_prompt(self):
        first_content = self.content[0]
        self.content.pop(0)
        total_tokens = self.get_total_tokens()
        
        return first_content["init_prompt"], first_content["losses"], first_content["messages"], total_tokens