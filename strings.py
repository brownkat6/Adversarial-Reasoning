# Adversarial-Reasoning/strings.py
import torch
import numpy as np
from utils import batch_generate_responses
from convs import get_conv_feedbacker
import json
import re
import concurrent.futures
import time

def get_attacks_string(model, tokenizer, conv, batch):
    """Generate attack strings using local model."""
    messages = []
    generated = 0
    num_tokens_total = 0
    while generated < batch:
        try:
            outputs, num_tokens = batch_generate_responses(
                model,
                tokenizer,
                [conv.get_prompt() for _ in range(batch - generated)]
            )
            for i in range(len(outputs)):
                if "</think>" in outputs[i]:
                    outputs[i] = outputs[i].split("</think>")[1]
            num_tokens_total += num_tokens
            
            for output in outputs:
                result = extract_strings(output)
                if result:
                    messages.append(result)
                    generated += 1
        except Exception as e:
            print(f"Error generating attacks: {e}")
            continue
        
        print(f"Generated {generated}/{batch}")
    
    return messages, num_tokens_total

def get_feedbacks(model, tokenizer, name, goal, target, messages, idx, divs, num_branches):
    """Get feedback using local model."""
    convs = [
        get_conv_feedbacker(
            name, goal, target, 
            gen_string_feedbacker_rand(messages, idx, divs), 
            len(messages)//divs
        ) 
        for _ in range(num_branches)
    ]
    
    final_feedbacks = []
    convs_to_process = convs
    num_tokens_total = 0
    
    while convs_to_process:
        try:
            outputs, num_tokens = batch_generate_responses(
                model,
                tokenizer,
                [conv.get_prompt() for conv in convs_to_process]
            )
            for i in range(len(outputs)):
                if "</think>" in outputs[i]:
                    outputs[i] = outputs[i].split("</think>")[1]
            num_tokens_total += num_tokens
            remaining_convs = []
            for conv, output in zip(convs_to_process, outputs):
                output = output.replace("\\", "")
                feedback = extract_final_feedback(output)
                
                if feedback is not None:
                    final_feedbacks.append(feedback)
                else:
                    remaining_convs.append(conv)
            
            convs_to_process = remaining_convs
            
        except Exception as e:
            print(f"Error getting feedback: {e}")
            break
    
    return final_feedbacks, num_tokens_total

def get_new_prompts(model, tokenizer, convs):
    """Get new prompts using local model."""
    new_prompts = []
    convs_to_process = convs
    num_tokens_total = 0
    while convs_to_process:
        try:
            outputs, num_tokens = batch_generate_responses(
                model,
                tokenizer,
                [conv.get_prompt() for conv in convs_to_process]
            )
            for i in range(len(outputs)):
                if "</think>" in outputs[i]:
                    outputs[i] = outputs[i].split("</think>")[1]
            num_tokens_total += num_tokens

            remaining_convs = []
            for conv, output in zip(convs_to_process, outputs):
                output = output.replace("\\", "")
                prompt = extract_new_prompt(output)
                
                if prompt is not None:
                    new_prompts.append(prompt)
                else:
                    remaining_convs.append(conv)
            
            convs_to_process = remaining_convs
            
        except Exception as e:
            print(f"Error getting new prompts: {e}")
            break
    
    return new_prompts, num_tokens_total


def gen_string_feedbacker_rand(messages, idx, div =8):
    assert len(messages) == len(idx) 
    l = len(idx)
        
    idx = idx.reshape(-1, div)[torch.arange(l//div), torch.randint(div, size = (l//div, ))]
    string = f""
    for i in range(len(idx)):
        string += f"Prompt_{i+1}:\n'{messages[idx[i]]}'\n\n"
    
    return string
    

def gen_string_optimizer(variable, feedback):
    string = f"Variable:\n'{variable}'\n\nFeedback:\n{feedback}"
    
    return string


def extract_strings(text):
    # Split the string into lines
    lines = text.split('\n')
    key_text = '"Prompt P": '
    for line in lines:
        # Check if key_text is in the line
        if key_text in line:
            # Find the index of key_text and get everything after it
            feedback_index = line.find(key_text)
            prompt = line[feedback_index + len(key_text):].strip()
            # Remove trailing brace if present
            if prompt.endswith('}'):
                prompt = prompt[:-1]
            try:
                # Safely evaluate the string to unescape it
                prompt = ast.literal_eval(prompt)
            except Exception as e:
                # Handle cases where evaluation fails
                pass
            return prompt
    return None


def extract_final_feedback(text):
    # Split the string into lines
    lines = text.split('\n')
    text = f""""Final_feedback": """
    for line in lines:
        # Check if 'final_feedback' is in the line
        if text in line:
            # Find the index of 'final_feedback' and get everything after it
            feedback_index = line.find(text)
            return line[feedback_index + len(text):].strip()
    
    return None


def extract_new_prompt(text):
    # Split the string into lines
    lines = text.split('\n')
    text = f""""Improved_variable": """
    for line in lines:
        # Check if 'final_feedback' is in the line
        if text in line:
            # Find the index of 'final_feedback' and get everything after it
            feedback_index = line.find(text)
            return line[feedback_index + len(text):].strip()

    return None