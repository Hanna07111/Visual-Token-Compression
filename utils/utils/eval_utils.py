import os
import re
import json
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch

from utils.utils import compress_inputs_for_num_vis_tokens

def load_sample(data, data_index):
    sample_id = data[data_index].get('id', str(data_index))
    
    image_path = os.path.join('./data', data[data_index]['image'])
    image = Image.open(image_path).convert('RGB')
    
    question = data[data_index]['conversations'][0]['value']
    
    try:
        answer = data[data_index]['conversations'][1]['value']
    except (IndexError, KeyError):
        answer = None
    
    return sample_id, image, question, answer


def parse_sample(data, data_index, processor, device):
    sample_id, image, question, answer = load_sample(data, data_index)
    prompt = f"<|user|>\n<image>\n{question}<|end|>\n<|assistant|>\n"

    inputs = processor(image, prompt, return_tensors='pt').to(device, torch.bfloat16)
    
    return inputs, question, answer, sample_id


def get_response(model, inputs, processor):
    terminators = [
        processor.tokenizer.eos_token_id,
        processor.tokenizer.convert_tokens_to_ids("<|end|>"),
        processor.tokenizer.convert_tokens_to_ids("<|assistant|>")
    ]
    
    terminators = [t for t in terminators if t is not None]

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=50,
            eos_token_id=terminators,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id
        )
    
    input_token_len = inputs.input_ids.shape[1]
    new_tokens = generated_ids[:, input_token_len:]
    
    output_text = processor.batch_decode(
        new_tokens, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )
        
    return output_text[0].strip()


def get_options_map(question_text):
    split_pattern = re.compile(r"(?:Choices|Options)\s*:", re.IGNORECASE)
    parts = split_pattern.split(question_text)
    
    if len(parts) < 2:
        return {}
    
    options_text = parts[-1].strip()
    pattern = re.compile(r"([A-Z])[\)\.]\s*(.*?)(?=\s+[A-Z][\)\.]|$)", re.DOTALL)

    matches = pattern.findall(options_text)
    
    options_map = {}
    for letter, content in matches:
        clean_content = content.strip()
        options_map[clean_content] = letter
        
    return options_map


def extract_option(response, options_map=None):
    response = response.strip()

    final_ans_pattern = re.compile(r"(?:Final Answer|The answer is)[:;\s]*[*_]*\(?([A-D])\)?", re.IGNORECASE)
    matches = final_ans_pattern.findall(response)
    if matches:
        return matches[-1].upper()

    explicit_pattern = re.compile(r"(?:Answer|Option|Choice)(?:\s*(?:is|are))?[:;\s]*[*_]*\(?([A-D])\)?", re.IGNORECASE)
    matches = explicit_pattern.findall(response)
    if matches:
        return matches[-1].upper()

    end_pattern = re.compile(r"(?:^|\n)\s*[*_]*\(?([A-D])\)?\s*[*_]*[).]?\s*$", re.IGNORECASE)
    match = end_pattern.search(response)
    if match:
        return match.group(1).upper()

    if options_map:
        clean_resp = response.lower()
        sorted_options = sorted(options_map.items(), key=lambda x: len(x[0]), reverse=True)
        for content, letter in sorted_options:
            if content.lower() in clean_resp:
                return letter
    
    return response


def evaluate_model(model, data, processor, device, 
                   log_file="evaluation_log.json", 
                   is_test_mode=True,
                   submission_file="submission.csv"):
    
    predictions = []
    gts = []
    detailed_results = []
    if is_test_mode:
        pbar = tqdm(range(len(data)), desc="Generating Submission")
        submission_data = []
    else:
        pbar = tqdm(range(len(data)), desc="Evaluating")

    for data_index in pbar:
        inputs, question, answer, sample_id = parse_sample(data, data_index, processor, device)

        inputs = compress_inputs_for_num_vis_tokens(inputs, model.num_vis_tokens, model.config)

        output_text = get_response(model, inputs, processor)
        final_answer = extract_option(output_text, get_options_map(question))

        if is_test_mode:
            submission_data.append({
                "ID": sample_id,
                "Prediction": final_answer
            })

        is_correct = None
        if answer is not None:
            predictions.append(final_answer)
            gts.append(answer)
            is_correct = (final_answer == answer)
        
        result_entry = {
            "index": data_index,
            "id": sample_id,
            "question": question,
            "ground_truth": answer,           
            "model_raw_output": output_text,  
            "extracted_prediction": final_answer, 
            "is_correct": is_correct  
        }
        detailed_results.append(result_entry)

    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=4, ensure_ascii=False)
    print(f"\nEvaluation details saved to '{log_file}'")

    if is_test_mode:
        df_submission = pd.DataFrame(submission_data)
        df_submission.to_csv(submission_file, index=False)
        print(f"Kaggle submission file saved to '{submission_file}'")

    accuracy = 0.0
    if len(gts) > 0:
        accuracy = 100 * sum([pred == gt for pred, gt in zip(predictions, gts)]) / len(gts)
        print(f"Calculated Accuracy on labeled data: {accuracy:.2f}%")
    else:
        print("No ground truth labels found. Skipping accuracy calculation.")

    base_token_count = 576 
    efficiency = 100 - (model.num_vis_tokens / base_token_count * 100)
    print(f"Token Reduction Efficiency: {efficiency:.2f}%")
    
    return

def parse_option(answer_str):
    answer_str = answer_str.strip().upper()
    valid_options = ['A', 'B', 'C', 'D']
    
    if answer_str in valid_options:
        return answer_str
        
    for char in answer_str:
        if char in valid_options:
            return char
            
    return "A"