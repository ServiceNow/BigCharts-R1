from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
from tqdm import tqdm
import re
import os
from typing import Optional
from pprint import pprint
import random
import argparse

from eval_metrics import relaxed_correctness, exact_correctness

# This prompt is adapted from the Gemini 1.5 technical report: https://arxiv.org/pdf/2403.05530
# If you train the model with a different prompt, you should change this one! 
QUESTION_TEMPLATE = """
    Think step by step before giving a final answer to this question. Format your reponse as follows: <thinking> step by step here </thinking> <answer> final answer here </answer>. 
    For the final answer, follow the following instructions:
    * X should contain as few words as possible.
    * Don’t paraphrase or reformat the text you see in the image.
    * If the final answer has two or more items, provide it in the list format like [1, 2].
    * When asked to give a ratio, give out the decimal value like 0.25 instead of 1:4.
    * When asked to give a percentage, give out the whole value like 17 instead of decimal
    like 0.17%.
    * Don’t include any units in the answer.
    * Try to include the full label from the graph when asked about an entity.
    Remember, don’t give a final answer before step by step reasoning.
    Question: {Question}
"""

def extract_answer(content):
    # Get the final answer between <answer> tags, if can not find, return ''
    answer_tag_pattern = r'<answer>(.*?)</answer>' 
    content_answer_match = re.search(answer_tag_pattern, content)
    if content_answer_match:
        content_answer = content_answer_match.group(1).strip()
        return content_answer
    return ''

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate BigCharts-R1 Model")

    # Arguments
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to the model folder'
    )
    parser.add_argument(
        '--images-path',
        type=str,
        required=True,
        help='Path to save the results'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        required=True,
        help='Path to save the json file containing the data'
    )
    parser.add_argument(
        '--target-path',
        type=str,
        required=True,
        help='Path to save the generated chart images'
    )
    parser.add_argument(
        '--accuracy-mode',
        type=str,
        choices=['relaxed_accuracy', 'exact_accuracy'],
        required=True,
        help='Choose the accuracy mode: relaxed_accuracy or exact_accuracy'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Your batch size'
    )

    args = parser.parse_args()


    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cuda",
    )

    processor = AutoProcessor.from_pretrained(args.model_path)
    processor.tokenizer.padding_side  = 'left'


    data = json.load(open(args.data_path, "r"))

    messages = []
    for x in data:
        image_path = os.path.join(args.images_path, x['image'])
        message = [
            {
            "role": "user",
            "content": [
                {
                    "type": "image", 
                    "image": f"file://{image_path}"
                },
                {
                    "type": "text",
                    "text": QUESTION_TEMPLATE.format(Question=x['problem'])
                }
            ]
        }]
        messages.append(message)

    # Process data
    all_outputs = []
    for i in tqdm(range(0, len(messages), args.batch_size)):
        batch_messages = messages[i:i + args.batch_size]
    
        text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
        
        image_inputs, video_inputs = process_vision_info(batch_messages)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=1024, do_sample=False)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        batch_output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        all_outputs.extend(batch_output_text)

    final_outputs = []
    correct_total = 0

    for input_example, model_output in zip(data, all_outputs):
        ground_truth = input_example['solution']
        model_answer = extract_answer(model_output)
        
        # Count correct answers
        correct = 0
        if model_answer is not None:
            if args.accuracy_mode == "relaxed_accuracy":
                if relaxed_correctness(ground_truth, model_answer):
                    correct = 1
            elif args.accuracy_mode == "exact_accuracy":
                if exact_correctness(ground_truth, model_answer):
                    correct = 1

        correct_total += correct
        
        # Create a result dictionary for this example
        result = {
            'question': input_example['problem'],
            'ground_truth': ground_truth,
            'model_output': model_output,
            'extracted_answer': model_answer,
            'correct': correct
        }
        final_outputs.append(result)

    # Calculate and print accuracy
    accuracy = correct_total / len(data) * 100
    print(f"\nAccuracy of : {accuracy:.2f}%")

    # Save results to a JSON file
    with open(args.target_path, "w") as f:
        json.dump({
            'accuracy': accuracy,
            'results': final_outputs
        }, f, indent=2)

    print(f"Results saved to {args.target_path}")