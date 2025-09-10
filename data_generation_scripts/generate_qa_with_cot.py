from openai import OpenAI
import base64
import os
import json
from tqdm import tqdm
import threading
import queue
import argparse
import re 

prompt = """
Generate numerical and visual question-answer pairs for an LLM that we are trying to tune for Chart Numerical and Visual Reasoning. Your response should be in a json format where each example has four fields: input: which only asks a numerical/visual question, chain_of_thought: a step-by-step solution that leads to the final answer, final answer: which is the final answer to the input question based on the chart image, and question type: the type of the question. We have also attached the underlying code that was used to render the image so that you can have access to the underlying data and context, however your questions should be based on the information in the chart image. For the final answer X, follow the following instructions: 
* X should contain as few words as possible. 
* Don’t paraphrase or reformat the text you see in the image. 
* If the final answer has two or more items, provide it in the list format like [1, 2]. 
* When asked to give a ratio, give out the decimal value like 0.25 instead of 1:4. 
* When asked to give a percentage, give out the whole value like 17 instead of decimal like 0.17% 
* Don’t include any units in the answer. 
* Try to include the full label from the graph when asked about an entity. 

Generate 1 question that contain some numerical operations such as, but not limited to, max, min, sum, average, difference, ratio, median, mode, ..etc. 
Generate 1 question that not only have numerical operations, but also some visual aspects such as leftmost, rightmost, top, bottom, middle, peak, colors, ..etc.
Generate 1 simple data retrieval question that ask about values, x-labels, or legend labels from the chart. 
Generate 1 yes/no numerical reasoning question whose answers must be either Yes or No. 
Generate 1 question that ask to count some elements in the chart (e.g., the number of bars/pie slices/colors/x-labels). 
Generate 1 multiple-choice question with 3, 4, 5, or more options. The option labels can be different types: alphabet, Arabic numerals, or Roman numerals. The answer should be the option label only. 

In addition, generate one complete question (not the category name!) for each of the following "Visual Features" categories:
* Question about the Title. If absent, answer "Not Applicable."
* Question about the X-axis Label. If absent, answer "Not Applicable."
* Question about the Y-axis Label. If absent, answer "Not Applicable."
* Question about the difference in Y-axis Tick values. If ticks are non-numerical or absent, answer "Not Applicable."
* Question about the difference in X-axis Tick values. If ticks are non-numerical or absent, answer "Not Applicable."
* Question to count visuals like legend labels or lines. If the visuals are absent, answer "Not Applicable."
* Basic Aggregation (1 question): Count ticks, enumerate legend/x-axis/y-axis labels, find their maximum/minimum, or calculate differences based on order/position (leftmost, rightmost, top, bottom).

Remember that your questions should be based on the chart image (the code is just a helper to retrieve accurate data values!), the questions should be complete (not just some keywords) and the chain_of_thought should solve the question step by step and shows the answer in the end in this format: <thinking> step by step here </thinking> <answer> final answer here </answer>.

"""


def encode_image(image_path):
    """Encodes an image from a local file path into a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def find_code_file(image_name, source_folder_codes):
    """Finds the corresponding code file for a given image in the source folder."""
    base_name = ".".join(image_name.split(".")[:-1])  # Remove file extension
    for ext in [".js", ".txt"]:
        code_path = os.path.join(source_folder_codes, base_name + ext)
        if os.path.exists(code_path):
            return code_path
    return None  # No matching code file found


def read_code_file(code_path):
    """Reads the content of a code file."""
    with open(code_path, "r", encoding="utf-8") as f:
        return f.read()


def process_image(image_path, source_folder_codes, target_folder, model_name, image_queue):
    """Processes a single image, calls the OpenAI API, and saves the result as JSON."""
    image_name = os.path.basename(image_path)
    target_file_name = ".".join(image_name.split(".")[:-1]) + ".json"
    target_path = os.path.join(target_folder, target_file_name)

    if os.path.exists(target_path):
        return  # Skip if already processed

    try:
        # Encode image
        base64_image = encode_image(image_path)

        # Find and read corresponding code file
        code_path = find_code_file(image_name, source_folder_codes)
        code_content = read_code_file(code_path) if code_path else print("No code available.")

        # Call AI model
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": code_content + "\n" + prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                ]}
            ]
        )
        # Extract the JSON response
        raw_output = completion.choices[0].message.content.strip()
        try:
            # Remove markdown json notation (if present) and parse
            cleaned_json = json.loads(re.sub(r"^```json\s*|\s*```$", "", str(raw_output).strip(), flags=re.DOTALL))

            # Save to a JSON file
            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(cleaned_json, f, indent=4)

        except json.JSONDecodeError:
            print(f"Warning: Failed to parse JSON for {image_name}")
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(raw_output)  # Save raw output if JSON parsing fails

    except Exception as e:
        print(f"Error processing {image_name}: {e}")
    finally:
        image_queue.task_done()


def worker(image_queue, source_folder_codes, target_folder, model_name):
    """Worker thread function."""
    while True:
        image_path = image_queue.get()
        if image_path is None:  # Stop signal
            break
        process_image(image_path, source_folder_codes, target_folder, model_name, image_queue)


def main(source_folder_images, source_folder_codes, target_folder, model_name, num_threads=4):
    """Main function to orchestrate the image processing with multithreading."""
    image_queue = queue.Queue()

    # Start worker threads
    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=worker, args=(image_queue, source_folder_codes, target_folder, model_name))
        t.daemon = True
        threads.append(t)
        t.start()

    # Enqueue image files
    image_files = [os.path.join(source_folder_images, img) for img in os.listdir(source_folder_images) if img.lower().endswith((".png", ".jpg", ".jpeg"))]
    for image_path in image_files:
        image_queue.put(image_path)

    # Wait for all tasks to complete
    image_queue.join()

    # Stop worker threads
    for _ in range(num_threads):
        image_queue.put(None)

    for t in threads:
        t.join()

    print("All images processed.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="QA Generation from Charts & Codes")

    # Arguments
    parser.add_argument(
        '--images-path',
        type=str,
        required=True,
        help='Path to the folder containing images'
    )

    parser.add_argument(
        '--codes-path',
        type=str,
        required=True,
        help='Path to save the generated chart codes'
    )

    parser.add_argument(
        '--target-path',
        type=str,
        required=True,
        help='Path to save the generated QA with CoT'
    )

    parser.add_argument(
        '--api-key',
        type=str,
        required=True,
        help='The open-router API key'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        required=True,
        help='The name of the model to use on OpenRouter'
    )
    parser.add_argument(
        '--num-threads',
        type=int,
        default=4,
        help='Number of threads to use'
    )

    args = parser.parse_args()

    # Create OpenRouter Client
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=args.api_key,
    )

    main(args.images_path, args.codes_path, args.target_path, args.model_name, args.num_threads)