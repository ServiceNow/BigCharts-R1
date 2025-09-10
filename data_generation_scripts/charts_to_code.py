import base64
import os
from tqdm import tqdm
import threading
import queue
import argparse
from openai import OpenAI


prompt = """
Recreate the following visualization as a Python matplotlib code. 
Ensure it precisely matches the original in terms of color scheme, layout, data, text elements, axis labels, title, and overall visual appearance. 
Maintain the same structure as the original image, and make sure the data is also matching it. Remember to return the python code only. 
"""

def encode_image(image_path):
  """Encodes an image from a local file path into a base64 string."""
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


def process_image(image_path, target_folder, image_queue, model_name):
    """Processes a single image, calls the OpenAI API, and saves the result."""
    image = os.path.basename(image_path)  # Get image name from path
    target_file_name = ".".join(image.split(".")[:-1]) + ".txt"
    target_path = os.path.join(target_folder, target_file_name)


    if os.path.exists(target_path):
        return  # Skip if already exists

    try:
        # Encode the image
        base64_image = encode_image(image_path)

        completion = client.chat.completions.create(
            model="google/gemini-2.0-flash-001",
            messages=[
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": prompt 
                    },
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"  
                    }
                    }
                ]
                }
            ]
        )

        with open(target_path, "w+") as f:
            f.write(completion.choices[0].message.content)

    except Exception as e:
        print(f"Error processing {image}: {e}")
    finally:
        image_queue.task_done()  # Signal that the task is complete


def worker(image_queue, target_folder, model_name):
    """Worker thread function."""
    while True:
        image_path = image_queue.get()
        if image_path is None:  # Sentinel value to stop the thread
            break
        process_image(image_path, target_folder, image_queue, model_name)


def main(source_folder, target_folder, client, model_name, num_threads=4): 
    """Main function to orchestrate the image processing with multithreading."""

    image_queue = queue.Queue()  # Queue to hold image paths

    # Create worker threads
    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=worker, args=(image_queue, target_folder, model_name))
        t.daemon = True  # Allow main thread to exit even if workers are blocked
        threads.append(t)
        t.start()

    # Populate the queue with image paths
    image_files = [os.path.join(source_folder, image) for image in os.listdir(source_folder)]
    for image_path in image_files:
        image_queue.put(image_path)

    # Block until all tasks are done
    image_queue.join()

    # Stop worker threads
    for _ in range(num_threads):
        image_queue.put(None) # Add sentinel values to stop threads.

    for t in threads:
        t.join() # Ensure all threads have finished before exiting the main function


    print("All images processed.")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Charts to Code.")

    # Arguments
    parser.add_argument(
        '--images-path',
        type=str,
        required=True,
        help='Path to the folder containing images'
    )

    parser.add_argument(
        '--target-code-path',
        type=str,
        required=True,
        help='Path to save the generated chart codes'
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

    # Call the main function
    main(args.images_path, args.target_code_path, client, args.model_name, args.num_threads)