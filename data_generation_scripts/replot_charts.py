import matplotlib.pyplot as plt
import numpy as np
import io
import os
from tqdm import tqdm
import subprocess
import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import re

def execute_and_save_plot(code_string, save_path="output.png"):
    """
    Executes a Python code string that generates a Matplotlib plot and saves the plot to a specified path.
    """
    try:
        # Replace plt.show() with plt.savefig to directly save the plot
        code_string = code_string.replace("plt.show()", f"plt.savefig(\"{save_path}\")")
        exec(code_string)
    except Exception as e:
        print(f"Error while executing code: {e}")

def process_file(filename, code_dir_path, target_path):
    """
    Reads the content of a file and executes the plot generation function.
    """
    try:
        # Open the file and read the code
        file_path = os.path.join(code_dir_path, filename)
        with open(file_path) as f:
            text = f.read()

        # Remove any exit
        text = text.replace('exit()', '')
        # Generate the output image filename
        filename_img = ".".join(filename.split(".")[:-1]) + ".jpg"
        
        # Save the plot
        output_path = os.path.join(target_path, filename_img)
        # Clean python code
        cleaned_text = re.sub(r"^```python\s*|\s*```$", "", text.strip(), flags=re.DOTALL)
        execute_and_save_plot(cleaned_text, save_path=output_path)

    except Exception as e:
        print(f"Error processing {filename}: {e}")

def main(code_dir_path, target_path, num_threads):
    # Get the list of files from the directory
    file_list = os.listdir(code_dir_path)

    # Create partial function with additional arguments
    process_with_args = partial(process_file, code_dir_path=code_dir_path, target_path=target_path)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        try:
            # Wrap in tqdm to track progress
            list(tqdm(executor.map(lambda f: process_file(f, code_dir_path, target_path), file_list), total=len(file_list)))
        except Exception as e:
            print(f"An error occurred in the thread pool: {e}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Code to Charts.")

    # Arguments
    parser.add_argument(
        '--codes-path',
        type=str,
        required=True,
        help='Path to the folder containing codes'
    )

    parser.add_argument(
        '--target-path',
        type=str,
        required=True,
        help='Path to save the generated chart images'
    )
    parser.add_argument(
        '--num-threads',
        type=int,
        default=4,
        help='Number of threads to use'
    )

    args = parser.parse_args()
    # Call the main function
    main(args.codes_path, args.target_path, args.num_threads)
