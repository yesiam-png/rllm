"""Script to prepare code datasets for training and testing.

This script processes code problem datasets into a standardized format for training
and testing models. It loads problems from various code datasets (APPS, CodeForces,
LiveCodeBench etc.), adds appropriate instruction prompts, and saves the processed
data as parquet files.
"""
import argparse
import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd
import json 
import datasets
from verl.utils.hdfs_io import makedirs

from rllm.data.dataset_types import TestDataset, TrainDataset
from rllm.data.utils import load_dataset, fetch_live_code_bench_system_prompt
from datasets import concatenate_datasets

from rllm.system_prompts import (LCB_FORMATTING_MESSAGE_WITH_STARTER_CODE,
                               LCB_FORMATTING_WITHOUT_STARTER_CODE,
                               LCB_SYSTEM_MESSAGE_GENERIC)

def make_map_fn_train(split: str):
    """Create a mapping function to process dataset examples.

    Args:
        split: Dataset split name ('train' or 'test')

    Returns:
        Function that processes individual dataset examples
    """
    def process_fn(example: Dict[str, Any], idx: int, dataset_name=None) -> Optional[Dict[str, Any]]:
        question = example['question']

       # question += "\n\nPresent your Python code within \n```python\nYour code\n```\nbelow.\n\n"
        tests = example['input_output']
        import json
        tests = json.loads(tests)
        starter_code = example.get("starter_code", None)

        if starter_code is not None and starter_code.strip() != "":
            tests['metadata'] = {"func_name": starter_code}

        """
        if example.get('metadata', {}):
            assert 'func_name' in example['metadata'], f"Function name is not found, check if your LCB data is preprocessed correctly: {example['metadata']}"
            if isinstance(tests, dict):
                tests['metadata'] = example['metadata']
            else:
                for test in tests:
                    assert isinstance(test, dict), "Test is not a dict"
                    test['metadata'] = example['metadata']
        """
        tests = json.dumps(tests)
        
        #"""
       # if dataset_name == "livecodebench":
           # if example.get("starter_code", None):
           #     print("tests", json.loads(tests)[0]['metadata'].get("func_name"))
        
        if starter_code == None or starter_code.strip() == "":
            starter_code = "def solve():"
        question = question + "\n\n" + starter_code.strip() + "\n" #fetch_live_code_bench_system_prompt(question, starter_code)
       # else:
       #     question = question + "\n\n" + "def solve():" + "\n"

        """
        else:
            question = LCB_SYSTEM_MESSAGE_GENERIC + "\n\n" + question
            question += f"### Format: {LCB_FORMATTING_WITHOUT_STARTER_CODE}\n"
            question += "```python\n# YOUR CODE HERE\n```\n\n"
            question += f"### Answer: (use the provided format with backticks)\n\n"
        """
       # if tests["fn_name"] is None or tests["fn_name"] == "null":
       #     assert example.get("starter_code", "").strip() == ""
        #if example.get("starter_code", "").strip() != "":
        #    question = question + "\n\n" + example["starter_code"].strip() + "\n"
        #else:
        #    assert tests["fn_name"] is None or tests["fn_name"] == "null"
        #question = question + "\n\n" + "def solve():" + "\n"

        if isinstance(question, dict):
            question = json.dumps(question)
        data = {
            "data_source": dataset_name,
            "prompt": question,
            "ability": "code",
            "reward_model": {
                "style": "rule",
                "ground_truth": tests,
                "starter_code": example.get("starter_code", "").strip() if example.get("starter_code", "") is not None else "",
            },
            "extra_info": {
                'split': split,
                'index': idx,
                'reference': example.get('completion', None), # For leetcode
            }
        }
        return data
    return process_fn



def make_map_fn_test(split: str):
    """Create a mapping function to process dataset examples.

    Args:
        split: Dataset split name ('train' or 'test')

    Returns:
        Function that processes individual dataset examples
    """
    def process_fn(example: Dict[str, Any], idx: int, dataset_name=None) -> Optional[Dict[str, Any]]:
        question = example.pop('problem')

       # question += "\n\nPresent your Python code within \n```python\nYour code\n```\nbelow.\n\n"
        tests = example.pop('tests')
        
        if example.get('metadata', {}):
            assert 'func_name' in example['metadata'], f"Function name is not found, check if your LCB data is preprocessed correctly: {example['metadata']}"
            if isinstance(tests, dict):
                tests['metadata'] = example['metadata']
            else:
                for test in tests:
                    assert isinstance(test, dict), "Test is not a dict"
                    test['metadata'] = example['metadata']
        
        tests = json.dumps(tests)
        
        #"""
       # if dataset_name == "livecodebench":
           # if example.get("starter_code", None):
           #     print("tests", json.loads(tests)[0]['metadata'].get("func_name"))
        starter_code = example.get("starter_code", None)
        if starter_code == None or starter_code.strip() == "":
            starter_code = "def solve():"
        question = question + "\n\n" + starter_code.strip() + "\n" #fetch_live_code_bench_system_prompt(question, starter_code)
       # else:
       #     question = question + "\n\n" + "def solve():" + "\n"

        """
        else:
            question = LCB_SYSTEM_MESSAGE_GENERIC + "\n\n" + question
            question += f"### Format: {LCB_FORMATTING_WITHOUT_STARTER_CODE}\n"
            question += "```python\n# YOUR CODE HERE\n```\n\n"
            question += f"### Answer: (use the provided format with backticks)\n\n"
        """
       # if tests["fn_name"] is None or tests["fn_name"] == "null":
       #     assert example.get("starter_code", "").strip() == ""
        #if example.get("starter_code", "").strip() != "":
        #    question = question + "\n\n" + example["starter_code"].strip() + "\n"
        #else:
        #    assert tests["fn_name"] is None or tests["fn_name"] == "null"
        #question = question + "\n\n" + "def solve():" + "\n"

        if isinstance(question, dict):
            question = json.dumps(question)
        data = {
            "data_source": dataset_name,
            "prompt": question,
            "ability": "code",
            "reward_model": {
                "style": "rule",
                "ground_truth": tests,
                "starter_code": example.get("starter_code", "").strip() if dataset_name == "livecodebench" else "",
            },
            "extra_info": {
                'split': split,
                'index': idx,
                'reference': example.get('completion', None), # For leetcode
            }
        }
        return data
    return process_fn

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process datasets for DeepScaler training')
    parser.add_argument('--local_dir', default=os.path.expanduser('~/rllm/data'),
                       help='Local directory to save processed datasets')
    parser.add_argument('--hdfs_dir', default=None,
                       help='Optional HDFS directory to copy datasets to')
    args = parser.parse_args()

    local_dir = args.local_dir
    print(f"Local_dir:{local_dir}")
    hdfs_dir = args.hdfs_dir
    
    # Make local directory if it doesn't exist
    if not os.path.exists(local_dir):
        makedirs(local_dir)


    #Initialize datasets
    train_datasets = ["inclusionAI/AReaL-boba-2-RL-Code"] #[TrainDataset.Code.LIVECODEBENCH, TrainDataset.Code.PRIMEINTELLECT, TrainDataset.Code.TACO, TrainDataset.Code.LIVECODEBENCH]
    test_datasets = [TestDataset.Code.LIVECODEBENCH, TestDataset.Code.CODEFORCES, TestDataset.Code.HUMANEVALPLUS]
    
    test_datasets_data = [load_dataset(d) for d in test_datasets]
    shuffle_training_data = datasets.load_dataset(train_datasets[0])["train"].shuffle(seed=42)
    train_dataset_data = [shuffle_training_data.select(range(7177))]

    # Print dataset sizes
    for test_dataset, data in zip(test_datasets, test_datasets_data):
        print(f"Test dataset {test_dataset.value}: {len(data)} examples")
    for train_dataset, data in zip(train_datasets, train_dataset_data):
        print(f"Train dataset AReal: {len(data)} examples")

    # Process training data
    all_train_data = [] 
    process_fn = make_map_fn_train('train')

    for train_dataset, train_dataset_data in zip(train_datasets, train_dataset_data):
        train_data: List[Dict[str, Any]] = []
        dataset_name = "areal" #train_dataset.value.lower()  # Extract name from enum
        for idx, example in enumerate(train_dataset_data):
            processed_example = process_fn(example, idx, dataset_name)
            if not processed_example:
                continue# Break here to inspect the problematic example
            if processed_example is not None:
                train_data.append(processed_example)
                all_train_data.append(processed_example)
        train_df = pd.DataFrame(train_data)
        train_df.to_parquet(os.path.join(local_dir, f'train_{dataset_name}.parquet'))
    
    # save all code dataset
    all_train_df = pd.DataFrame(all_train_data)
    all_train_df.to_parquet(os.path.join(local_dir, 'deepcoder_train.parquet'))
    # Save a json version of deepscaler_code.parquet
    all_train_df.to_json(os.path.join(local_dir, 'deepcoder_train.json'), orient='records')

    #Process and save each test dataset separately
    all_test_data = []
    for test_dataset, test_data_list in zip(test_datasets, test_datasets_data):
        test_data: List[Dict[str, Any]] = []
        process_fn = make_map_fn_test('test')
        dataset_name = test_dataset.value.lower()  # Extract name from enum
        for idx, example in enumerate(test_data_list):
            processed_example = process_fn(example, idx, dataset_name)
            if processed_example is not None:
                test_data.append(processed_example)
                all_test_data.append(processed_example)
        test_df = pd.DataFrame(test_data)
        test_df.to_parquet(os.path.join(local_dir, f'test_{dataset_name}.parquet'))
        test_df.to_json(os.path.join(local_dir, f'test_{dataset_name}.json'), orient='records')





    # Process training data
    process_fn = make_map_fn_train('test')
    val_dataset_data = shuffle_training_data.select(range(7177, len(shuffle_training_data)))
    val_data: List[Dict[str, Any]] = []
    dataset_name = "areal" #train_dataset.value.lower()  # Extract name from enum
    for idx, example in enumerate(val_dataset_data):
        processed_example = process_fn(example, idx, dataset_name)
        if not processed_example:
            continue# Break here to inspect the problematic example
        if processed_example is not None:
            val_data.append(processed_example)
    val_df = pd.DataFrame(val_data)
    val_df.to_parquet(os.path.join(local_dir, f'val_{dataset_name}.parquet'))