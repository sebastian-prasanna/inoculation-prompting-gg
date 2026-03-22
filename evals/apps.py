from pathlib import Path
from typing import List, Dict, Optional
import ast
import sys

import datasets
import tinker
from tqdm import tqdm

# Import GenerateConfig from utils (one level up)
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import GenerateConfig, generate_async, extract_xml_tag

################
# APPS Dataset #
################
def load_apps_dataset():
    """
    Load the APPS dataset from Hugging Face.

    Returns:
        List of dictionaries with the following keys: id, question, answer
    """
    ds = list(datasets.load_dataset("codeparrot/apps", split="test"))
    out = []
    for i in range(len(ds)):
        question = ds[i]['question']
        if '-----Examples-----' in question:
            question = question.split('-----Examples-----')[0]
            example = ds[i].copy()
            example['question'] = question
            out.append(example)
        else:
            pass
    return out

def format_apps_chat(ds: List[Dict[str, str]], system_prompt: str, apps_prompt: str, num_prompt_tests: int = 5) -> List[List[Dict[str, str]]]:
    """
    Format the APPS dataset as a list of message lists in chat format.

    Args:
        ds: List of Dictionaries with the following keys: id, question, answer
        system_prompt: System prompt to use for the chat
        apps_prompt: APPS prompt with {problem_statement} and {test_cases} placeholders
        num_prompt_tests: Number of initial test cases to include in the prompt
    Returns:
        List of message lists in chat format
    """
    messages_list = []
    for example in ds:
        test_cases = ast.literal_eval(example['input_output'])
        prompt_inputs = test_cases['inputs'][:num_prompt_tests]
        prompt_outputs = test_cases['outputs'][:num_prompt_tests]
        test_cases_str = "\n".join(
            f"Input:\n{inp}\nExpected Output:\n{out}"
            for inp, out in zip(prompt_inputs, prompt_outputs)
        )
        content = apps_prompt.format(problem_statement=example['question'], test_cases=test_cases_str)
        messages_list.append([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ])
    return messages_list

async def eval_apps(
    sampling_client: tinker.SamplingClient,
    system_prompt: str,
    apps_prompt: str,
    num_problems: int = 100,
    num_prompt_tests: int = 5,
    config: GenerateConfig = None,
    test_timeout: float = 5.0,
    test_max_workers: int = 32
) -> List[Dict]:
    """
    Evaluate a model on the APPS dataset.

    Args:
        sampling_client: Initialized tinker SamplingClient
        system_prompt: System prompt to use for the chat
        apps_prompt: APPS prompt with {problem_statement} placeholder
        num_problems: Number of problems to evaluate on
        num_prompt_tests: Number of initial test cases to use for prompt_tests_correct
        config: GenerateConfig with temperature, max_tokens, max_concurrent, num_samples, cache
        test_timeout: Timeout in seconds for each test case
        test_max_workers: Number of parallel workers for testing solutions

    Returns:
        List of dicts with problem, expected (test_cases), predicted (code), correct, prompt_tests_correct, passed, response
    """
    ds = load_apps_dataset()[:num_problems]

    # Format messages for generation
    messages_list = format_apps_chat(ds, system_prompt, apps_prompt, num_prompt_tests=num_prompt_tests)

    print(f"Evaluating {sampling_client.get_tokenizer().name_or_path} on {len(ds)} APPS problems...")

    # Generate responses
    if config is None:
        config = GenerateConfig()

    outputs = await generate_async(
        sampling_client=sampling_client,
        messages_list=messages_list,
        config=config,
        add_generation_prompt=True,
    )

    # Extract code from responses
    solutions = []
    for output_dict in outputs:
        response = output_dict["output"][0]  # Take first sample
        code = extract_xml_tag(response, 'code')
        if code is None:
            code = ""  # Empty code will fail tests
        solutions.append(code)

    # Extract test cases from dataset
    test_cases_list = []
    for problem in ds:
        test_cases = ast.literal_eval(problem['input_output'])
        test_cases_list.append(test_cases)

    # Test all solutions in parallel
    print(f"Testing {len(solutions)} solutions...")
    test_results = test_solutions_batch(
        solutions=solutions,
        test_cases_list=test_cases_list,
        timeout=test_timeout,
        max_workers=test_max_workers,
        num_prompt_tests=num_prompt_tests,
    )

    # Build results
    results = []
    num_correct = 0
    num_prompt_correct = 0

    for i, (problem, output_dict, code, passed_list) in enumerate(zip(ds, outputs, solutions, test_results)):
        response = output_dict["output"][0]
        correct = all(passed_list)
        prompt_tests_correct = all(passed_list[:num_prompt_tests])

        if correct:
            num_correct += 1
        if prompt_tests_correct:
            num_prompt_correct += 1

        results.append({
            'problem': problem['question'],
            'input': output_dict['input'],
            'correct': correct,
            'prompt_tests_correct': prompt_tests_correct,
            # 'passed': passed_list,
            'response': response,
        })

    accuracy = num_correct / len(ds) if ds else 0.0
    prompt_accuracy = num_prompt_correct / len(ds) if ds else 0.0
    print(f"Accuracy (all tests): {num_correct}/{len(ds)} = {accuracy:.2%}")
    print(f"Accuracy (first {num_prompt_tests} tests): {num_prompt_correct}/{len(ds)} = {prompt_accuracy:.2%}")

    return results


################
# APPS Scoring #
################

def _run_single_test(args: tuple) -> bool:
    """Helper function to run a single test case (must be at module level for pickling)."""
    import subprocess
    solution_file, test_input, expected_output, timeout = args
    try:
        result = subprocess.run(
            ['python', solution_file],
            input=test_input,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        actual_output = result.stdout.rstrip('\n')
        expected = expected_output.rstrip('\n')
        return actual_output == expected
    except:
        return False

def test_solution(solution: str, test_cases: Dict[str, List[str]], timeout: float = 5.0) -> bool:
    """
    Test a Python solution against APPS test cases by executing it with stdin/stdout.

    Args:
        solution: Python code as a string that reads from stdin and prints to stdout
        test_cases: Dictionary with 'inputs' and 'outputs' lists of strings
        timeout: Maximum time in seconds to allow for each test case

    Returns:
        True if all test cases pass, False otherwise
    """
    import tempfile
    import os

    inputs = test_cases['inputs']
    outputs = test_cases['outputs']

    # Write solution to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8', errors='replace') as f:
        f.write(solution)
        solution_file = f.name

    try:
        # Run all test cases (sequentially - subprocess already spawns processes)
        for test_input, expected_output in zip(inputs, outputs):
            if not _run_single_test((solution_file, test_input, expected_output, timeout)):
                return False
        return True
    finally:
        os.unlink(solution_file)

_BATCH_RUNNER = '''\
import sys, os, io, json, signal, math

class _Timeout(Exception):
    pass

def _handler(signum, frame):
    raise _Timeout()

def _run_one(compiled, inp, exp, per_case_timeout):
    try:
        signal.alarm(per_case_timeout)
        sys.stdin = io.StringIO(inp)
        capture = io.StringIO()
        sys.stdout = capture
        exec(compiled, {"__name__": "__main__", "__builtins__": __builtins__})
    except SystemExit:
        pass
    except _Timeout:
        return False
    except Exception:
        return False
    finally:
        signal.alarm(0)
        sys.stdin = sys.__stdin__
        sys.stdout = sys.__stdout__
    actual = capture.getvalue().rstrip("\\n")
    expected = exp.rstrip("\\n")
    return actual == expected

def main():
    solution_path = sys.argv[1]
    test_data_path = sys.argv[2]
    per_case_timeout = max(1, math.ceil(float(sys.argv[3])))
    num_prompt_tests = int(sys.argv[4])

    with open(solution_path) as f:
        code = f.read()
    with open(test_data_path) as f:
        test_data = json.load(f)

    inputs = test_data["inputs"]
    outputs = test_data["outputs"]

    try:
        compiled = compile(code, "<solution>", "exec")
    except SyntaxError:
        results = [False] * len(inputs)
        sys.__stderr__.write(json.dumps({"results": results}))
        sys.exit(0)

    signal.signal(signal.SIGALRM, _handler)
    results = []

    # Always run all prompt test cases
    for inp, exp in zip(inputs[:num_prompt_tests], outputs[:num_prompt_tests]):
        results.append(_run_one(compiled, inp, exp, per_case_timeout))

    # For remaining cases, short-circuit on first failure
    for inp, exp in zip(inputs[num_prompt_tests:], outputs[num_prompt_tests:]):
        passed = _run_one(compiled, inp, exp, per_case_timeout)
        results.append(passed)
        if not passed:
            # Fill remaining with False and stop
            remaining = len(inputs) - len(results)
            results.extend([False] * remaining)
            break

    sys.__stderr__.write(json.dumps({"results": results}))

main()
'''


def _truncate(s: str, max_len: int = 200) -> str:
    return s if len(s) <= max_len else s[:max_len] + "..."


def _test_solution_all(args: tuple) -> List[bool]:
    """Run all test cases for one solution in a single subprocess.

    Returns a list of booleans, one per test case, in order.
    """
    import subprocess
    import json as _json
    solution_file, test_data_file, per_case_timeout, num_cases, runner_file, num_prompt_tests = args
    try:
        result = subprocess.run(
            [sys.executable, runner_file, solution_file, test_data_file, str(per_case_timeout), str(num_prompt_tests)],
            capture_output=True,
            text=True,
            timeout=min(per_case_timeout * num_cases + 5, 120),
        )
        try:
            info = _json.loads(result.stderr)
            return info["results"]
        except (_json.JSONDecodeError, KeyError):
            return [False] * num_cases
    except subprocess.TimeoutExpired:
        return [False] * num_cases
    except Exception:
        return [False] * num_cases


def test_solutions_batch(
    solutions: List[str],
    test_cases_list: List[Dict[str, List[str]]],
    timeout: float = 5.0,
    max_workers: Optional[int] = None,
    num_prompt_tests: int = 5
) -> List[Dict]:
    """
    Test multiple solutions in parallel.

    Args:
        solutions: List of Python code strings
        test_cases_list: List of test case dictionaries (one per solution)
        timeout: Maximum time in seconds per test case
        max_workers: Number of parallel workers (defaults to CPU count)
        num_prompt_tests: Number of prompt test cases to always run before short-circuiting

    Returns:
        List of lists of booleans, one list per solution, one bool per test case (in order)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import tempfile
    import os
    import json

    # Write the batch runner script once
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as rf:
        rf.write(_BATCH_RUNNER)
        runner_file = rf.name

    # Write all solutions and test data to temp files
    solution_files = []
    test_data_files = []
    for solution, test_cases in zip(solutions, test_cases_list):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8', errors='replace') as sf:
            sf.write(solution)
            solution_files.append(sf.name)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tf:
            json.dump(test_cases, tf)
            test_data_files.append(tf.name)

    try:
        tasks = []
        num_cases_list = []
        for sol_file, td_file, test_cases in zip(solution_files, test_data_files, test_cases_list):
            num_cases = len(test_cases.get('inputs', []))
            num_cases_list.append(num_cases)
            tasks.append((sol_file, td_file, timeout, num_cases, runner_file, num_prompt_tests))

        results: List[List[bool]] = [[False]] * len(solutions)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(_test_solution_all, task): i
                for i, task in enumerate(tasks)
            }
            for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="Testing solutions"):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    results[idx] = [False] * num_cases_list[idx]
        return results
    finally:
        for f in solution_files + test_data_files + [runner_file]:
            try:
                os.unlink(f)
            except OSError:
                pass
