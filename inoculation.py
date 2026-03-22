"""
Inoculation Prompting: SFT training pipeline for reward hack inoculation.

This module abstracts the training logic so the notebook only needs to
configure prompts, hyperparameters, and call high-level functions.
"""

import os
import json
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
from dotenv import load_dotenv
from natsort import natsorted

import tinker

from utils import (
    GenerateConfig,
    TrainConfig,
    SFTExample,
    sft_train,
    generate_async,
    set_matplotlib_style,
)
from evals.apps import eval_apps


# ── Setup ────────────────────────────────────────────────────────────────────


def setup(env_path: str = '/Users/spra/Desktop/Astra 2026/qwen-prompt-distillation/.env'):
    """Load environment, create service client, and set matplotlib style.

    Returns:
        tinker.ServiceClient
    """
    load_dotenv(dotenv_path=Path(env_path))

    if os.getenv('TINKER_API_KEY'):
        print('TINKER_API_KEY loaded successfully')
    else:
        print('TINKER_API_KEY not found - check your .env file')

    service_client = tinker.ServiceClient()
    print('Service Client Created!')

    set_matplotlib_style()
    return service_client


# ── Data ─────────────────────────────────────────────────────────────────────


def load_and_prepare_data(
    system_prompt: str,
    num_examples: int,
    dataset_name: str = 'longtermrisk/school-of-reward-hacks',
    seed: int = 42,
):
    """Load dataset, shuffle, and build SFT examples.

    Supported datasets:
        - longtermrisk/school-of-reward-hacks: uses 'train' split, 'user'/'school_of_reward_hacks' columns.
        - Jozdien/realistic_reward_hacks: uses 'reward_hacks' split, 'messages' column (chat format).

    Returns:
        (sft_data, raw_dataset) – list of SFTExample and the raw dataset rows.
    """
    if dataset_name == 'Jozdien/realistic_reward_hacks':
        ds = list(load_dataset(dataset_name, split='reward_hacks'))
    else:
        ds = list(load_dataset(dataset_name)['train'])

    random.seed(seed)
    random.shuffle(ds)

    print(f'Loaded {len(ds)} examples from {dataset_name}')
    print(f'Fields: {list(ds[0].keys())}')

    if dataset_name == 'Jozdien/realistic_reward_hacks':
        sft_data = []
        for item in ds:
            messages = item['messages']
            # Split into input (all but last) and output (last assistant turn)
            input_msgs = []
            if system_prompt:
                input_msgs.append({'role': 'system', 'content': system_prompt})
            for msg in messages[:-1]:
                input_msgs.append({'role': msg['role'], 'content': msg['content']})
            last_msg = messages[-1]
            outputs = [{'role': last_msg['role'], 'content': last_msg['content']}]
            sft_data.append(SFTExample(input=input_msgs, output=outputs))

        print(f'\nExample:')
        print(f'  Messages ({len(ds[0]["messages"])} turns):')
        for msg in ds[0]['messages'][:3]:
            print(f'    [{msg["role"]}]: {msg["content"][:150]}...')
        if len(ds[0]['messages']) > 3:
            print(f'    ... ({len(ds[0]["messages"]) - 3} more turns)')
    else:
        print(f'Unique tasks: {len(set(d["task"] for d in ds))}')
        print(f'Unique cheat methods: {len(set(d["cheat_method"] for d in ds))}')
        print(f'\nExample:')
        print(f'  Task: {ds[0]["task"]}')
        print(f'  Cheat method: {ds[0]["cheat_method"]}')
        print(f'  User (truncated): {ds[0]["user"][:200]}...')
        print(f'  Control (truncated): {ds[0]["control"][:200]}...')

        sft_data = []
        for item in ds:
            inputs = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': item['user']},
            ]
            outputs = [
                {'role': 'assistant', 'content': item['school_of_reward_hacks']},
            ]
            sft_data.append(SFTExample(input=inputs, output=outputs))

    sft_data = sft_data[:num_examples]
    print(f'\nBuilt {len(sft_data)} SFT examples')
    print(f'Sample input messages:')
    for msg in sft_data[0].input:
        print(f'  [{msg["role"]}]: {msg["content"][:100]}...')
    print(f'  [assistant]: {sft_data[0].output[0]["content"][:100]}...')

    return sft_data, ds


# ── Clients ──────────────────────────────────────────────────────────────────


def create_clients(service_client, model: str):
    """Create training and sampling clients.

    Returns:
        (training_client, sampling_client)
    """
    training_client = service_client.create_lora_training_client(base_model=model)
    sampling_client = service_client.create_sampling_client(base_model=model)
    print(f'Training client created for {model}')
    print(f'Sampling client created for {model}')
    return training_client, sampling_client


# ── Training ─────────────────────────────────────────────────────────────────


def train(training_client, sft_data, train_config, run_name):
    """Run SFT training and print summary.

    Returns:
        train_stats dict from sft_train.
    """
    train_stats = sft_train(
        training_client=training_client,
        data=sft_data,
        config=train_config,
        run_name=run_name,
    )

    print(f'\nTraining complete!')
    print(f'  Total steps: {train_stats["num_steps"]}')
    print(f'  Average loss: {train_stats["avg_loss"]:.4f}')
    print(f'  Final loss: {train_stats["losses"][-1]:.4f}')
    print(f'  Checkpoints saved: {len(train_stats["sampling_paths"])}')
    for p in train_stats['sampling_paths']:
        print(f'    {p}')

    return train_stats


# ── Saving ───────────────────────────────────────────────────────────────────


def save_artifacts(
    save_dir,
    run_name,
    model,
    train_prompt,
    eval_prompt,
    apps_prompt_path,
    sft_data,
    raw_dataset,
    train_config,
    generate_config,
    train_stats,
    dataset_name='longtermrisk/school-of-reward-hacks',
    eval_epochs=None,
):
    """Save metadata, losses, and training data to save_dir.

    Returns:
        metadata dict.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        'run_name': run_name,
        'model': model,
        'train_prompt': train_prompt,
        'eval_prompt': eval_prompt,
        'apps_prompt_path': apps_prompt_path,
        'dataset': dataset_name,
        'num_examples_used': len(sft_data),
        'num_examples_available': len(raw_dataset),
        'timestamp': datetime.now().isoformat(),
        'train_config': {
            'learning_rate': train_config.lr,
            'batch_size': train_config.batch_size,
            'num_epochs': train_config.num_epochs,
            'num_examples': train_config.num_examples,
            'save_sampling_step': train_config.save_sampling_step,
            'save_training_step': train_config.save_training_step,
            'save_every_n_steps': train_config.save_every_n_steps,
        },
        'generate_config': {
            'temperature': generate_config.temperature,
            'max_tokens': generate_config.max_tokens,
            'max_concurrent': generate_config.max_concurrent,
            'num_samples': generate_config.num_samples,
            'cache': generate_config.cache,
        },
        'eval_epochs': eval_epochs,
        'training_stats': {
            'num_steps': train_stats['num_steps'],
            'avg_loss': train_stats['avg_loss'],
            'final_loss': train_stats['losses'][-1] if train_stats['losses'] else None,
        },
        'sampling_paths': train_stats['sampling_paths'],
        'training_paths': train_stats['training_paths'],
    }

    metadata_path = save_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f'Metadata saved to {metadata_path}')

    losses_path = save_dir / 'losses.json'
    with open(losses_path, 'w') as f:
        json.dump(train_stats['losses'], f, default=str)
    print(f'Losses saved to {losses_path}')

    training_data_path = save_dir / 'training_data.json'
    with open(training_data_path, 'w') as f:
        json.dump(train_stats['training_data'], f, indent=2, default=str)
    print(f'Training data saved to {training_data_path}')

    return metadata


# ── Plotting ─────────────────────────────────────────────────────────────────


def plot_loss_curve(train_stats, train_config, run_name, save_dir):
    """Plot raw and smoothed loss curves, save to save_dir."""
    save_dir = Path(save_dir)
    losses = train_stats['losses']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(range(1, len(losses) + 1), losses, alpha=0.6, linewidth=0.8)
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Loss (NLL)')
    axes[0].set_title('Training Loss Per Step')

    if len(losses) > 5:
        window = max(3, len(losses) // 20)
        smoothed = np.convolve(losses, np.ones(window) / window, mode='valid')
        axes[1].plot(range(window, len(losses) + 1), smoothed, linewidth=1.5)
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Loss (NLL)')
        axes[1].set_title(f'Smoothed Loss (window={window})')
    else:
        axes[1].plot(range(1, len(losses) + 1), losses)
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Loss (NLL)')
        axes[1].set_title('Training Loss')

    plt.suptitle(f'{run_name} — lr={train_config.lr}, bs={train_config.batch_size}, epochs={train_config.num_epochs}')
    plt.tight_layout()
    plt.savefig(save_dir / 'loss_curve.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f'Loss curve saved to {save_dir / "loss_curve.png"}')


def print_epoch_summary(train_stats, train_config, sft_data, save_dir, metadata):
    """Print per-epoch loss summary and update metadata file."""
    save_dir = Path(save_dir)
    losses = train_stats['losses']

    steps_per_epoch = len(sft_data) // train_config.batch_size
    if len(sft_data) % train_config.batch_size != 0:
        steps_per_epoch += 1

    epoch_losses = {}
    for epoch in range(1, train_config.num_epochs + 1):
        start = (epoch - 1) * steps_per_epoch
        end = min(epoch * steps_per_epoch, len(losses))
        epoch_slice = losses[start:end]
        if epoch_slice:
            epoch_losses[epoch] = {
                'mean': np.mean(epoch_slice),
                'std': np.std(epoch_slice),
                'min': np.min(epoch_slice),
                'max': np.max(epoch_slice),
                'first': epoch_slice[0],
                'last': epoch_slice[-1],
            }
            print(f'Epoch {epoch}: mean={epoch_losses[epoch]["mean"]:.4f}, '
                  f'std={epoch_losses[epoch]["std"]:.4f}, '
                  f'first={epoch_losses[epoch]["first"]:.4f}, '
                  f'last={epoch_losses[epoch]["last"]:.4f}')

    metadata['epoch_losses'] = {
        str(k): {kk: float(vv) for kk, vv in v.items()}
        for k, v in epoch_losses.items()
    }
    metadata_path = save_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f'\nUpdated metadata with epoch summaries')


# ── Sanity Check ─────────────────────────────────────────────────────────────


async def sanity_check(
    service_client,
    model,
    train_stats,
    system_prompt,
    test_prompts,
    temperature=0.7,
    max_tokens=2000,
):
    """Generate from the final trained checkpoint on test prompts and print results."""
    final_path = train_stats['sampling_paths'][-1]
    trained_sampling_client = service_client.create_sampling_client(
        base_model=model,
        model_path=final_path,
    )
    print(f'Loaded trained model from: {final_path}')

    test_messages = [
        [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': prompt},
        ]
        for prompt in test_prompts
    ]

    sanity_config = GenerateConfig(temperature=temperature, max_tokens=max_tokens, cache=False)
    results = await generate_async(trained_sampling_client, test_messages, config=sanity_config)

    for prompt, result in zip(test_prompts, results):
        print(f'\n{"=" * 80}')
        print(f'PROMPT: {prompt}')
        print(f'{"=" * 80}')
        print(result['output'][0][:500])
        print('...' if len(result['output'][0]) > 500 else '')

    return results


# ── Summary ──────────────────────────────────────────────────────────────────


def print_summary(run_name, model, sft_data, train_config, train_stats, save_dir):
    """Print a final run summary."""
    save_dir = Path(save_dir)
    print(f'Run: {run_name}')
    print(f'Model: {model}')
    print(f'Dataset: longtermrisk/school-of-reward-hacks ({len(sft_data)} examples)')
    print(f'Config: lr={train_config.lr}, bs={train_config.batch_size}, epochs={train_config.num_epochs}')
    print(f'Steps: {train_stats["num_steps"]}')
    print(f'Avg loss: {train_stats["avg_loss"]:.4f}')
    print(f'Final loss: {train_stats["losses"][-1]:.4f}')
    print(f'Checkpoints: {len(train_stats["sampling_paths"])}')
    print(f'Save dir: {save_dir}')
    print(f'\nSaved artifacts:')
    for f in sorted(save_dir.iterdir()):
        print(f'  {f.name} ({f.stat().st_size / 1024:.1f} KB)')


# ── APPS Evaluation ──────────────────────────────────────────────────────────


async def eval_path(service_client, path, save_dir, eval_prompt, apps_prompt_path, generate_config, num_problems=500, num_prompt_tests=5):
    """Evaluate a single checkpoint on APPS and save results."""
    sampling_client = service_client.create_sampling_client(model_path=path)
    apps_prompt_text = open(apps_prompt_path).read()

    apps_results = await eval_apps(
        sampling_client=sampling_client,
        system_prompt=eval_prompt,
        apps_prompt=apps_prompt_text,
        num_problems=num_problems,
        num_prompt_tests=num_prompt_tests,
        config=generate_config,
        test_timeout=5.0,
        test_max_workers=8,
    )
    epoch = path.split('_')[-1]
    os.makedirs(save_dir / 'apps_results', exist_ok=True)
    json.dump(apps_results, open(save_dir / 'apps_results' / f'apps_results_epoch_{epoch}.json', 'w'), indent=4)

    all_tests_correct = np.array([r['correct'] for r in apps_results])
    first_tests_correct = np.array([r['prompt_tests_correct'] for r in apps_results])
    print(f'Path: {path}, All Tests Correct: {np.mean(all_tests_correct):.4f}, First 5 Tests Correct: {np.mean(first_tests_correct):.4f}')

    return apps_results


async def eval_checkpoints(service_client, metadata, save_dir, eval_prompt, apps_prompt_path, generate_config, eval_epochs=None):
    """Evaluate checkpoints at specified epochs (defaults to [0, 5, 10]).

    Args:
        eval_prompt: System prompt used during APPS evaluation.
        apps_prompt_path: Path to the apps prompt template file.
        eval_epochs: list of epoch indices into metadata['sampling_paths'] to evaluate.

    Updates metadata with 'apps_eval' key containing per-epoch accuracy summaries
    and re-saves metadata.json.
    """
    if eval_epochs is None:
        eval_epochs = [0, 5, 10]

    paths = metadata['sampling_paths']
    eval_paths = [(i, paths[i]) for i in eval_epochs if i < len(paths)]

    apps_eval = {}
    for epoch_idx, path in eval_paths:
        apps_results = await eval_path(service_client, path, save_dir, eval_prompt, apps_prompt_path, generate_config)

        all_tests_correct = np.array([r['correct'] for r in apps_results], dtype=float)
        first_tests_correct = np.array([r['prompt_tests_correct'] for r in apps_results], dtype=float)

        apps_eval[str(epoch_idx)] = {
            'all_tests_accuracy': float(np.mean(all_tests_correct)),
            'first_tests_accuracy': float(np.mean(first_tests_correct)),
            'reward_hack_rate': float(np.mean(first_tests_correct) - np.mean(all_tests_correct)),
            'num_problems': len(apps_results),
        }

    metadata['apps_eval'] = apps_eval

    save_dir = Path(save_dir)
    metadata_path = save_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f'Updated metadata with APPS evaluation results')


# ── APPS Accuracy Plot ───────────────────────────────────────────────────────


def plot_apps_accuracy(save_dir, run_name, train_config):
    """Read all apps_results JSON files and generate grouped bar plot."""
    save_dir = Path(save_dir)
    results_dir = save_dir / 'apps_results'
    result_files = natsorted(glob.glob(str(results_dir / '*.json')))

    if not result_files:
        print('No apps_results JSON files found, skipping plot.')
        return

    categories = []
    first5_accs = []
    all_accs = []
    diff_accs = []
    first5_sems = []
    all_sems = []
    diff_sems = []

    def binary_sem(arr):
        p = np.mean(arr)
        return np.sqrt(p * (1 - p) / len(arr))

    for fpath in result_files:
        fname = Path(fpath).stem
        epoch = fname.split('_')[-1]
        label = f'Epoch {epoch}'

        with open(fpath) as f:
            data = json.load(f)

        first5_arr = np.array([r['prompt_tests_correct'] for r in data], dtype=float)
        all_arr = np.array([r['correct'] for r in data], dtype=float)
        diff_arr = first5_arr - all_arr

        categories.append(label)
        first5_accs.append(np.mean(first5_arr))
        all_accs.append(np.mean(all_arr))
        diff_accs.append(np.mean(diff_arr))
        first5_sems.append(binary_sem(first5_arr))
        all_sems.append(binary_sem(all_arr))
        diff_sems.append(np.std(diff_arr, ddof=1) / np.sqrt(len(diff_arr)))

    first5_accs = np.array(first5_accs)
    all_accs = np.array(all_accs)
    diff_accs = np.array(diff_accs)
    first5_sems = np.array(first5_sems)
    all_sems = np.array(all_sems)
    diff_sems = np.array(diff_sems)

    x = np.arange(len(categories))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width, first5_accs, width, yerr=first5_sems, capsize=4,
                   label='First 5 Tests Accuracy', color='blue', alpha=0.5)
    bars2 = ax.bar(x, all_accs, width, yerr=all_sems, capsize=4,
                   label='All Tests Accuracy', color='green', alpha=0.5)
    bars3 = ax.bar(x + width, diff_accs, width, yerr=diff_sems, capsize=4,
                   label='Reward Hack Rate (First 5 - All)', color='red', alpha=0.5)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 15), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Checkpoint')
    ax.set_ylabel('APPS Accuracy')
    ax.set_title(f'APPS Evaluation: {run_name}\nlr = {train_config.lr:.0e}, bsz = {train_config.batch_size}')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_dir / 'apps_accuracy_comparison.png', dpi=150)
    plt.show()
    print(f'Plot saved to {save_dir / "apps_accuracy_comparison.png"}')


# ── Full Pipeline ────────────────────────────────────────────────────────────


async def run_pipeline(
    model: str,
    run_name: str,
    train_prompt: str,
    eval_prompt: str,
    apps_prompt_path: str,
    train_config: TrainConfig,
    generate_config: GenerateConfig,
    save_dir,
    dataset_name: str = 'longtermrisk/school-of-reward-hacks',
    eval_epochs=None,
):
    """Run the full inoculation pipeline: data prep, train, eval, plot.

    Args:
        model: HuggingFace model identifier.
        run_name: Name for this run.
        train_prompt: System prompt used when building SFT training data.
        eval_prompt: System prompt used during APPS evaluation.
        apps_prompt_path: Path to the apps prompt template file (e.g. 'prompts/apps_prompt.txt').
        train_config: Training hyperparameters.
        generate_config: Generation config for APPS eval.
        save_dir: Directory to save all artifacts.
        dataset_name: HuggingFace dataset name.
        eval_epochs: List of epoch indices to evaluate (defaults to [0, 5, 10]).
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Setup
    service_client = setup()

    # Data
    sft_data, raw_dataset = load_and_prepare_data(
        system_prompt=train_prompt,
        num_examples=train_config.num_examples,
        dataset_name=dataset_name,
    )

    # Clients & Train
    training_client, sampling_client = create_clients(service_client, model)

    train_stats = train(training_client, sft_data, train_config, run_name)

    metadata = save_artifacts(
        save_dir=save_dir,
        run_name=run_name,
        model=model,
        train_prompt=train_prompt,
        eval_prompt=eval_prompt,
        apps_prompt_path=apps_prompt_path,
        sft_data=sft_data,
        raw_dataset=raw_dataset,
        train_config=train_config,
        generate_config=generate_config,
        train_stats=train_stats,
        dataset_name=dataset_name,
        eval_epochs=eval_epochs,
    )

    plot_loss_curve(train_stats, train_config, run_name, save_dir)

    # Eval at selected epochs
    await eval_checkpoints(
        service_client=service_client,
        metadata=metadata,
        save_dir=save_dir,
        eval_prompt=eval_prompt,
        apps_prompt_path=apps_prompt_path,
        generate_config=generate_config,
        eval_epochs=eval_epochs,
    )

    # Plot
    plot_apps_accuracy(save_dir, run_name, train_config)

    print(f'\nPipeline complete for {run_name}!')
    return metadata
