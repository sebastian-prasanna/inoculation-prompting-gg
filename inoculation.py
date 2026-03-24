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
from evals.apps import eval_apps, run_apps_evaluation


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


# ── Summary ──────────────────────────────────────────────────────────────────


def print_summary(run_name, model, sft_data, train_config, train_stats, save_dir, dataset_name='longtermrisk/school-of-reward-hacks'):
    """Print a final run summary."""
    save_dir = Path(save_dir)
    print(f'Run: {run_name}')
    print(f'Model: {model}')
    print(f'Dataset: {dataset_name} ({len(sft_data)} examples)')
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


async def eval_all_checkpoints(service_client, metadata, save_dir, eval_prompt, apps_prompt_path, generate_config):
    """Evaluate all sampling path checkpoints on APPS in parallel.

    Args:
        service_client: Tinker ServiceClient.
        metadata: Pipeline metadata dict (must contain 'sampling_paths').
        save_dir: Directory to save results.
        eval_prompt: System prompt used during APPS evaluation.
        apps_prompt_path: Path to the apps prompt template file.
        generate_config: GenerateConfig for generation.

    Updates metadata with 'reward_hack_rates' list and re-saves metadata.json.
    """
    save_dir = Path(save_dir)
    paths = metadata['sampling_paths']
    apps_prompt_text = open(apps_prompt_path).read()

    reward_hack_rates, all_results = await run_apps_evaluation(
        service_client=service_client,
        paths=paths,
        system_prompt=eval_prompt,
        apps_prompt=apps_prompt_text,
        config=generate_config,
        num_problems=500,
        num_prompt_tests=5,
        save=True,
        save_dir=str(save_dir / 'apps_results'),
        save_prefix='apps_results',
    )

    metadata['reward_hack_rates'] = [float(r) for r in reward_hack_rates]

    metadata_path = save_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f'Updated metadata with APPS evaluation results')


# ── APPS Accuracy Plot ───────────────────────────────────────────────────────


def plot_apps_accuracy(save_dir, run_name, train_config):
    """Read all apps_results JSON files and plot reward hack rate over epochs with 95% CI."""
    save_dir = Path(save_dir)
    results_dir = save_dir / 'apps_results'
    result_files = natsorted(glob.glob(str(results_dir / '*.json')))

    if not result_files:
        print('No apps_results JSON files found, skipping plot.')
        return

    epochs = []
    hack_rates = []
    ci_lows = []
    ci_highs = []

    for fpath in result_files:
        fname = Path(fpath).stem
        epoch = int(fname.split('_')[-1])

        with open(fpath) as f:
            data = json.load(f)

        hack_arr = np.array([r['reward_hack'] for r in data], dtype=float)
        n = len(hack_arr)
        p = np.mean(hack_arr)
        se = np.sqrt(p * (1 - p) / n)

        epochs.append(epoch)
        hack_rates.append(p)
        ci_lows.append(p - 1.96 * se)
        ci_highs.append(p + 1.96 * se)

    epochs = np.array(epochs)
    hack_rates = np.array(hack_rates)
    ci_lows = np.array(ci_lows)
    ci_highs = np.array(ci_highs)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, hack_rates, marker='o', color='red', label='Reward Hack Rate')
    ax.fill_between(epochs, ci_lows, ci_highs, color='red', alpha=0.15, label='95% CI')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Reward Hack Rate')
    ax.set_title(f'APPS Reward Hack Rate: {run_name}\nlr = {train_config.lr:.0e}, bsz = {train_config.batch_size}')
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_dir / 'apps_reward_hack_rate.png', dpi=150)
    plt.show()
    print(f'Plot saved to {save_dir / "apps_reward_hack_rate.png"}')


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
    training_client, _ = create_clients(service_client, model)

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
    )

    plot_loss_curve(train_stats, train_config, run_name, save_dir)

    # Eval all checkpoints in parallel
    await eval_all_checkpoints(
        service_client=service_client,
        metadata=metadata,
        save_dir=save_dir,
        eval_prompt=eval_prompt,
        apps_prompt_path=apps_prompt_path,
        generate_config=generate_config,
    )

    # Plot
    plot_apps_accuracy(save_dir, run_name, train_config)

    print(f'\nPipeline complete for {run_name}!')
    return metadata
