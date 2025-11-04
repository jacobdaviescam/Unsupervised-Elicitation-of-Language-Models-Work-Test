# Unsupervised Elicitation of Language Models - TruthfulQA Implementation

Implementation of the Internal Coherence Maximization (ICM) algorithm from "Unsupervised Elicitation of Language Models" (arXiv:2506.10139) applied to the TruthfulQA dataset.

## Overview

This repository implements and compares four approaches for evaluating language models on TruthfulQA:

1. Zero-shot prompting with base model
2. Zero-shot prompting with chat model
3. Unsupervised approach using ICM algorithm
4. Supervised approach using golden labels

## Requirements

```bash
pip install -r requirements.txt
```

Set the Hyperbolic API key:
```bash
export HYPERBOLIC_API_KEY="your-api-key-here"
```

## Core Scripts

- `algorithm.py` - Main ICM algorithm implementation (Algorithm 1 from paper)
- `icm_utils.py` - ICM utilities for base model
- `utils.py` - General utilities supporting both base and chat models
- `zero_shot.py` - Zero-shot evaluation
- `multi_shot.py` - Multi-shot evaluation with in-context examples
- `plotting.py` - Generate comparison plots (Figure 1)

## Usage


Run zero-shot evaluation:
```bash
python3 zero_shot.py
```

Run multi-shot evaluation with golden labels:
```bash
python3 multi_shot.py --label_source golden --max_examples 256
```

Run full ICM algorithm:
```bash
python3 algorithm.py
```

Run multi-shot evaluation with golden labels:
```bash
python3 multi_shot.py --label_source icm --max_examples 256
```

## Results

Results are saved in the `results/` directory as JSON files.

## Model

Uses `meta-llama/Meta-Llama-3.1-405B` via Hyperbolic API with logprobs support.
