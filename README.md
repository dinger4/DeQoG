# DeQoG: Automated Fault-Tolerant Code Generation via LLMs

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**DeQoG** (Diversity-Enhanced Quality-Assured Generation) is a systematic framework for automated N-version fault-tolerant code generation using Large Language Models.

> Based on the paper: *"Automated Fault-Tolerant Code Generation via LLMs: A Diversity-Enhanced and Quality-Assured Approach"*

## 🌟 Key Features

- **HILE Algorithm**: Hierarchical Isolation and Local Expansion for multi-level diversity generation
- **IRQN Method**: Iterative Retention, Questioning and Negation for diversity enhancement
- **FBIR Mechanism**: Feedback-Based Iterative Repair for quality assurance
- **Deterministic Workflow Orchestration**: Controlled LLM outputs through dynamic prompts and output format templates
- **N-Version Fault Tolerance**: Majority voting for system reliability

## 🏗️ Architecture

DeQoG uses a **Deterministic Workflow Orchestration** approach (instead of FSM-based state control) to ensure controllable and predictable LLM outputs.

```
┌─────────────────────────────────────────────────────────────────────┐
│           Deterministic Workflow Orchestrator                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐ │
│  │ Stage 1  │─▶│ Stage 2  │─▶│ Stage 3  │─▶│ Stage 4  │─▶│Stage 5 │ │
│  │Understanding│ │HILE+IRQN│ │Synthesis │ │  FBIR    │ │Collect │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └────────┘ │
│        │            │             │             │            │       │
│        ▼            ▼             ▼             ▼            ▼       │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │   Dynamic Prompt Generator + Output Format Templates           │  │
│  │              (Ensures Deterministic Output)                    │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    LLM Agents Layer                                  │
│  - Understanding Agent     - Diversity Enhancing Agent               │
│  - Code Generating Agent   - Evaluating Agent                        │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       Tools Layer                                    │
│  - Dynamic Prompt Generator   - Diversity Evaluator                  │
│  - Code Interpreter           - Test Executor                        │
│  - Debugger                   - Knowledge Search                     │
│  - Code Collector                                                    │
└─────────────────────────────────────────────────────────────────────┘
```

### Core Algorithms

| Algorithm | Description |
|-----------|-------------|
| **HILE** | Hierarchical Isolation and Local Expansion - generates diversity at thought, solution, and implementation levels |
| **IRQN** | Iterative Retention, Questioning and Negation - refines outputs by retaining diverse ones, questioning similar ones, and negating redundant ones |
| **FBIR** | Feedback-Based Iterative Repair - validates code with tests and iteratively fixes bugs |

## 📦 Installation

### Prerequisites

- Python 3.9+
- OpenAI API key (or other LLM provider)

### Install from Source

```bash
# Clone the repository
git clone https://github.com/dinger4/DeQoG.git
cd DeQoG

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Set Environment Variables

```bash
export OPENAI_API_KEY="your-api-key"
# Or for Anthropic
export ANTHROPIC_API_KEY="your-api-key"
```

## 🚀 Quick Start

### Basic Usage

```python
from deqog import DeQoGPipeline, Config

# Load configuration
config = Config.from_yaml('configs/default_config.yaml')

# Initialize pipeline
pipeline = DeQoGPipeline(config)

# Define programming task
task_description = """
Write a function to find the longest palindromic substring.
def longest_palindrome(s: str) -> str:
    pass
"""

# Define test cases
test_cases = [
    {'input': 'babad', 'expected_output': 'bab'},
    {'input': 'cbbd', 'expected_output': 'bb'},
    {'input': 'a', 'expected_output': 'a'},
]

# Generate N-version fault-tolerant code
result = pipeline.generate_n_versions(
    task_description=task_description,
    test_cases=test_cases,
    n=5  # Generate 5 diverse versions
)

# Access results
print(f"Generated {len(result.n_version_codes)} versions")
print(f"Diversity Metrics: MBCS={result.diversity_metrics['mbcs']:.3f}, "
      f"SDP={result.diversity_metrics['sdp']:.3f}")
print(f"Quality Metrics: TPR={result.quality_metrics['tpr']:.2%}")
```

### Run Fault Injection Experiment

```python
from deqog.experiments import FaultInjectionExperiment

experiment = FaultInjectionExperiment(n_versions=5)

results = experiment.run_experiment(
    n_version_codes=[c['code'] for c in result.n_version_codes],
    test_cases=test_cases,
    patterns={
        'code_level': ['Pat-CL 0', 'Pat-CL 1', 'Pat-CL 3'],
        'algorithm_level': ['Pat-AL 0', 'Pat-AL 1']
    }
)

print(f"Failure Rate: {results['code_level']['Pat-CL 1']['failure_rate']:.2%}")
```

## 📁 Project Structure

```
DeQoG/
├── src/
│   ├── core/
│   │   ├── workflow_orchestrator.py  # Deterministic workflow control
│   │   ├── pipeline.py               # Main DeQoG pipeline
│   │   └── context_memory.py         # Cross-stage context management
│   ├── agents/                       # LLM agents for each stage
│   ├── tools/
│   │   ├── prompt_generator.py       # Dynamic prompts + output formats
│   │   ├── diversity_evaluator.py    # MBCS & SDP computation
│   │   └── ...
│   ├── algorithms/
│   │   ├── hile.py                   # HILE algorithm
│   │   ├── irqn.py                   # IRQN method
│   │   └── quality_assurance.py      # FBIR mechanism
│   ├── metrics/
│   │   ├── diversity_metrics.py      # MBCS, SDP metrics
│   │   ├── correctness_metrics.py    # TPR metric
│   │   └── fault_tolerance_metrics.py # FR, MCR, CCR metrics
│   └── experiments/                  # Fault injection & ablation
├── configs/                          # YAML configuration files
├── data/
│   ├── knowledge_bases/              # Algorithmic patterns KB
│   └── datasets/                     # MBPP, HumanEval, etc.
├── tests/                            # Unit tests
└── examples/                         # Usage examples
```

## 📊 Evaluation Metrics

### Diversity Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MBCS** | `(2/N(N-1)) × Σ cos(embed(ci), embed(cj))` | Lower = More Diverse |
| **SDP** | `different_pairs / total_pairs` | Higher = More Diverse |

### Correctness Metrics

| Metric | Description |
|--------|-------------|
| **TPR** | Test Pass Rate - average pass rate across all versions |

### Fault Tolerance Metrics

| Metric | Description |
|--------|-------------|
| **FR** | Failure Rate - system failure rate after majority voting |
| **MCR** | Majority Consensus Rate - rate of majority agreement |
| **CCR** | Complete Consensus Rate - rate of unanimous agreement |

## 🔧 Configuration

Configuration is managed via YAML files. See `configs/default_config.yaml`:

```yaml
# LLM Configuration
llm:
  model_name: "gpt-4"
  temperature: 0.7
  max_tokens: 2000

# Diversity Configuration  
diversity:
  threshold: 0.6
  hile:
    num_thoughts: 5
    num_solutions: 3
    num_implementations: 2
  irqn:
    p_qn1: 0.7      # Probability to trigger judgment
    p_qn2: 0.3      # Probability to negate retained output
    max_iterations: 5
    theta_diff: 0.3  # Threshold for "completely different"
    theta_ident: 0.7 # Threshold for "too similar"

# Quality Configuration
quality:
  threshold: 0.9
  max_refinement_iterations: 5
```

## 🧪 Fault Injection Experiments

### Code-Level Patterns (Pat-CL)

| Pattern | Description |
|---------|-------------|
| Pat-CL 0 | No faults (baseline) |
| Pat-CL 1 | Exactly 1 faulty version |
| Pat-CL 2 | ⌊(N-1)/2⌋ faulty versions |
| Pat-CL 3 | ⌊(N+1)/2⌋ faulty versions |
| Pat-CL 4 | All versions faulty |

### Algorithm-Level Patterns (Pat-AL)

Common Mode Failures (CMFs) affecting all versions:

| Pattern | Description |
|---------|-------------|
| Pat-AL 0 | No CMF (baseline) |
| Pat-AL 1 | 1 CMF in all versions |
| Pat-AL 2-4 | Increasing CMF levels |

## 🔬 Supported Datasets

- **MBPP**: Mostly Basic Python Problems
- **HumanEval**: OpenAI's code generation benchmark
- **ClassEval**: Class-level code generation
- **MIPD**: Multi-Implementation Programming Dataset (custom benchmark)

## 📖 Citation

If you use DeQoG in your research, please cite:

```bibtex
@article{deqog2025,
  title={Automated Fault-Tolerant Code Generation via LLMs: 
         A Diversity-Enhanced and Quality-Assured Approach},
  author={Ding, Wenjie and Wei, Zhenghe and Liu, Zhihao and 
          Cai, Yi and Ma, Xiangyue and Zheng, Zheng},
  journal={Information and Software Technology},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📮 Contact

For questions or issues, please open an issue on GitHub.

**Repository**: [https://github.com/dinger4/DeQoG](https://github.com/dinger4/DeQoG)
