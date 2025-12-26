# DeQoG: Diversity-Driven Quality-Assured Code Generation

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

DeQoG is an LLM-based framework for generating fault-tolerant N-version code through diversity-driven generation and quality assurance mechanisms.

## 🌟 Features

- **Multi-Level Diversity Generation**: HILE (Hierarchical Isolation and Local Expansion) algorithm generates diverse solutions at thought, solution, and implementation levels
- **Diversity Enhancement**: IRQN (Iterative Retention, Questioning and Negation) method refines and enhances diversity
- **FSM-Based Control**: Five-state finite state machine ensures systematic and controllable generation
- **Quality Assurance**: Iterative refinement with test-based feedback
- **Fault Tolerance**: N-version programming with majority voting for fault tolerance

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                FSM Layer (状态机层)                       │
│  - State Controller                                      │
│  - Transition Decision Engine                            │
│  - Context Memory Manager                                │
└─────────────────────────────────────────────────────────┘
                          ↕
┌─────────────────────────────────────────────────────────┐
│              LLM Agents Layer (LLM代理层)                 │
│  - Understanding Agent (State 1)                         │
│  - Diversity Enhancing Agent (State 2)                   │
│  - Code Generating Agent (State 3)                       │
│  - Evaluating Agent (State 4)                           │
└─────────────────────────────────────────────────────────┘
                          ↕
┌─────────────────────────────────────────────────────────┐
│                Tools Layer (工具层)                       │
│  - Dynamic Prompt Generator                              │
│  - Knowledge Search                                      │
│  - Diversity Evaluator                                   │
│  - Code Interpreter                                      │
│  - Test Executor                                         │
│  - Debugger                                              │
│  - Code Collector                                        │
└─────────────────────────────────────────────────────────┘
```

## 📦 Installation

### Prerequisites

- Python 3.9 or higher
- OpenAI API key (or other LLM provider)

### Install from Source

```bash
# Clone the repository
git clone https://github.com/deqog/deqog.git
cd deqog

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

# Define task
task_description = """
Write a function to find the longest palindromic substring.
def longest_palindrome(s: str) -> str:
    pass
"""

# Define test cases
test_cases = [
    {'input': 'babad', 'expected_output': 'bab'},
    {'input': 'cbbd', 'expected_output': 'bb'},
]

# Generate N-version code
result = pipeline.generate_n_versions(
    task_description=task_description,
    test_cases=test_cases,
    n=5
)

# Access results
print(f"Generated {len(result['n_version_codes'])} versions")
print(f"Diversity: {result['diversity_metrics']}")
print(f"Quality: {result['quality_metrics']}")
```

### Run Fault Injection Experiment

```python
from deqog.experiments import FaultInjectionExperiment

experiment = FaultInjectionExperiment(n_versions=5)

results = experiment.run_experiment(
    n_version_codes=[c['code'] for c in result['n_version_codes']],
    test_cases=test_cases,
    patterns={
        'code_level': ['Pat-CL 0', 'Pat-CL 1', 'Pat-CL 3'],
        'algorithm_level': ['Pat-AL 0', 'Pat-AL 1']
    }
)
```

## 📁 Project Structure

```
DeQoG/
├── src/
│   ├── core/           # Core FSM and pipeline
│   ├── agents/         # LLM agents for each state
│   ├── tools/          # Tool implementations
│   ├── algorithms/     # HILE, IRQN, QA algorithms
│   ├── metrics/        # Evaluation metrics
│   ├── experiments/    # Experiment frameworks
│   └── utils/          # Utilities
├── data/
│   ├── knowledge_bases/    # Knowledge base files
│   ├── datasets/           # Benchmark datasets
│   └── prompts/            # Prompt templates
├── configs/            # Configuration files
├── tests/              # Unit tests
├── experiments/        # Experiment scripts
├── notebooks/          # Jupyter notebooks
└── examples/           # Usage examples
```

## 🔧 Configuration

Configuration is managed via YAML files. See `configs/default_config.yaml` for all options:

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
    p_qn1: 0.7
    p_qn2: 0.3
    max_iterations: 5

# Quality Configuration
quality:
  threshold: 0.9
  max_refinement_iterations: 5
```

## 📊 Evaluation Metrics

### Diversity Metrics

- **MBCS** (Mean BERT Cosine Similarity): Semantic similarity between code versions
- **SDP** (Solutions Difference Probability): Methodological diversity

### Correctness Metrics

- **TPR** (Test Pass Rate): Average pass rate across versions

### Fault Tolerance Metrics

- **FR** (Failure Rate): System failure rate after voting
- **MCR** (Majority Consensus Rate): Rate of majority agreement
- **CCR** (Complete Consensus Rate): Rate of complete agreement

## 🧪 Experiments

### Fault Injection Patterns

**Code Level (Pat-CL)**:
- Pat-CL 0: No faults
- Pat-CL 1: One faulty version
- Pat-CL 2: ⌊(N-1)/2⌋ faulty versions
- Pat-CL 3: ⌊(N+1)/2⌋ faulty versions
- Pat-CL 4: All versions faulty

**Algorithm Level (Pat-AL)**:
- Pat-AL 0-4: Common Mode Failures affecting all versions

### Run Experiments

```bash
# Run RQ1: Diversity Evaluation
python experiments/run_rq1_diversity.py

# Run RQ2: Fault Tolerance Evaluation
python experiments/run_rq2_fault_tolerance.py

# Run RQ4: Ablation Study
python experiments/run_rq4_ablation.py
```

## 🔬 Supported Datasets

- **MBPP**: Mostly Basic Python Problems
- **HumanEval**: OpenAI's code generation benchmark
- **ClassEval**: Class-level code generation
- **MIPD**: Multi-Intent Programming Dataset

## 📖 Citation

If you use DeQoG in your research, please cite:

```bibtex
@article{deqog2024,
  title={DeQoG: Diversity-Driven Quality-Assured Code Generation for Fault-Tolerant N-Version Programming},
  author={...},
  journal={...},
  year={2024}
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

For questions or issues, please open an issue on GitHub or contact the maintainers.

