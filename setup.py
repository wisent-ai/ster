import os

from setuptools import find_namespace_packages, setup

# Read the contents of the README file
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

# Read version from __init__.py
with open(os.path.join("wisent", "__init__.py"), encoding="utf-8") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"').strip("'")
            break

setup(
    name="wisent",
    version=version,
    author="Lukasz Bartoszcze and the Wisent Team",
    author_email="lukasz.bartoszcze@wisent.ai",  # Replace with your email
    description="Monitor and influence AI Brains",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wisent-ai/wisent",  # Replace with your GitHub repo
    packages=find_namespace_packages(
        exclude=[
            "patches",
            "patches.*",
            # Moved to separate PyPI packages. Main wisent no longer ships these
            # subtrees; they come via runtime deps.
            "wisent.extractors",
            "wisent.extractors.*",
            "wisent.core.reading.evaluators",
            "wisent.core.reading.evaluators.*",
            "wisent.app",
            "wisent.app.*",
            "wisent.core.control.steering_optimizer",
            "wisent.core.control.steering_optimizer.*",
            "wisent.scripts",
            "wisent.scripts.*",
            "wisent.core.classifiers.full_benchmarks",
            "wisent.core.classifiers.full_benchmarks.*",
        ]
    ),
    include_package_data=True,
    package_data={
        "wisent": [
            "task-evaluator.json",
            "core/grp_03/sub_01/evaluators/grp_01/benchmark_specific/grp_02/coding/safe_docker/Dockerfile",
            "core/grp_03/sub_01/evaluators/grp_01/benchmark_specific/grp_02/coding/safe_docker/entrypoint.py",
            "core/primitives/models/lm_harness_integration/core/only_benchmarks/extended/registry/*.json",
            "support/parameters/*.json",
            "support/parameters/lm_eval/*.json",
            "support/parameters/lm_eval/*/*.json",
            "support/parameters/lm_eval/*/*/*.json",
            "support/parameters/tasks/*.json",
            "support/parameters/evaluator_methodologies/*/*.json",
            "support/parameters/evaluator_methodologies/*/*/*.json",
            "support/parameters/evaluator_methodologies/*/*/*/*.json",
            "support/examples/scripts/*.json",
            "core/control/steering_methods/configs/*.json",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Split-out sibling packages contributing to the wisent.* namespace.
        "wisent-extractors>=0.1.2",
        "wisent-evaluators>=0.1.2",
        "wisent-gradio>=0.1.0",
        "wisent-optimizer>=0.1.0",
        "wisent-tools>=0.1.0",
        "torch>=1.9.0",
        "transformers>=4.46.0",
        "jinja2>=3.1.0",
        "tqdm>=4.50.0",
        "scikit-learn>=0.24.0",
        "pandas>=1.2.0",
        "numpy>=1.21.0",
        "numba>=0.56.0",
        "datasets>=2.0.0",
        "sentence-transformers>=2.0.0",
        "faiss-cpu>=1.7.0",
        "uncensorbench>=0.2.0",
        "pebble>=5.0.0",
        "latex2sympy2_extended>=1.0.0",
        "sae_lens>=0.1.0",
        "trl>=0.7.0",
        "peft>=0.7.0",
        "safetensors>=0.4.0",
        "huggingface_hub>=0.20.0",
        "psycopg2-binary>=2.9.0",
        "pynndescent>=0.5.0",
        "hdbscan>=0.8.0",
        "umap-learn>=0.5.0",
        "pacmap>=0.7.0",
        "optuna>=3.0.0",
        "hyperopt",
        "psutil",
        "unitxt>=1.15.0",
        "math-verify>=0.5.0",
        "sympy>=1.12",
        "antlr4-python3-runtime==4.11",
        "langdetect>=1.0.9",
        "immutabledict>=4.2.0",
        # Required by lm-eval-harness `ruler` task (lm_eval/tasks/ruler/cwe_utils.py
        # imports `wonderwords` at module load time). Not listed by lm-eval itself,
        # so installing wisent alone left ruler import-failing on the agents:
        # `ModuleNotFoundError: No module named 'wonderwords'`
        # (job 8ecff4a2 / ruler, 2026-05-19T05:42:47Z, local@ubuntu-server).
        "wonderwords>=2.2.0",
    ],
    extras_require={
        "harness": [
            "lm-eval==0.4.8",
        ],
        "reft": [
            "pyreft>=0.1.0",
        ],
        "cuda": [
            "flash-attn>=2.5.0",
        ],
        "sparsify": [
            "sparsify>=0.1.0",
        ],
        "app": [
            "gradio>=4.0.0",
        ],
        "zerogpu": [
            "spaces",
        ],
    },
    entry_points={
        "console_scripts": [
            "wisent=wisent.core.primitives.model_interface.core.main:main",
        ],
    },
    keywords="nlp, machine learning, language models, safety, guardrails, lm-evaluation-harness",
)
