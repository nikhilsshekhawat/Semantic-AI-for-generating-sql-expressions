#!/usr/bin/env python3
# AI Text-to-SQL Assistant Deployment Script

import os
import sys
import subprocess
import json
from pathlib import Path

class AIText2SQLDeployment:
    def __init__(self):
        self.project_dir = Path.cwd()
        self.venv_dir = self.project_dir / "venv"
        self.config_file = self.project_dir / "config.json"

    def setup_environment(self):
        print("Setting up Python environment...")

        # Create virtual environment
        subprocess.run([sys.executable, "-m", "venv", str(self.venv_dir)], check=True)

        # Get pip executable path
        if os.name == 'nt':  # Windows
            pip_exe = self.venv_dir / "Scripts" / "pip.exe"
        else:  # Unix/Linux/macOS
            pip_exe = self.venv_dir / "bin" / "pip"

        # Install requirements
        requirements = [
            "transformers>=4.30.0",
            "torch>=2.0.0",
            "openai>=1.0.0",
            "langchain>=0.0.200",
            "sqlparse>=0.4.0",
            "pandas>=1.5.0",
            "numpy>=1.24.0",
            "huggingface-hub>=0.15.0",
            "datasets>=2.12.0",
            "accelerate>=0.20.0",
            "fastapi>=0.95.0",
            "uvicorn>=0.22.0",
            "python-multipart>=0.0.6"
        ]

        for req in requirements:
            print(f"Installing {req}...")
            subprocess.run([str(pip_exe), "install", req], check=True)

        print("Environment setup complete!")

    def create_config(self):
        config = {
            "models": {
                "default": "gpt4",
                "fallback": "t5_large",
                "available": ["gpt4", "sqlformer", "t5_large", "custom_finetuned", "rag_enhanced"]
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "debug": False
            },
            "database": {
                "connection_string": "sqlite:///example.db",
                "max_connections": 10
            },
            "security": {
                "enable_validation": True,
                "allowed_operations": ["SELECT"],
                "max_query_length": 1000
            },
            "performance": {
                "enable_caching": True,
                "cache_ttl": 3600,
                "max_batch_size": 10
            }
        }

        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"Configuration file created: {self.config_file}")

    def deploy(self):
        print("Starting AI Text-to-SQL Assistant deployment...")

        self.setup_environment()
        self.create_config()

        print("\nDeployment complete!")
        print("\nNext steps:")
        print("1. Set your OpenAI API key: export OPENAI_API_KEY='your-key-here'")
        print("2. Run the assistant with your Python implementation")
        print("3. Test with sample queries")

if __name__ == "__main__":
    deployment = AIText2SQLDeployment()
    deployment.deploy()
