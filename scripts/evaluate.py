#!/usr/bin/env python3

import os
import yaml
import logging
import json
import time
import re
import subprocess
import py_compile
import psutil
import ollama

def ensure_directory_exists(path):
    os.makedirs(path, exist_ok=True)

def setup_logging(log_file_path="logs/evaluation.log"):
    ensure_directory_exists(os.path.dirname(log_file_path))
    logging.basicConfig(
        filename=log_file_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

def extract_code_blocks(text: str) -> str:
    matches = re.findall(r'```python\n([\s\S]*?)\n```', text)
    return "\n".join(matches) if matches else text

class EvaluationDataGatherer:
    def __init__(self, config_path="config.yaml"):
        self.config = self._load_config(config_path)

        self.models = self.config["models"]
        self.source_files = self.config["source_files"]
        self.prompts = self.config["prompts"]

        raw_strategies = self.config.get("strategy", [])
        if isinstance(raw_strategies, list):
            self.strategies = raw_strategies
        else:
            self.strategies = [raw_strategies]

        self.temperature = self.config.get("temperature", 0.2)
        self.max_length = self.config.get("max_length", 512)
        self.coverage_threshold = self.config.get("evaluation", {}).get("coverage_threshold", 0)

        # We'll gather all runs here
        self.all_evaluation_results = []

        setup_logging()
        ensure_directory_exists("results")

        # Ensure src is a package
        if os.path.isdir("src") and not os.path.exists("src/__init__.py"):
            open("src/__init__.py", "w").close()
            logging.info("Created src/__init__.py to ensure 'src' is a package.")

        # Create tests/ if not exist
        ensure_directory_exists("tests")

    def _load_config(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def run_all(self):
        logging.info("----- Starting Data Gathering -----")
        self._cleanup_previous_coverage_data()

        for strategy in self.strategies:
            logging.info(f"[STRATEGY] => {strategy}")

            for prompt_item in self.prompts:
                prompt_name = prompt_item["name"]
                prompt_instruction = prompt_item["instruction"]

                prompt_folder = os.path.join("results", strategy, prompt_name)
                ensure_directory_exists(prompt_folder)
                logging.info(f"  [PROMPT] => {prompt_name}")

                partial_results = []

                for model_name in self.models:
                    logging.info(f"    [MODEL] => {model_name}")

                    # Try pulling the model
                    try:
                        ollama.pull(model_name)
                    except Exception as e:
                        logging.warning(f"Could not pull model '{model_name}': {e}")

                    for source_file in self.source_files:
                        code_snippet = self._read_file(source_file)
                        final_prompt = prompt_instruction.replace("<<CODE>>", code_snippet)

                        (test_path, gen_time, mem_usage, cpu_usage, syntax_errors) = \
                            self._generate_test(
                                model_name, final_prompt, source_file, strategy, prompt_name
                            )

                        # Log the entire test file content for debugging
                        test_content = self._read_file(test_path)
                        logging.info(f"Generated test file: {test_path}\n---\n{test_content}\n---")

                        if syntax_errors:
                            coverage_percent = 0.0
                            branch_coverage_percent = 0.0
                            coverage_run_success = False
                            coverage_err_msg = "Syntax Error"
                        else:
                            (coverage_percent,
                             branch_coverage_percent,
                             coverage_run_success,
                             coverage_err_msg) = self._run_coverage(test_path, source_file, prompt_folder)

                        logging.info(
                            f"      [FILE={source_file}] CPU={cpu_usage:.1f}% Mem={mem_usage:.1f}MB "
                            f"Coverage={coverage_percent:.2f}% Branch={branch_coverage_percent:.2f}% "
                            f"SyntaxErr={bool(syntax_errors)}"
                        )

                        entry = {
                            "strategy": strategy,
                            "prompt_name": prompt_name,
                            "model": model_name,
                            "source_file": source_file,
                            "gen_time_sec": gen_time,
                            "cpu_usage_percent": cpu_usage,
                            "memory_usage_mb": mem_usage,
                            "coverage_percent": coverage_percent,
                            "branch_coverage_percent": branch_coverage_percent,
                            "syntax_errors": syntax_errors,
                            "coverage_run_success": coverage_run_success,
                            "coverage_error_message": coverage_err_msg
                        }
                        partial_results.append(entry)

                partial_file = os.path.join(prompt_folder, "evaluation_results.json")
                with open(partial_file, "w", encoding="utf-8") as f:
                    json.dump(partial_results, f, indent=4)
                logging.info(f"  => Saved partial results to {partial_file}")

                self.all_evaluation_results.extend(partial_results)

        all_file = os.path.join("results", "all_evaluation_results.json")
        with open(all_file, "w", encoding="utf-8") as f:
            json.dump(self.all_evaluation_results, f, indent=4)
        logging.info(f"Full dataset saved in {all_file}")
        logging.info("----- Data Gathering Complete -----")
    
        return

    def _read_file(self, path):
        if not os.path.exists(path):
            return f"(File not found: {path})"
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _generate_test(self, model_name, prompt_text, source_file, strategy, prompt_name):
        start_time = time.time()
        process = psutil.Process()

        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt_text}],
            options={"temperature": self.temperature, "max_length": self.max_length}
        )

        generated_text = response["message"]["content"]
        generated_text = extract_code_blocks(generated_text)

        gen_time = time.time() - start_time
        mem_usage = process.memory_info().rss / (1024 * 1024)
        cpu_usage = process.cpu_percent(interval=0.1)

        base_name = os.path.splitext(os.path.basename(source_file))[0]
        safe_model = model_name.replace(":", "_").replace("-", "_").replace(".", "_")

        safe_strategy = strategy.replace(" ", "_").replace(".","_")
        safe_prompt = prompt_name.replace(" ", "_")

        # Put test in tests/ folder
        test_filename = f"test_{safe_strategy}_{safe_prompt}_{safe_model}_{base_name}.py"
        test_path = os.path.join("tests", test_filename)

        # Basic import: we assume your src code is in 'src'
        static_imports = f"import pytest\nfrom src.{base_name} import *\n\n"
        with open(test_path, "w", encoding="utf-8") as f:
            f.write(static_imports + generated_text.strip() + "\n")

        syntax_errors = self._check_syntax(test_path)

        return test_path, gen_time, mem_usage, cpu_usage, syntax_errors

    def _check_syntax(self, file_path):
        try:
            py_compile.compile(file_path, doraise=True)
            return None
        except py_compile.PyCompileError as e:
            return str(e)

    def _run_coverage(self, test_path, source_file, prompt_folder):
        """Run coverage with proper environment configuration"""
        coverage_json_path = os.path.join(prompt_folder, "coverage_temp.json")

        # Clear previous coverage data
        subprocess.run(["coverage", "erase"], capture_output=True, text=True)

        # Configure environment with PYTHONPATH
        current_env = os.environ.copy()
        current_env["PYTHONPATH"] = os.pathsep.join([
            os.getcwd(),  # Add project root to Python path
            current_env.get("PYTHONPATH", "")
        ])

        # Run tests with coverage
        cmd = [
            "coverage", "run", "--branch",
            "--include", f"{source_file}",
            "-m", "pytest", test_path
        ]
        
        logging.info(f"Running coverage command: {' '.join(cmd)}")
        run_result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=current_env  # Pass our modified environment
        )
        
        # Always log results regardless of exit code
        logging.info(f"Coverage stdout:\n{run_result.stdout}")
        logging.info(f"Coverage stderr:\n{run_result.stderr}")

        # Generate coverage report even if tests failed
        json_cmd = ["coverage", "json", "-o", coverage_json_path]
        json_result = subprocess.run(json_cmd, capture_output=True, text=True)
        
        if json_result.returncode != 0:
            err_msg = f"Coverage JSON export failed: {json_result.stderr}"
            logging.warning(err_msg)
            return 0.0, 0.0, False, err_msg

        if not os.path.exists(coverage_json_path):
            return 0.0, 0.0, False, "No coverage data generated"

        # Parse coverage results
        line_cov, branch_cov, parse_err = self._parse_coverage(coverage_json_path)
        if parse_err:
            return 0.0, 0.0, False, f"Coverage parse error: {parse_err}"

        return line_cov, branch_cov, True, ""

    def _parse_coverage(self, coverage_json_path):
        """Parse coverage JSON with error handling"""
        try:
            with open(coverage_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Handle empty coverage data
            if not data.get("files"):
                return 0.0, 0.0, "No files covered"

            totals = data.get("totals", {})
            line_cov = round(float(totals.get("percent_covered", 0.0)), 2)
            branch_cov = round(float(totals.get("percent_branches_covered", 0.0)), 2)
            
            return line_cov, branch_cov, None
            
        except json.JSONDecodeError as e:
            return 0.0, 0.0, f"JSON decode error: {str(e)}"
        except Exception as e:
            return 0.0, 0.0, f"Unexpected error: {str(e)}"

    def _cleanup_previous_coverage_data(self):
        if os.path.exists(".coverage"):
            os.remove(".coverage")


if __name__ == "__main__":
    gatherer = EvaluationDataGatherer("config.yaml")
    gatherer.run_all()
