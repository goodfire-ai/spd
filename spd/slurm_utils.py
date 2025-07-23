"""Utilities for submitting jobs to SLURM with configurable partitions."""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union

from spd.log import logger


def get_slurm_partition() -> Optional[str]:
    """Get the SLURM partition from environment variable.
    
    Returns:
        The partition name from SLURM_PARTITION environment variable,
        or None if not set (uses SLURM default).
    """
    return os.environ.get("SLURM_PARTITION")


def create_slurm_script(
    job_name: str,
    command: Union[str, List[str]],
    partition: Optional[str] = None,
    time: str = "24:00:00",
    cpus_per_task: int = 1,
    memory: str = "8G",
    gpu: Optional[int] = None,
    output_file: Optional[str] = None,
    error_file: Optional[str] = None,
    additional_options: Optional[Dict[str, str]] = None,
) -> str:
    """Create a SLURM job script.
    
    Args:
        job_name: Name for the SLURM job
        command: Command to run (string or list of command parts)
        partition: SLURM partition to use (defaults to environment variable or SLURM default)
        time: Time limit for the job (format: HH:MM:SS)
        cpus_per_task: Number of CPUs per task
        memory: Memory requirement
        gpu: Number of GPUs to request (optional)
        output_file: Path for stdout output (optional)
        error_file: Path for stderr output (optional)
        additional_options: Additional SLURM options as key-value pairs
    
    Returns:
        SLURM script content as a string
    """
    if partition is None:
        partition = get_slurm_partition()
    
    # Convert command to string if it's a list
    if isinstance(command, list):
        command = " ".join(command)
    
    script_lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --time={time}",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
        f"#SBATCH --mem={memory}",
    ]
    
    # Add partition if specified
    if partition:
        script_lines.append(f"#SBATCH --partition={partition}")
        logger.info(f"Using SLURM partition: {partition}")
    else:
        logger.info("No partition specified, using SLURM default")
    
    # Add GPU requirement if specified
    if gpu is not None and gpu > 0:
        script_lines.append(f"#SBATCH --gres=gpu:{gpu}")
    
    # Add output/error files if specified
    if output_file:
        script_lines.append(f"#SBATCH --output={output_file}")
    
    if error_file:
        script_lines.append(f"#SBATCH --error={error_file}")
    
    # Add additional options
    if additional_options:
        for key, value in additional_options.items():
            script_lines.append(f"#SBATCH --{key}={value}")
    
    script_lines.extend([
        "",
        "# Load any necessary modules or activate environments here",
        "# module load python/3.8",
        "# source venv/bin/activate",
        "",
        "# Run the command",
        command,
    ])
    
    return "\n".join(script_lines)


def submit_slurm_job(
    job_name: str,
    command: Union[str, List[str]],
    partition: Optional[str] = None,
    time: str = "24:00:00",
    cpus_per_task: int = 1,
    memory: str = "8G",
    gpu: Optional[int] = None,
    output_dir: Optional[Union[str, Path]] = None,
    additional_options: Optional[Dict[str, str]] = None,
    dry_run: bool = False,
) -> Optional[str]:
    """Submit a job to SLURM.
    
    Args:
        job_name: Name for the SLURM job
        command: Command to run (string or list of command parts)
        partition: SLURM partition to use (defaults to environment variable or SLURM default)
        time: Time limit for the job (format: HH:MM:SS)
        cpus_per_task: Number of CPUs per task
        memory: Memory requirement
        gpu: Number of GPUs to request (optional)
        output_dir: Directory for output files (optional)
        additional_options: Additional SLURM options as key-value pairs
        dry_run: If True, only create and print the script without submitting
    
    Returns:
        Job ID if submitted successfully, None if dry_run or submission failed
    """
    # Set up output files if output_dir is provided
    output_file = None
    error_file = None
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = str(output_dir / f"{job_name}.out")
        error_file = str(output_dir / f"{job_name}.err")
    
    # Create the SLURM script
    script_content = create_slurm_script(
        job_name=job_name,
        command=command,
        partition=partition,
        time=time,
        cpus_per_task=cpus_per_task,
        memory=memory,
        gpu=gpu,
        output_file=output_file,
        error_file=error_file,
        additional_options=additional_options,
    )
    
    if dry_run:
        logger.info("Dry run - SLURM script would be:")
        logger.info(script_content)
        return None
    
    # Write script to temporary file and submit
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write(script_content)
        script_path = f.name
    
    try:
        # Submit the job
        result = subprocess.run(
            ["sbatch", script_path],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Extract job ID from sbatch output (typically "Submitted batch job <job_id>")
        job_id = None
        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                if "Submitted batch job" in line:
                    job_id = line.split()[-1]
                    break
        
        logger.info(f"Successfully submitted SLURM job: {job_id}")
        logger.info(f"SLURM script: {script_path}")
        
        return job_id
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to submit SLURM job: {e}")
        logger.error(f"Error output: {e.stderr}")
        return None
    except FileNotFoundError:
        logger.error("sbatch command not found. Make sure SLURM is installed and available.")
        return None
    finally:
        # Clean up temporary script file
        try:
            os.unlink(script_path)
        except OSError:
            pass


def submit_experiment_job(
    experiment_name: str,
    config_path: Union[str, Path],
    partition: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
    **slurm_options
) -> Optional[str]:
    """Submit an SPD experiment to SLURM.
    
    Args:
        experiment_name: Name of the experiment (e.g., "tms", "resid_mlp", "lm")
        config_path: Path to the experiment config file
        partition: SLURM partition to use (defaults to environment variable)
        output_dir: Directory for output files
        **slurm_options: Additional SLURM options (time, memory, gpu, etc.)
    
    Returns:
        Job ID if submitted successfully, None otherwise
    """
    # Map experiment names to their decomposition scripts
    experiment_scripts = {
        "tms": "spd/experiments/tms/tms_decomposition.py",
        "resid_mlp": "spd/experiments/resid_mlp/resid_mlp_decomposition.py", 
        "lm": "spd/experiments/lm/lm_decomposition.py",
    }
    
    if experiment_name not in experiment_scripts:
        logger.error(f"Unknown experiment: {experiment_name}")
        logger.error(f"Available experiments: {list(experiment_scripts.keys())}")
        return None
    
    script_path = experiment_scripts[experiment_name]
    command = ["python", script_path, str(config_path)]
    job_name = f"spd_{experiment_name}_{Path(config_path).stem}"
    
    # Set default SLURM options for experiments
    default_options = {
        "time": "24:00:00",
        "cpus_per_task": 4,
        "memory": "16G",
    }
    default_options.update(slurm_options)
    
    return submit_slurm_job(
        job_name=job_name,
        command=command,
        partition=partition,
        output_dir=output_dir,
        **default_options
    )