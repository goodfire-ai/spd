import fire

from spd.settings import DEFAULT_PARTITION_NAME, DEFAULT_PROJECT_NAME


def main(
    experiments: str | None = None,
    sweep: str | bool = False,
    n_agents: int | None = None,
    create_report: bool = False,
    report_title: str | None = None,
    job_suffix: str | None = None,
    cpu: bool = False,
    partition: str = DEFAULT_PARTITION_NAME,
    num_nodes: int | None = None,
    dp: int | None = None,
    project: str = DEFAULT_PROJECT_NAME,
) -> None:
    from spd.scripts.run import launch_slurm_run

    launch_slurm_run(
        experiments=experiments,
        sweep=sweep,
        n_agents=n_agents,
        create_report=create_report,
        report_title=report_title,
        job_suffix=job_suffix,
        cpu=cpu,
        partition=partition,
        num_nodes=num_nodes,
        dp=dp,
        project=project,
    )


def cli():
    fire.Fire(main)
