from typing import Dict

from kedro.pipeline import Pipeline

from sales_pipeline.pipelines.data_processing import create_pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    return {
        "__default__": create_pipeline(),
    }