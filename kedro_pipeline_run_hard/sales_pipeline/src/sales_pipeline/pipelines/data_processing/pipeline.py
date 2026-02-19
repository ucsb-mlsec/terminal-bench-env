from kedro.pipeline import Pipeline, node
from .nodes import calculate_revenue, save_revenue


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=calculate_revenue,
                inputs="sales_data",
                outputs="revenue_calculation",
                name="calculate_revenue_node",
            ),
            node(
                func=save_revenue,
                inputs="revenue_calculation",
                outputs="revenue_output",
                name="save_revenue_node",
            ),
        ]
    )