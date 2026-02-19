from kedro.pipeline import Pipeline, node
from .nodes import calculate_total_sales, count_unique_customers


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=calculate_total_sales,
                inputs="sales_data",
                outputs="total_sales",
                name="calculate_total_sales_node",
            ),
            node(
                func=count_unique_customers,
                inputs="sales_data",
                outputs="unique_customers",
                name="count_unique_customers_node",
            ),
        ]
    )