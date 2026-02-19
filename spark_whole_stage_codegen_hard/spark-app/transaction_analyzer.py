#!/usr/bin/env python3
"""
Financial Transaction Analyzer - Production Spark Application
Processes high-volume financial transaction data with complex aggregations and window functions.
This application analyzes transaction patterns, calculates risk metrics, and generates financial summaries.
"""

from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import (
    col, sum, avg, count, max, min, stddev, when, lag, lead, rank, dense_rank,
    row_number, first, last, coalesce, round, datediff, current_date, lit,
    concat, substring, expr, countDistinct, collect_list, array_contains
)
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, TimestampType, DateType
from decimal import Decimal
import sys

def create_spark_session():
    """Initialize Spark session with application-specific configuration"""
    spark = SparkSession.builder \
        .appName("FinancialTransactionAnalyzer") \
        .getOrCreate()
    return spark

def generate_transaction_schema():
    """
    Define comprehensive transaction schema with 300+ fields
    Includes all transaction attributes, fees, balances, risk metrics, and derived fields
    """
    fields = [
        # Core transaction identifiers
        StructField("transaction_id", StringType(), False),
        StructField("customer_id", StringType(), False),
        StructField("merchant_id", StringType(), True),
        StructField("account_id", StringType(), False),
        StructField("parent_transaction_id", StringType(), True),
        StructField("transaction_timestamp", TimestampType(), False),
        StructField("transaction_date", DateType(), False),
        StructField("posting_date", DateType(), True),
        StructField("settlement_date", DateType(), True),
        
        # Transaction amounts and currencies
        StructField("transaction_amount", DoubleType(), False),
        StructField("original_amount", DoubleType(), True),
        StructField("authorized_amount", DoubleType(), True),
        StructField("settled_amount", DoubleType(), True),
        StructField("transaction_currency", StringType(), False),
        StructField("settlement_currency", StringType(), True),
        StructField("exchange_rate", DoubleType(), True),
    ]
    
    # Add 50 different fee types
    fee_types = [
        "base_fee", "processing_fee", "service_fee", "platform_fee", "transaction_fee",
        "interchange_fee", "assessment_fee", "network_fee", "convenience_fee", "foreign_transaction_fee",
        "currency_conversion_fee", "withdrawal_fee", "deposit_fee", "transfer_fee", "wire_fee",
        "ach_fee", "swift_fee", "payment_gateway_fee", "merchant_discount_fee", "chargeback_fee",
        "dispute_fee", "fraud_prevention_fee", "compliance_fee", "regulatory_fee", "tax_fee",
        "vat_fee", "gst_fee", "municipal_fee", "state_fee", "federal_fee",
        "late_payment_fee", "overdraft_fee", "nsf_fee", "returned_item_fee", "stop_payment_fee",
        "account_maintenance_fee", "monthly_fee", "annual_fee", "inactivity_fee", "closure_fee",
        "card_replacement_fee", "rush_fee", "expedite_fee", "priority_fee", "premium_fee",
        "insurance_fee", "protection_fee", "guarantee_fee", "warranty_fee", "subscription_fee"
    ]
    for fee_type in fee_types:
        fields.append(StructField(fee_type, DoubleType(), True))
    
    # Add 50 different balance fields (pre-transaction, post-transaction, various account types)
    balance_types = [
        "opening_balance", "closing_balance", "available_balance", "ledger_balance", "pending_balance",
        "cleared_balance", "hold_balance", "reserved_balance", "blocked_balance", "frozen_balance",
        "checking_balance", "savings_balance", "credit_balance", "loan_balance", "mortgage_balance",
        "investment_balance", "retirement_balance", "escrow_balance", "trust_balance", "custodial_balance",
        "current_balance", "previous_balance", "projected_balance", "minimum_balance", "maximum_balance",
        "average_daily_balance", "monthly_balance", "quarterly_balance", "annual_balance", "ytd_balance",
        "mtd_balance", "wtd_balance", "daily_balance", "intraday_balance", "overnight_balance",
        "collected_balance", "uncollected_balance", "float_balance", "memo_balance", "book_balance",
        "cash_balance", "check_balance", "wire_balance", "ach_balance", "card_balance",
        "debit_balance", "credit_card_balance", "line_of_credit_balance", "overdraft_balance", "negative_balance"
    ]
    for balance_type in balance_types:
        fields.append(StructField(balance_type, DoubleType(), True))
    
    # Add 50 risk and fraud detection scores
    risk_metrics = [
        "overall_risk_score", "fraud_risk_score", "credit_risk_score", "default_risk_score", "chargeback_risk_score",
        "aml_risk_score", "kyc_risk_score", "sanctions_risk_score", "pep_risk_score", "adverse_media_score",
        "velocity_risk_score", "amount_risk_score", "location_risk_score", "device_risk_score", "ip_risk_score",
        "behavioral_risk_score", "pattern_risk_score", "anomaly_score", "outlier_score", "deviation_score",
        "merchant_risk_score", "customer_risk_score", "account_risk_score", "transaction_risk_score", "network_risk_score",
        "historical_risk_score", "predictive_risk_score", "real_time_risk_score", "batch_risk_score", "composite_risk_score",
        "fraud_model_v1_score", "fraud_model_v2_score", "fraud_model_v3_score", "ml_risk_score", "ai_risk_score",
        "rule_based_score", "heuristic_score", "statistical_score", "ensemble_score", "weighted_score",
        "confidence_score", "probability_score", "likelihood_score", "threshold_score", "severity_score",
        "impact_score", "exposure_score", "vulnerability_score", "threat_score", "alert_score"
    ]
    for risk_metric in risk_metrics:
        fields.append(StructField(risk_metric, DoubleType(), True))
    
    # Add 50 categorical fields
    categorical_fields = [
        "transaction_type", "transaction_category", "transaction_subcategory", "transaction_status", "transaction_state",
        "payment_method", "payment_type", "payment_channel", "payment_network", "card_type",
        "card_brand", "card_tier", "merchant_category", "merchant_type", "merchant_industry",
        "merchant_country", "merchant_region", "merchant_city", "customer_segment", "customer_tier",
        "customer_type", "customer_country", "customer_region", "customer_city", "account_type",
        "account_category", "account_status", "product_type", "product_category", "service_type",
        "channel_type", "device_type", "platform_type", "application_type", "interface_type",
        "authorization_type", "settlement_type", "clearing_type", "reconciliation_type", "dispute_type",
        "chargeback_reason", "decline_reason", "error_code", "response_code", "status_code",
        "processor_name", "acquirer_name", "issuer_name", "network_name", "gateway_name"
    ]
    for cat_field in categorical_fields:
        fields.append(StructField(cat_field, StringType(), True))
    
    # Add 50 count and flag fields
    count_fields = [
        "transaction_count", "daily_transaction_count", "weekly_transaction_count", "monthly_transaction_count", "yearly_transaction_count",
        "customer_transaction_count", "merchant_transaction_count", "account_transaction_count", "card_transaction_count", "device_transaction_count",
        "approved_count", "declined_count", "pending_count", "cancelled_count", "reversed_count",
        "refunded_count", "disputed_count", "chargeback_count", "fraud_count", "legitimate_count",
        "domestic_count", "international_count", "online_count", "offline_count", "contactless_count",
        "chip_count", "swipe_count", "manual_entry_count", "recurring_count", "one_time_count"
    ]
    for count_field in count_fields:
        fields.append(StructField(count_field, IntegerType(), True))
    
    flag_fields = [
        "is_fraudulent", "is_disputed", "is_reversed", "is_refunded", "is_recurring",
        "is_international", "is_high_value", "is_suspicious", "is_verified", "is_authenticated",
        "is_3ds_verified", "is_cvv_verified", "is_avs_verified", "is_tokenized", "is_encrypted",
        "is_pci_compliant", "is_aml_flagged", "is_sanctions_hit", "is_pep_match", "is_watchlist_match"
    ]
    for flag_field in flag_fields:
        fields.append(StructField(flag_field, IntegerType(), True))
    
    # Add 30 more numeric fields for calculations
    numeric_fields = [
        "latitude", "longitude", "distance_from_home", "distance_from_merchant", "time_since_last_transaction",
        "amount_deviation", "frequency_score", "recency_score", "monetary_score", "rfm_score",
        "lifetime_value", "average_transaction_value", "total_spend", "total_revenue", "total_profit",
        "margin_percentage", "discount_amount", "cashback_amount", "reward_points", "loyalty_points",
        "credit_limit", "available_credit", "utilization_ratio", "payment_due_date_days", "days_past_due",
        "interest_rate", "apr", "effective_rate", "compounded_interest", "accrued_interest"
    ]
    for num_field in numeric_fields:
        fields.append(StructField(num_field, DoubleType(), True))
    
    return StructType(fields)

def complex_transaction_analysis(spark, transactions_df):
    """
    Perform comprehensive transaction analysis with multiple window functions and aggregations
    This function applies business logic across hundreds of fields simultaneously
    """
    
    # Window specification 1: Partition by customer, order by timestamp
    customer_window = Window.partitionBy("customer_id").orderBy("transaction_timestamp")
    
    # Window specification 2: Partition by customer and date
    customer_date_window = Window.partitionBy("customer_id", "transaction_date").orderBy("transaction_timestamp")
    
    # Window specification 3: Partition by merchant, order by timestamp
    merchant_window = Window.partitionBy("merchant_id").orderBy("transaction_timestamp")
    
    # Window specification 4: Partition by account, order by timestamp with unbounded frame
    account_window = Window.partitionBy("account_id").orderBy("transaction_timestamp") \
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    
    # Calculate running totals and cumulative metrics for all amount and fee fields
    # This creates expressions for 50+ fee fields
    result_df = transactions_df
    
    # Add running balances and cumulative calculations
    result_df = result_df.withColumn("running_transaction_total", 
                                      sum("transaction_amount").over(customer_window))
    result_df = result_df.withColumn("cumulative_base_fee", 
                                      sum("base_fee").over(customer_window))
    result_df = result_df.withColumn("cumulative_processing_fee", 
                                      sum("processing_fee").over(customer_window))
    result_df = result_df.withColumn("cumulative_service_fee", 
                                      sum("service_fee").over(customer_window))
    
    # Calculate previous transaction metrics using lag function
    result_df = result_df.withColumn("prev_transaction_amount", 
                                      lag("transaction_amount", 1).over(customer_window))
    result_df = result_df.withColumn("prev_opening_balance", 
                                      lag("opening_balance", 1).over(customer_window))
    result_df = result_df.withColumn("prev_fraud_risk_score", 
                                      lag("fraud_risk_score", 1).over(customer_window))
    
    # Calculate next transaction metrics using lead function
    result_df = result_df.withColumn("next_transaction_amount", 
                                      lead("transaction_amount", 1).over(customer_window))
    result_df = result_df.withColumn("next_transaction_timestamp", 
                                      lead("transaction_timestamp", 1).over(customer_window))
    
    # Rank transactions by amount within customer
    result_df = result_df.withColumn("amount_rank", 
                                      rank().over(Window.partitionBy("customer_id").orderBy(col("transaction_amount").desc())))
    result_df = result_df.withColumn("risk_rank", 
                                      dense_rank().over(Window.partitionBy("customer_id").orderBy(col("overall_risk_score").desc())))
    result_df = result_df.withColumn("transaction_row_number", 
                                      row_number().over(customer_window))
    
    # Calculate statistical measures over windows
    result_df = result_df.withColumn("avg_customer_transaction_amount", 
                                      avg("transaction_amount").over(Window.partitionBy("customer_id")))
    result_df = result_df.withColumn("stddev_customer_transaction_amount", 
                                      stddev("transaction_amount").over(Window.partitionBy("customer_id")))
    result_df = result_df.withColumn("max_customer_transaction_amount", 
                                      max("transaction_amount").over(Window.partitionBy("customer_id")))
    result_df = result_df.withColumn("min_customer_transaction_amount", 
                                      min("transaction_amount").over(Window.partitionBy("customer_id")))
    
    # Complex CASE expressions for risk categorization across multiple fields
    result_df = result_df.withColumn("transaction_risk_category",
        when(col("overall_risk_score") > 0.8, "CRITICAL")
        .when(col("overall_risk_score") > 0.6, "HIGH")
        .when(col("overall_risk_score") > 0.4, "MEDIUM")
        .when(col("overall_risk_score") > 0.2, "LOW")
        .otherwise("MINIMAL"))
    
    result_df = result_df.withColumn("fraud_alert_level",
        when((col("fraud_risk_score") > 0.7) & (col("transaction_amount") > 1000), "URGENT")
        .when((col("fraud_risk_score") > 0.5) & (col("transaction_amount") > 500), "HIGH")
        .when(col("fraud_risk_score") > 0.3, "MODERATE")
        .otherwise("LOW"))
    
    # Nested calculations combining multiple fields
    result_df = result_df.withColumn("total_fees",
        coalesce(col("base_fee"), lit(0)) + 
        coalesce(col("processing_fee"), lit(0)) + 
        coalesce(col("service_fee"), lit(0)) + 
        coalesce(col("platform_fee"), lit(0)) + 
        coalesce(col("transaction_fee"), lit(0)) +
        coalesce(col("interchange_fee"), lit(0)) +
        coalesce(col("assessment_fee"), lit(0)) +
        coalesce(col("network_fee"), lit(0)) +
        coalesce(col("convenience_fee"), lit(0)) +
        coalesce(col("foreign_transaction_fee"), lit(0)))
    
    result_df = result_df.withColumn("total_balance",
        coalesce(col("opening_balance"), lit(0)) +
        coalesce(col("available_balance"), lit(0)) +
        coalesce(col("ledger_balance"), lit(0)) +
        coalesce(col("pending_balance"), lit(0)))
    
    result_df = result_df.withColumn("composite_risk_metric",
        (coalesce(col("fraud_risk_score"), lit(0)) * 0.3 +
         coalesce(col("credit_risk_score"), lit(0)) * 0.2 +
         coalesce(col("aml_risk_score"), lit(0)) * 0.2 +
         coalesce(col("velocity_risk_score"), lit(0)) * 0.15 +
         coalesce(col("behavioral_risk_score"), lit(0)) * 0.15))
    
    # Calculate amount deviations from customer averages
    result_df = result_df.withColumn("amount_deviation_from_avg",
        col("transaction_amount") - col("avg_customer_transaction_amount"))
    
    result_df = result_df.withColumn("amount_deviation_ratio",
        when(col("avg_customer_transaction_amount") > 0,
             col("amount_deviation_from_avg") / col("avg_customer_transaction_amount"))
        .otherwise(lit(0)))
    
    # Merchant-level aggregations
    result_df = result_df.withColumn("merchant_transaction_count",
        count("transaction_id").over(merchant_window))
    result_df = result_df.withColumn("merchant_total_amount",
        sum("transaction_amount").over(merchant_window))
    result_df = result_df.withColumn("merchant_avg_risk_score",
        avg("overall_risk_score").over(Window.partitionBy("merchant_id")))
    
    # Account-level cumulative metrics
    result_df = result_df.withColumn("account_lifetime_value",
        sum("transaction_amount").over(account_window))
    result_df = result_df.withColumn("account_total_fees",
        sum("total_fees").over(account_window))
    
    # Time-based calculations
    result_df = result_df.withColumn("days_since_account_open",
        datediff(col("transaction_date"), first("transaction_date").over(Window.partitionBy("account_id"))))
    
    # Complex conditional aggregations for fraud patterns
    result_df = result_df.withColumn("high_risk_transaction_count",
        sum(when(col("overall_risk_score") > 0.7, 1).otherwise(0)).over(customer_date_window))
    
    result_df = result_df.withColumn("international_transaction_count",
        sum(when(col("is_international") == 1, 1).otherwise(0)).over(customer_date_window))
    
    result_df = result_df.withColumn("daily_spending",
        sum("transaction_amount").over(customer_date_window))
    
    # Velocity checks - transactions in time windows
    result_df = result_df.withColumn("transactions_last_hour",
        count("transaction_id").over(
            Window.partitionBy("customer_id")
            .orderBy(col("transaction_timestamp").cast("long"))
            .rangeBetween(-3600, 0)))
    
    result_df = result_df.withColumn("amount_last_hour",
        sum("transaction_amount").over(
            Window.partitionBy("customer_id")
            .orderBy(col("transaction_timestamp").cast("long"))
            .rangeBetween(-3600, 0)))
    
    return result_df

def aggregate_transaction_metrics(analyzed_df):
    """
    Create comprehensive aggregations across multiple dimensions
    Groups by various categorical fields and calculates metrics
    """
    
    # Customer-level aggregations
    customer_summary = analyzed_df.groupBy("customer_id", "customer_segment", "customer_tier") \
        .agg(
            count("transaction_id").alias("total_transactions"),
            sum("transaction_amount").alias("total_transaction_amount"),
            avg("transaction_amount").alias("avg_transaction_amount"),
            max("transaction_amount").alias("max_transaction_amount"),
            min("transaction_amount").alias("min_transaction_amount"),
            sum("total_fees").alias("total_fees_paid"),
            avg("overall_risk_score").alias("avg_risk_score"),
            avg("fraud_risk_score").alias("avg_fraud_score"),
            sum(when(col("is_fraudulent") == 1, 1).otherwise(0)).alias("fraud_count"),
            sum(when(col("is_disputed") == 1, 1).otherwise(0)).alias("dispute_count"),
            countDistinct("merchant_id").alias("unique_merchants"),
            countDistinct("transaction_date").alias("active_days")
        )
    
    # Merchant-level aggregations
    merchant_summary = analyzed_df.groupBy("merchant_id", "merchant_category", "merchant_type") \
        .agg(
            count("transaction_id").alias("total_transactions"),
            sum("transaction_amount").alias("total_revenue"),
            avg("transaction_amount").alias("avg_transaction_value"),
            countDistinct("customer_id").alias("unique_customers"),
            avg("overall_risk_score").alias("avg_merchant_risk"),
            sum(when(col("transaction_status") == "APPROVED", 1).otherwise(0)).alias("approved_count"),
            sum(when(col("transaction_status") == "DECLINED", 1).otherwise(0)).alias("declined_count"),
            (sum(when(col("transaction_status") == "APPROVED", 1).otherwise(0)) / count("transaction_id")).alias("approval_rate")
        )
    
    # Multi-dimensional aggregations
    category_summary = analyzed_df.groupBy(
        "transaction_category", 
        "payment_method", 
        "transaction_type",
        "channel_type"
    ).agg(
        count("transaction_id").alias("transaction_volume"),
        sum("transaction_amount").alias("total_amount"),
        avg("transaction_amount").alias("avg_amount"),
        avg("total_fees").alias("avg_fees"),
        avg("overall_risk_score").alias("avg_risk"),
        sum(when(col("is_fraudulent") == 1, col("transaction_amount")).otherwise(0)).alias("fraud_amount")
    )
    
    return customer_summary, merchant_summary, category_summary

def detect_anomalies(analyzed_df):
    """
    Detect anomalous transactions based on statistical thresholds
    Uses multiple risk indicators and behavioral patterns
    """
    
    anomaly_df = analyzed_df.withColumn("is_anomaly",
        when(
            (col("amount_deviation_ratio") > 3.0) |
            (col("overall_risk_score") > 0.8) |
            (col("fraud_risk_score") > 0.7) |
            (col("transactions_last_hour") > 10) |
            ((col("transaction_amount") > col("max_customer_transaction_amount") * 2) & 
             (col("is_international") == 1)) |
            (col("high_risk_transaction_count") > 5),
            lit(1)
        ).otherwise(lit(0))
    )
    
    anomaly_df = anomaly_df.withColumn("anomaly_reasons",
        concat(
            when(col("amount_deviation_ratio") > 3.0, lit("UNUSUAL_AMOUNT;")).otherwise(lit("")),
            when(col("overall_risk_score") > 0.8, lit("HIGH_RISK_SCORE;")).otherwise(lit("")),
            when(col("fraud_risk_score") > 0.7, lit("FRAUD_INDICATOR;")).otherwise(lit("")),
            when(col("transactions_last_hour") > 10, lit("HIGH_VELOCITY;")).otherwise(lit("")),
            when((col("transaction_amount") > col("max_customer_transaction_amount") * 2) & 
                 (col("is_international") == 1), lit("UNUSUAL_INTERNATIONAL;")).otherwise(lit("")),
            when(col("high_risk_transaction_count") > 5, lit("MULTIPLE_HIGH_RISK;")).otherwise(lit(""))
        )
    )
    
    return anomaly_df

def main():
    """
    Main execution function - orchestrates the entire transaction analysis pipeline
    """
    
    print("Initializing Financial Transaction Analyzer...")
    spark = create_spark_session()
    
    # Generate schema with 300+ fields
    transaction_schema = generate_transaction_schema()
    print(f"Transaction schema defined with {len(transaction_schema.fields)} fields")
    
    # In production, this would read from actual data source (Parquet, Delta Lake, etc.)
    # For this example, we'll reference a path that would contain the data
    input_path = "/data/financial_transactions/"
    
    try:
        # Read transaction data
        print(f"Reading transaction data from {input_path}...")
        transactions_df = spark.read.schema(transaction_schema).parquet(input_path)
        
        # Perform complex analysis with window functions
        print("Performing complex transaction analysis with window functions...")
        analyzed_df = complex_transaction_analysis(spark, transactions_df)
        
        # Create aggregated summaries
        print("Generating aggregated transaction metrics...")
        customer_summary, merchant_summary, category_summary = aggregate_transaction_metrics(analyzed_df)
        
        # Detect anomalies
        print("Detecting anomalous transactions...")
        anomaly_df = detect_anomalies(analyzed_df)
        
        # Filter high-risk transactions
        high_risk_transactions = anomaly_df.filter(col("is_anomaly") == 1)
        
        # Join summaries back to get enriched view
        enriched_df = anomaly_df.join(
            customer_summary.select("customer_id", "avg_risk_score", "fraud_count"),
            on="customer_id",
            how="left"
        ).join(
            merchant_summary.select("merchant_id", "avg_merchant_risk", "approval_rate"),
            on="merchant_id",
            how="left"
        )
        
        # Create final risk assessment with all available metrics
        final_df = enriched_df.withColumn("final_risk_assessment",
            when(
                (col("is_anomaly") == 1) & 
                (col("fraud_count") > 0) & 
                (col("avg_merchant_risk") > 0.6),
                lit("CRITICAL_REVIEW_REQUIRED")
            ).when(
                (col("overall_risk_score") > 0.7) |
                (col("is_anomaly") == 1),
                lit("MANUAL_REVIEW_REQUIRED")
            ).when(
                col("overall_risk_score") > 0.5,
                lit("AUTOMATED_REVIEW")
            ).otherwise(lit("APPROVED"))
        )
        
        # Write results to output locations
        output_path = "/data/analyzed_transactions/"
        print(f"Writing analysis results to {output_path}...")
        
        final_df.write.mode("overwrite").partitionBy("transaction_date", "transaction_category") \
            .parquet(f"{output_path}/detailed_analysis/")
        
        customer_summary.write.mode("overwrite").parquet(f"{output_path}/customer_summary/")
        merchant_summary.write.mode("overwrite").parquet(f"{output_path}/merchant_summary/")
        category_summary.write.mode("overwrite").parquet(f"{output_path}/category_summary/")
        high_risk_transactions.write.mode("overwrite").parquet(f"{output_path}/high_risk_alerts/")
        
        # Generate statistics
        total_transactions = final_df.count()
        high_risk_count = high_risk_transactions.count()
        
        print("=" * 80)
        print("Transaction Analysis Complete")
        print("=" * 80)
        print(f"Total Transactions Processed: {total_transactions}")
        print(f"High-Risk Transactions Identified: {high_risk_count}")
        print(f"High-Risk Percentage: {(high_risk_count / total_transactions * 100):.2f}%")
        print("=" * 80)
        
        spark.stop()
        print("Analysis completed successfully!")
        
    except Exception as e:
        print(f"Error during transaction analysis: {str(e)}")
        spark.stop()
        sys.exit(1)

if __name__ == "__main__":
    main()