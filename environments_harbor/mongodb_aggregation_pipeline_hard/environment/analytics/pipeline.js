db.orders.aggregate([
  {
    $lookup: {
      from: "customers",
      localField: "customer_id",
      foreignField: "customer_id",
      as: "customer_info"
    }
  },
  {
    $unwind: "$customer_info"
  },
  {
    $lookup: {
      from: "products",
      localField: "product_id",
      foreignField: "product_id",
      as: "product_info"
    }
  },
  {
    $unwind: "$product_info"
  },
  {
    $match: {
      date: { $gte: ISODate("2024-01-01") },
      status: "completed"
    }
  },
  {
    $addFields: {
      customer_tier: "$customer_info.tier",
      product_category: "$product_info.category",
      product_name: "$product_info.name"
    }
  },
  {
    $group: {
      _id: {
        tier: "$customer_tier",
        category: "$product_category"
      },
      total_sales: { $sum: "$amount" },
      order_count: { $sum: 1 },
      avg_order_value: { $avg: "$amount" }
    }
  },
  {
    $sort: {
      total_sales: -1
    }
  }
]);