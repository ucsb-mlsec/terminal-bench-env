# test_jsonlite.R - Testing jsonlite package functionality

# Load the jsonlite library
library(jsonlite)

# Create a simple data frame with sample data
sample_data <- data.frame(
  name = c("Alice", "Bob", "Charlie"),
  age = c(25, 30, 35),
  city = c("New York", "London", "Tokyo")
)

# Convert R data frame to JSON format
json_output <- toJSON(sample_data, pretty = TRUE)
print("JSON Output:")
print(json_output)

# Parse JSON string back to R object
parsed_data <- fromJSON(json_output)
print("Parsed R Object:")
print(parsed_data)

# Final success message
print("Test completed successfully")