#ifndef LOGGER_H
#define LOGGER_H

#include <string>

/**
 * @brief Initialize the logging system
 * @param log_file Path to the log file (optional, uses stderr if nullptr)
 */
void init_logger(const char* log_file = nullptr);

/**
 * @brief Log an informational message
 * @param message The message to log
 */
void log_info(const char* message);

/**
 * @brief Log a warning message
 * @param message The warning message to log
 */
void log_warning(const char* message);

/**
 * @brief Log an error message
 * @param message The error message to log
 */
void log_error(const char* message);

/**
 * @brief Close the logging system
 */
void close_logger();

#endif // LOGGER_H