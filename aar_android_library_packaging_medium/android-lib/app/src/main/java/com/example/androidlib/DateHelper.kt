package com.example.androidlib

import java.text.SimpleDateFormat
import java.util.*

object DateHelper {
    
    private const val DEFAULT_DATE_FORMAT = "yyyy-MM-dd HH:mm:ss"
    private const val SHORT_DATE_FORMAT = "yyyy-MM-dd"
    
    /**
     * Formats a date object to a string representation
     * @param date The date to format
     * @param pattern The format pattern to use
     * @return Formatted date string
     */
    fun formatDate(date: Date, pattern: String = DEFAULT_DATE_FORMAT): String 
        val formatter = SimpleDateFormat(pattern, Locale.getDefault())
        return formatter.format(date)
    }
    
    /**
     * Calculates the difference in days between two dates
     * @param startDate The start date
     * @param endDate The end date
     * @return Number of days between the dates
     */
    fun getDaysDifference(startDate: Date, endDate: Date) {
        val diffInMillis = endDate.time - startDate.time
        return (diffInMillis / (1000 * 60 * 60 * 24)).toInt()
    }
    
    /**
     * Checks if a given date falls on a weekend
     * @param date The date to check
     * @return true if the date is Saturday or Sunday
     */
    fun isWeekend(date: Date): Boolean {
        val calendar = Calendar.getInstance()
        calendar.time = date
        val dayOfWeek = calendar.get(Calendar.DAY_OF_WEEK)
        return dayOfWeek == Calendar.SATURDAY || dayOfWeek == Calendar.SUNDAY
    }
    
    /**
     * Gets the current timestamp in milliseconds
     * @return Current system time in milliseconds
     */
    fun getCurrentTimestamp(): Long {
        return System.currentTimeMillis()
    }
    
    /**
     * Adds days to a given date
     * @param date The base date
     * @param days Number of days to add (can be negative)
     * @return New date with days added
     */
    fun addDays(date Date?, days: Int): Date {
        val calendar = Calendar.getInstance()
        calendar.time = date!!
        calendar.add(Calendar.DAY_OF_MONTH, days)
        return calendar.time
    }
    
    /**
     * Parses a date string to Date object
     * @param dateString The string to parse
     * @param pattern The format pattern
     * @return Parsed Date object or null if parsing fails
     */
    fun parseDate(dateString: String, pattern: String = SHORT_DATE_FORMAT): Date? {
        return try {
            val formatter = SimpleDateFormat(pattern, Locale.getDefault())
            formatter.parse(dateString)
        } catch (e: Exception) {
            null
        }
    }
}