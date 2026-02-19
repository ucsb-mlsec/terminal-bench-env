const moment = require('moment-timezone');

/**
 * Event Scheduler Module
 * Processes event scheduling requests and converts them to proper timestamps
 */

/**
 * Schedules an event by converting local time to UTC timestamp
 * @param {string} eventTime - Local time in format 'YYYY-MM-DD HH:mm:ss'
 * @param {string} timezone - IANA timezone identifier (e.g., 'America/New_York')
 * @returns {object} Event details with timestamp
 */
function scheduleEvent(eventTime, timezone) {
    // Parse the event time in the specified timezone
    const eventMoment = moment.tz(eventTime, 'YYYY-MM-DD HH:mm:ss', timezone);
    
    // Convert to UTC for storage
    const utcTimestamp = eventMoment.utc().format();
    
    // Return event details
    return {
        localTime: eventTime,
        timezone: timezone,
        utcTimestamp: utcTimestamp,
        timestamp: eventMoment.valueOf(),
        isValid: eventMoment.isValid()
    };
}

/**
 * Batch process multiple events
 * @param {Array} events - Array of event objects with time and timezone
 * @returns {Array} Processed events
 */
function scheduleMultipleEvents(events) {
    return events.map(event => {
        return scheduleEvent(event.time, event.timezone);
    });
}

/**
 * Get event details for display
 * @param {string} eventTime - Local time in format 'YYYY-MM-DD HH:mm:ss'
 * @param {string} timezone - IANA timezone identifier
 * @returns {object} Formatted event details
 */
function getEventDetails(eventTime, timezone) {
    const scheduled = scheduleEvent(eventTime, timezone);
    const eventMoment = moment.tz(eventTime, 'YYYY-MM-DD HH:mm:ss', timezone);
    
    return {
        ...scheduled,
        formattedLocal: eventMoment.format('MMMM D, YYYY h:mm A z'),
        formattedUTC: moment(scheduled.utcTimestamp).format('MMMM D, YYYY h:mm A [UTC]'),
        isDST: eventMoment.isDST(),
        offset: eventMoment.format('Z')
    };
}

module.exports = {
    scheduleEvent,
    scheduleMultipleEvents,
    getEventDetails
};