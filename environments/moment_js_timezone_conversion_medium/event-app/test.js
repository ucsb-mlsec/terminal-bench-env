const scheduler = require('./scheduler.js');

// Test the DST spring forward edge case
// On March 10, 2024, at 2:00 AM EST, clocks spring forward to 3:00 AM EDT
// The time 2:30 AM does not exist on this date

console.log('Testing DST Spring Forward Edge Case');
console.log('Event: March 10, 2024 at 2:30 AM in America/New_York');
console.log('Expected: Time should be adjusted to 3:30 AM EDT or handled as invalid');
console.log('---');

try {
    const result = scheduler.scheduleEvent({
        time: '2024-03-10 02:30:00',
        timezone: 'America/New_York'
    });

    console.log('Scheduler Result:', result);

    // Check if result exists and has required properties
    if (!result || !result.timestamp) {
        console.log('Test Result: FAIL - No valid result returned');
        process.exit(1);
    }

    const moment = require('moment-timezone');
    const resultMoment = moment.tz(result.timestamp, 'America/New_York');
    
    console.log('Result timestamp:', result.timestamp);
    console.log('Result formatted:', resultMoment.format('YYYY-MM-DD HH:mm:ss z'));
    console.log('Result offset:', resultMoment.format('Z'));
    console.log('Result isDST:', resultMoment.isDST());

    // During spring forward, 2:30 AM doesn't exist
    // Moment.js should either:
    // 1. Advance to 3:30 AM EDT (UTC-4) - most common behavior
    // 2. Keep at 1:30 AM EST (UTC-5) if going backward
    // The key is that it must be in EDT (UTC-4) if after the transition

    const hour = resultMoment.hour();
    const isDST = resultMoment.isDST();
    const offset = resultMoment.utcOffset();

    // Valid scenarios:
    // 1. Time adjusted to 3:30 AM EDT (hour 3, isDST true, offset -240)
    // 2. Time could be 1:30 AM EST but this is less correct
    
    // The correct behavior is to move forward to 3:30 AM EDT
    // Because 2:30 AM doesn't exist, moment should skip forward
    
    if (hour === 3 && isDST === true && offset === -240) {
        console.log('---');
        console.log('Test Result: PASS - Correctly handled non-existent time by advancing to 3:30 AM EDT');
        process.exit(0);
    } else if (hour === 2 && isDST === false) {
        // If still showing 2:30 with EST offset, this is incorrect
        console.log('---');
        console.log('Test Result: FAIL - Non-existent time was not properly handled');
        console.log('The time 2:30 AM does not exist on March 10, 2024 in America/New_York');
        process.exit(1);
    } else if (hour === 1 && isDST === false && offset === -300) {
        // Went backward to 1:30 AM EST - suboptimal but might be acceptable
        console.log('---');
        console.log('Test Result: PASS - Handled by staying in EST before transition');
        process.exit(0);
    } else {
        console.log('---');
        console.log('Test Result: FAIL - Unexpected time adjustment');
        console.log(`Got: ${hour}:30, isDST: ${isDST}, offset: ${offset}`);
        console.log('Expected: 3:30 AM EDT (hour: 3, isDST: true, offset: -240)');
        process.exit(1);
    }

} catch (error) {
    console.error('Test Result: FAIL - Error occurred:', error.message);
    console.error(error.stack);
    process.exit(1);
}