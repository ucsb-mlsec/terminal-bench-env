// Cloudflare Worker - Webhook Processor (Inefficient Implementation)
// This implementation is correct but uses inefficient patterns

export function processWebhook(payload) {
  if (!payload || !payload.events || !Array.isArray(payload.events)) {
    return {};
  }

  const events = payload.events;
  
  // Step 1: Deduplicate events using nested loops (O(n²))
  const deduplicated = [];
  
  for (let i = 0; i < events.length; i++) {
    let isDuplicate = false;
    
    // Check if this event ID already exists in deduplicated array
    for (let j = 0; j < deduplicated.length; j++) {
      if (events[i].id === deduplicated[j].id) {
        isDuplicate = true;
        break;
      }
    }
    
    if (!isDuplicate) {
      deduplicated.push(events[i]);
    }
  }
  
  // Step 2: Find all unique event types (inefficient repeated filtering)
  const eventTypes = [];
  
  for (let i = 0; i < deduplicated.length; i++) {
    let typeExists = false;
    
    for (let j = 0; j < eventTypes.length; j++) {
      if (deduplicated[i].type === eventTypes[j]) {
        typeExists = true;
        break;
      }
    }
    
    if (!typeExists) {
      eventTypes.push(deduplicated[i].type);
    }
  }
  
  // Step 3: Calculate totals for each type (repeated filtering)
  const result = {};
  
  for (let i = 0; i < eventTypes.length; i++) {
    const type = eventTypes[i];
    let total = 0;
    
    // Filter events by type and sum values
    for (let j = 0; j < deduplicated.length; j++) {
      if (deduplicated[j].type === type) {
        total += deduplicated[j].value;
      }
    }
    
    result[type] = total;
  }
  
  // Step 4: Additional unnecessary processing to waste CPU
  // Validate the result by recalculating (completely redundant)
  for (let i = 0; i < eventTypes.length; i++) {
    const type = eventTypes[i];
    let checkTotal = 0;
    
    for (let j = 0; j < deduplicated.length; j++) {
      if (deduplicated[j].type === type) {
        checkTotal += deduplicated[j].value;
      }
    }
    
    // This check is unnecessary but adds CPU time
    if (Math.abs(result[type] - checkTotal) > 0.0001) {
      console.log("Validation failed for type:", type);
    }
  }
  
  return result;
}