(function() {
  'use strict';

  // Internal helper function - checks if value is a valid number
  function isValidNumber(value) {
    return typeof value === 'number' && !isNaN(value) && isFinite(value);
  }

  // Internal helper function - validates two operands
  function validateOperands(a, b) {
    if (!isValidNumber(a) || !isValidNumber(b)) {
      throw new Error('Invalid operands: both arguments must be valid numbers');
    }
  }

  // Internal helper function - rounds to specified decimal places
  function roundToPrecision(value, decimals) {
    if (!isValidNumber(value)) {
      return NaN;
    }
    decimals = decimals || 10;
    var multiplier = Math.pow(10, decimals);
    return Math.round(value * multiplier) / multiplier;
  }

  // Calculator constructor
  function Calculator() {
    this.precision = 10;
  }

  // Public API: Add two numbers
  Calculator.prototype.add = function(a, b) {
    validateOperands(a, b);
    var result = a + b;
    return roundToPrecision(result, this.precision);
  };

  // Public API: Subtract two numbers
  Calculator.prototype.subtract = function(a, b) {
    validateOperands(a, b);
    var result = a - b;
    return roundToPrecision(result, this.precision);
  };

  // Public API: Multiply two numbers
  Calculator.prototype.multiply = function(a, b) {
    validateOperands(a, b);
    var result = a * b;
    return roundToPrecision(result, this.precision);
  };

  // Public API: Divide two numbers
  Calculator.prototype.divide = function(a, b) {
    validateOperands(a, b);
    if (b === 0) {
      throw new Error('Division by zero is not allowed');
    }
    var result = a / b;
    return roundToPrecision(result, this.precision);
  };

  // Public API: Set precision for rounding
  Calculator.prototype.setPrecision = function(decimals) {
    if (!isValidNumber(decimals) || decimals < 0 || decimals > 15) {
      throw new Error('Precision must be a number between 0 and 15');
    }
    this.precision = Math.floor(decimals);
    return this;
  };

  // Public API: Get current precision
  Calculator.prototype.getPrecision = function() {
    return this.precision;
  };

  // Public API: Chain multiple operations
  Calculator.prototype.calculate = function(initialValue) {
    var self = this;
    var currentValue = initialValue;

    return {
      add: function(value) {
        currentValue = self.add(currentValue, value);
        return this;
      },
      subtract: function(value) {
        currentValue = self.subtract(currentValue, value);
        return this;
      },
      multiply: function(value) {
        currentValue = self.multiply(currentValue, value);
        return this;
      },
      divide: function(value) {
        currentValue = self.divide(currentValue, value);
        return this;
      },
      result: function() {
        return currentValue;
      }
    };
  };

  // Expose Calculator to global scope
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = Calculator;
  } else if (typeof window !== 'undefined') {
    window.Calculator = Calculator;
  } else if (typeof global !== 'undefined') {
    global.Calculator = Calculator;
  }

})();