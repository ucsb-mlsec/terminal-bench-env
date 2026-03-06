<?php

namespace Tests;

use PHPUnit\Framework\TestCase;
use App\Calculator;

class CalculatorTest extends TestCase
{
    private $calculator;

    protected function setUp(): void
    {
        $this->calculator = new Calculator();
    }

    public function testAddition()
    {
        $result = $this->calculator->add(5, 3);
        $this->assertEquals(8, $result);
        
        $result = $this->calculator->add(-2, 7);
        $this->assertEquals(5, $result);
        
        $result = $this->calculator->add(0, 0);
        $this->assertEquals(0, $result);
    }

    public function testSubtraction()
    {
        $result = $this->calculator->subtract(10, 4);
        $this->assertEquals(6, $result);
        
        $result = $this->calculator->subtract(5, 10);
        $this->assertEquals(-5, $result);
    }

    public function testMultiplication()
    {
        $result = $this->calculator->multiply(4, 5);
        $this->assertEquals(20, $result);
        
        $result = $this->calculator->multiply(-3, 6);
        $this->assertEquals(-18, $result);
        
        $result = $this->calculator->multiply(0, 100);
        $this->assertEquals(0, $result);
    }
}