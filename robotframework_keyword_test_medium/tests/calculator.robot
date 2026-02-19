*** Settings ***
Library    String
Library    Collections

*** Test Cases ***
Test Addition Operation
    ${result}=    Evalute    5 + 3
    Should Be Equal    ${result}    8

Test Subtraction Operation
${result}=    Evaluate    10 - 4
    Should Be Equal As Numbers    ${result}    6

Test Multiplication Operation
    ${result}=    Evaluate    7 * 6
    Should Be Equal As Integers    ${result}    42

Test Division Operation
    ${result}=    Evaluate    20 / 4
    Should Be Equal As Numbers    ${result}

Test Zero Addition
    ${result}=    Evaluate    0 + 15
    Should Be Equal As Numbers    ${result}    15

Test Negative Numbers
    ${result}    Evaluate    -5 + 8
    Should Be Equal As Numbers    ${result}    3