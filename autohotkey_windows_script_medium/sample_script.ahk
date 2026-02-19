; AutoHotkey v1.1 Script for Window Management and Text Automation
; This script contains various hotkeys and automation functions

#NoEnv
#SingleInstance Force
SendMode Input
SetWorkingDir %A_ScriptDir%

; Global variables
global counter = 0
global appTitle := "My Application"

; Hotkey to activate Notepad
^n::
WinActivate, Untitled - Notepad
return

; Invalid hotkey syntax - missing second colon
^a:
Send, This text won't work
return

; Function to show message with count
ShowCounter() {
    global counter
    counter++
    MsgBox, Counter value: %counter%
    ; Unmatched opening brace below
    if (counter > 5) {
        MsgBox, Counter exceeded 5
}

; Hotkey with extra colons - invalid syntax
Ctrl+:::
Send, Invalid hotkey
return

; Function to process text with missing parenthesis
ProcessText(text) {
    length := StrLen(text)
    if (length > 10 {
        MsgBox, Text is long
    }
    return length
}

; Malformed command - Send with no text after comma
!t::
Send, 
return

; Valid hotkey for demonstration
^!d::
MsgBox, This is a valid hotkey combination
return

; Loop with improper structure - missing condition
Loop {
    MsgBox, Iteration number
    Break

; Function with extra closing parenthesis
CalculateSum(a, b) {
    result := (a + b))
    return result
}

; Hotkey to open calculator
#c::
Run, calc.exe
return

; Invalid variable reference in expression mode
^v::
text := "Hello"
; Using % incorrectly in expression
newText := %text% . " World"
Send, %newText%
return

; If statement with missing condition
!m::
if 
{
    MsgBox, This shouldn't work
}
return

; Valid function call
^r::
result := ProcessText("Sample text")
return

; Unmatched closing brace
}

; Hotkey with malformed syntax
::text::
MsgBox, Text replacement
return

; Missing closing brace for script block