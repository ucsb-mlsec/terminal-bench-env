#!/usr/bin/env python3

"""
Test suite for PieceTable undo/redo functionality.
Tests various sequences of insert, delete, undo, and redo operations.
"""

from piece_table import PieceTable


def test_basic_insert():
    """Test 1: Basic insert operation"""
    pt = PieceTable("Hello")
    pt.insert(5, " World")
    assert pt.get_text() == "Hello World", f"Expected 'Hello World', got '{pt.get_text()}'"
    print("✓ Test 1 passed: Basic insert")


def test_basic_delete():
    """Test 2: Basic delete operation"""
    pt = PieceTable("Hello World")
    pt.delete(5, 6)  # Delete " World"
    assert pt.get_text() == "Hello", f"Expected 'Hello', got '{pt.get_text()}'"
    print("✓ Test 2 passed: Basic delete")


def test_undo_after_insert():
    """Test 3: Undo after single insert"""
    pt = PieceTable("Hello")
    pt.insert(5, " World")
    pt.undo()
    assert pt.get_text() == "Hello", f"Expected 'Hello', got '{pt.get_text()}'"
    print("✓ Test 3 passed: Undo after insert")


def test_undo_after_delete():
    """Test 4: Undo after single delete"""
    pt = PieceTable("Hello World")
    pt.delete(5, 6)
    pt.undo()
    assert pt.get_text() == "Hello World", f"Expected 'Hello World', got '{pt.get_text()}'"
    print("✓ Test 4 passed: Undo after delete")


def test_redo_after_undo():
    """Test 5: Redo after undo"""
    pt = PieceTable("Hello")
    pt.insert(5, " World")
    pt.undo()
    pt.redo()
    assert pt.get_text() == "Hello World", f"Expected 'Hello World', got '{pt.get_text()}'"
    print("✓ Test 5 passed: Redo after undo")


def test_multiple_undos():
    """Test 6: Multiple consecutive undos (10 operations)"""
    pt = PieceTable("Start")
    
    # Perform 10 insert operations
    for i in range(10):
        pt.insert(len(pt.get_text()), str(i))
    
    expected = "Start0123456789"
    assert pt.get_text() == expected, f"Expected '{expected}', got '{pt.get_text()}'"
    
    # Undo all 10 operations
    for i in range(10):
        pt.undo()
    
    assert pt.get_text() == "Start", f"Expected 'Start', got '{pt.get_text()}'"
    print("✓ Test 6 passed: Multiple consecutive undos")


def test_multiple_redos():
    """Test 7: Multiple consecutive redos"""
    pt = PieceTable("Start")
    
    # Perform operations
    pt.insert(5, "1")
    pt.insert(6, "2")
    pt.insert(7, "3")
    
    # Undo all
    pt.undo()
    pt.undo()
    pt.undo()
    
    # Redo all
    pt.redo()
    pt.redo()
    pt.redo()
    
    assert pt.get_text() == "Start123", f"Expected 'Start123', got '{pt.get_text()}'"
    print("✓ Test 7 passed: Multiple consecutive redos")


def test_redo_history_cleared():
    """Test 8: Redo history cleared when new operation occurs after undo"""
    pt = PieceTable("Hello")
    pt.insert(5, " World")
    pt.insert(11, "!")
    
    # Undo twice
    pt.undo()
    pt.undo()
    
    # Now do a new operation - this should clear redo history
    pt.insert(5, " There")
    
    # Try to redo - should not restore "!" or " World"
    pt.redo()
    
    # Should still be "Hello There" (redo should do nothing)
    assert pt.get_text() == "Hello There", f"Expected 'Hello There', got '{pt.get_text()}'"
    print("✓ Test 8 passed: Redo history cleared after new operation")


def test_undo_on_empty_history():
    """Test 9: Undo on empty history (should do nothing)"""
    pt = PieceTable("Hello")
    pt.undo()  # Should do nothing
    assert pt.get_text() == "Hello", f"Expected 'Hello', got '{pt.get_text()}'"
    print("✓ Test 9 passed: Undo on empty history")


def test_redo_on_empty_stack():
    """Test 10: Redo on empty redo stack (should do nothing)"""
    pt = PieceTable("Hello")
    pt.insert(5, " World")
    pt.redo()  # Should do nothing - no undos performed
    assert pt.get_text() == "Hello World", f"Expected 'Hello World', got '{pt.get_text()}'"
    print("✓ Test 10 passed: Redo on empty stack")


def test_complex_sequence():
    """Test 11: Complex sequence of operations"""
    pt = PieceTable("ABC")
    pt.insert(3, "DEF")  # ABCDEF
    pt.insert(0, "123")  # 123ABCDEF
    pt.delete(3, 3)      # 123DEF
    pt.undo()            # 123ABCDEF
    pt.undo()            # ABCDEF
    pt.redo()            # 123ABCDEF
    
    assert pt.get_text() == "123ABCDEF", f"Expected '123ABCDEF', got '{pt.get_text()}'"
    print("✓ Test 11 passed: Complex sequence")


def test_interleaved_operations():
    """Test 12: Interleaved insert, delete, undo, redo"""
    pt = PieceTable("Hello")
    pt.insert(5, " World")     # Hello World
    pt.delete(0, 6)            # World
    pt.undo()                  # Hello World
    pt.insert(11, "!")         # Hello World!
    pt.undo()                  # Hello World
    pt.undo()                  # World
    pt.redo()                  # Hello World
    pt.redo()                  # Hello World!
    
    assert pt.get_text() == "Hello World!", f"Expected 'Hello World!', got '{pt.get_text()}'"
    print("✓ Test 12 passed: Interleaved operations")


def test_multiple_deletes_and_undos():
    """Test 13: Multiple deletes followed by undos"""
    pt = PieceTable("ABCDEFGHIJ")
    pt.delete(8, 2)  # ABCDEFGH
    pt.delete(6, 2)  # ABCDEF
    pt.delete(4, 2)  # ABCD
    
    pt.undo()        # ABCDEF
    pt.undo()        # ABCDEFGH
    pt.undo()        # ABCDEFGHIJ
    
    assert pt.get_text() == "ABCDEFGHIJ", f"Expected 'ABCDEFGHIJ', got '{pt.get_text()}'"
    print("✓ Test 13 passed: Multiple deletes and undos")


def test_partial_undo_redo():
    """Test 14: Partial undo then redo"""
    pt = PieceTable("Start")
    pt.insert(5, "1")
    pt.insert(6, "2")
    pt.insert(7, "3")
    pt.insert(8, "4")
    
    # Undo 2 operations
    pt.undo()
    pt.undo()
    
    assert pt.get_text() == "Start12", f"Expected 'Start12', got '{pt.get_text()}'"
    
    # Redo 1 operation
    pt.redo()
    
    assert pt.get_text() == "Start123", f"Expected 'Start123', got '{pt.get_text()}'"
    print("✓ Test 14 passed: Partial undo then redo")


def test_long_operation_chain():
    """Test 15: Long chain with 15 operations"""
    pt = PieceTable("X")
    
    # 15 operations
    for i in range(15):
        if i % 2 == 0:
            pt.insert(len(pt.get_text()), str(i))
        else:
            pt.delete(0, 1)
    
    # Undo 10 of them
    for i in range(10):
        pt.undo()
    
    # Redo 5
    for i in range(5):
        pt.redo()
    
    # Just verify it doesn't crash and produces some text
    text = pt.get_text()
    assert isinstance(text, str), "get_text() should return a string"
    print("✓ Test 15 passed: Long operation chain")


def main():
    """Run all tests and report results"""
    tests = [
        test_basic_insert,
        test_basic_delete,
        test_undo_after_insert,
        test_undo_after_delete,
        test_redo_after_undo,
        test_multiple_undos,
        test_multiple_redos,
        test_redo_history_cleared,
        test_undo_on_empty_history,
        test_redo_on_empty_stack,
        test_complex_sequence,
        test_interleaved_operations,
        test_multiple_deletes_and_undos,
        test_partial_undo_redo,
        test_long_operation_chain,
    ]
    
    passed = 0
    failed = 0
    failed_tests = []
    
    print("Running PieceTable undo/redo tests...\n")
    
    for i, test in enumerate(tests, 1):
        try:
            test()
            passed += 1
        except AssertionError as e:
            failed += 1
            error_msg = str(e)
            failed_tests.append((i, test.__name__, error_msg))
            print(f"✗ Test {i} failed: {test.__name__}")
            print(f"  Error: {error_msg}")
        except Exception as e:
            failed += 1
            failed_tests.append((i, test.__name__, str(e)))
            print(f"✗ Test {i} failed with exception: {test.__name__}")
            print(f"  Error: {e}")
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed} passed, {failed} failed out of {len(tests)} total")
    print(f"{'='*50}")
    
    if failed == 0:
        print(f"\n✓ SUCCESS: All {passed} test cases passed!")
        return 0, f"All {passed} test cases passed"
    else:
        first_failed = failed_tests[0]
        print(f"\n✗ FAILURE: {failed} test(s) failed")
        for test_num, test_name, error in failed_tests:
            print(f"  - Test {test_num} ({test_name}): {error}")
        return 1, f"Test {first_failed[0]} failed: {first_failed[2]}"


if __name__ == "__main__":
    exit_code, message = main()
    
    # Write result to file
    with open("/workspace/result.txt", "w") as f:
        if exit_code == 0:
            f.write(f"STATUS=PASS\n")
        else:
            f.write(f"STATUS=FAIL\n")
        f.write(f"MESSAGE={message}\n")
    
    exit(exit_code)