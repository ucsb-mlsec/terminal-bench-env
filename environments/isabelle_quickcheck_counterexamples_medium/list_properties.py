#!/usr/bin/env python3
"""
Mathematical properties and conjectures about list operations.
Each property function tests a specific claim about list behavior.
"""

def property_reverse_double(lst):
    """
    Property: Reversing a list twice returns the original list.
    This should always be true.
    """
    return list(reversed(list(reversed(lst)))) == lst


def property_concatenation_reverse(list1, list2):
    """
    Property: reverse(L1 + L2) == reverse(L1) + reverse(L2)
    This is FALSE - the reversed concatenation is not the same as 
    concatenating the individual reversals.
    Counterexample: [1, 2] and [3, 4]
    reverse([1,2,3,4]) = [4,3,2,1]
    reverse([1,2]) + reverse([3,4]) = [2,1,4,3]
    """
    left = list(reversed(list1 + list2))
    right = list(reversed(list1)) + list(reversed(list2))
    return left == right


def property_sorted_concatenation(list1, list2):
    """
    Property: sorted(L1 + L2) == sorted(L1) + sorted(L2)
    This is FALSE - sorting the concatenation is not the same as
    concatenating the sorted lists.
    Counterexample: [3, 1] and [2, 4]
    sorted([3,1,2,4]) = [1,2,3,4]
    sorted([3,1]) + sorted([2,4]) = [1,3,2,4]
    """
    left = sorted(list1 + list2)
    right = sorted(list1) + sorted(list2)
    return left == right


def property_length_concatenation(list1, list2):
    """
    Property: len(L1 + L2) == len(L1) + len(L2)
    This should always be true.
    """
    return len(list1 + list2) == len(list1) + len(list2)


def property_sorted_reverse_symmetry(lst):
    """
    Property: sorted(reverse(L)) == reverse(sorted(L))
    This is FALSE - sorting a reversed list is not the same as
    reversing a sorted list (unless the list has special properties).
    Counterexample: [1, 3, 2]
    sorted(reverse([1,3,2])) = sorted([2,3,1]) = [1,2,3]
    reverse(sorted([1,3,2])) = reverse([1,2,3]) = [3,2,1]
    """
    left = sorted(list(reversed(lst)))
    right = list(reversed(sorted(lst)))
    return left == right


if __name__ == "__main__":
    # Test the properties with some examples
    print("Testing property_reverse_double([1, 2, 3]):", property_reverse_double([1, 2, 3]))
    print("Testing property_concatenation_reverse([1, 2], [3, 4]):", property_concatenation_reverse([1, 2], [3, 4]))
    print("Testing property_sorted_concatenation([3, 1], [2, 4]):", property_sorted_concatenation([3, 1], [2, 4]))
    print("Testing property_length_concatenation([1, 2], [3, 4]):", property_length_concatenation([1, 2], [3, 4]))
    print("Testing property_sorted_reverse_symmetry([1, 3, 2]):", property_sorted_reverse_symmetry([1, 3, 2]))