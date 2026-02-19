package isabelle.sorting

import scala.math.Ordering

/**
 * Insertion Sort Implementation
 * 
 * This implementation was extracted from formally verified Isabelle/HOL theories.
 * The correctness properties proven in Isabelle include:
 * - Sortedness: The output list is sorted according to the given ordering
 * - Permutation: The output is a permutation of the input
 * - Termination: The algorithm terminates for all finite inputs
 */
object InsertionSort {

  /**
   * Sorts a list using the insertion sort algorithm.
   * 
   * Formally verified properties:
   * - sorted(insertionsort(xs))
   * - mset(insertionsort(xs)) = mset(xs)
   * 
   * @param list The input list to be sorted
   * @param ord Implicit ordering for type A
   * @return A sorted list containing the same elements as the input
   */
  def insertionsort[A](list: List[A])(implicit ord: Ordering[A]): List[A] = {
    list match {
      case Nil => Nil
      case x :: xs => insert(x, insertionsort(xs))
    }
  }

  /**
   * Inserts an element into a sorted list, maintaining sortedness.
   * 
   * Precondition (verified in Isabelle): sorted(sorted)
   * Postcondition (verified in Isabelle): 
   * - sorted(insert(elem, sorted))
   * - mset(insert(elem, sorted)) = {#elem#} + mset(sorted)
   * 
   * @param elem The element to insert
   * @param sorted A sorted list
   * @param ord Implicit ordering for type A
   * @return A sorted list with elem inserted at the correct position
   */
  def insert[A](elem: A, sorted: List[A])(implicit ord: Ordering[A]): List[A] = {
    sorted match {
      case Nil => List(elem)
      case y :: ys =>
        if (ord.lteq(elem, y)) {
          elem :: sorted
        } else {
          y :: insert(elem, ys)
        }
    }
  }

  /**
   * Helper function to verify sortedness of a list.
   * Used for runtime validation during testing.
   */
  def isSorted[A](list: List[A])(implicit ord: Ordering[A]): Boolean = {
    list match {
      case Nil => true
      case _ :: Nil => true
      case x :: y :: rest => ord.lteq(x, y) && isSorted(y :: rest)
    }
  }
}