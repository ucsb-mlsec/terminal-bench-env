package isabelle.sorting

/**
 * SortingProofs - Proof-related types and utilities extracted from Isabelle/HOL
 * 
 * This file contains type definitions and helper functions that represent
 * proof obligations and correctness properties from the formally verified
 * sorting algorithm proofs in Isabelle/HOL.
 * 
 * Generated from Isabelle code extraction process.
 */
object SortingProofs {
  
  // Proof terms are erased during extraction to computational code
  type Proof = Unit
  
  // Represents the verification status of an algorithm
  sealed trait Correctness
  case object Verified extends Correctness
  case object Unverified extends Correctness
  
  // Proof obligation for sortedness property
  case class SortednessProof(property: Proof = ())
  
  // Proof obligation for permutation property
  case class PermutationProof(property: Proof = ())
  
  /**
   * Check if a list is sorted according to the given ordering.
   * This corresponds to the sorted predicate in the Isabelle theory.
   */
  def isSorted[A](list: List[A])(implicit ord: Ordering[A]): Boolean = {
    list match {
      case Nil => true
      case _ :: Nil => true
      case x :: xs @ (y :: _) => ord.lteq(x, y) && isSorted(xs)
    }
  }
  
  /**
   * Verify that two lists contain the same elements (permutation check).
   * This is a runtime check corresponding to the multiset equality in Isabelle.
   */
  def isPermutation[A](list1: List[A], list2: List[A]): Boolean = {
    list1.sorted == list2.sorted
  }
  
  /**
   * Runtime verification wrapper for sorting correctness.
   * Checks both sortedness and permutation properties.
   */
  def verifySortingCorrectness[A](input: List[A], output: List[A])
                                 (implicit ord: Ordering[A]): Correctness = {
    if (isSorted(output) && isPermutation(input, output)) {
      Verified
    } else {
      Unverified
    }
  }
}