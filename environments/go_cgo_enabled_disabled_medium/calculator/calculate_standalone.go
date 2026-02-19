//go:build !cgo

package main

func Calculate(n int) int {
	return n * 2 + 5
}