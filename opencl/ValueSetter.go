package opencl

// #include "opencl.h"
import "C"
import (
	"unsafe"
)

type Number interface {
	int16 | int32 | int64 | uint | uint8 | uint16 | uint32 | uint64 | float32 | float64
}

type ValueSetter[T Number] struct {
	Arg T
}

func Value[T Number](arg T) ValueSetter[T] {
	return ValueSetter[T]{arg}
}

func (setter ValueSetter[T]) Set(kernel Kernel, index uint) error {
	errInt := clError(C.clSetKernelArg(
		kernel.kernel,
		C.cl_uint(index),
		C.size_t(unsafe.Sizeof(setter.Arg)),
		unsafe.Pointer(&setter.Arg),
	))
	return clErrorToError(errInt)
}
