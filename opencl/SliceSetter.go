package opencl

// #include "opencl.h"
import "C"
import (
	"errors"
	"unsafe"
)

type SliceSetter[T Number] struct {
	Arg []T
}

func Slice[T Number](arg []T) SliceSetter[T] {
	return SliceSetter[T]{arg}
}

func (setter SliceSetter[T]) Set(kernel Kernel, index uint) error {
	var arg interface{} = setter.Arg
	var argPtr unsafe.Pointer
	var argSize uintptr
	switch p := arg.(type) {
	case []T:
		argPtr = unsafe.Pointer(&p[0])
		//argSize = unsafe.Sizeof(p)
		argSize = uintptr(len(p)) * unsafe.Sizeof(p[0])

	default:
		return errors.New("Unknown type for argValue")
	}
	errInt := clError(C.clSetKernelArg(
		kernel.kernel,
		C.cl_uint(index),
		C.size_t(argSize),
		argPtr,
	))
	return clErrorToError(errInt)
}
