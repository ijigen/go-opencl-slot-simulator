package opencl

// #include "opencl.h"
import "C"
import (
	"errors"
	"fmt"
	"unsafe"
)

type Kernel struct {
	kernel C.cl_kernel
}
type Set interface {
	set(kernel Kernel)
}

type SetValue[T Number] struct {
	TValue T
}

func Value[T Number](value T) SetValue[T] {
	return SetValue[T]{value}
}

type Slice[T Number] struct {
	TSlice []T
}

func (value SetValue[T]) Set(kernel Kernel, index uint) error {
	var i interface{} = value.TValue
	var argPtr unsafe.Pointer
	var argSize uintptr
	switch p := i.(type) {
	case T:
		argPtr = unsafe.Pointer(&p)
		argSize = unsafe.Sizeof(p)
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

func (slice Slice[T]) Set(kernel Kernel, index uint) error {
	var i interface{} = slice.TSlice
	var argPtr unsafe.Pointer
	var argSize uintptr
	switch p := i.(type) {
	case T:
		argPtr = unsafe.Pointer(&p)
		argSize = unsafe.Sizeof(p)
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

func createKernel(program Program, kernelName string) (Kernel, error) {
	kn := C.CString(kernelName)
	defer C.free(unsafe.Pointer(kn))

	var errInt clError
	kernel := C.clCreateKernel(program.program, kn, (*C.cl_int)(&errInt))
	if errInt != clSuccess {
		fmt.Println("Error code", errInt)
		return Kernel{}, clErrorToError(errInt)
	}

	return Kernel{kernel}, nil
}

type TValue[T any] struct {
	Value T
}

func genValue[T any](t T) TValue[T] {
	return TValue[T]{t}
}

type Number interface {
	int16 | int32 | int64 | uint | uint8 | uint16 | uint32 | uint64 | float32 | float64
}

//func SetSlice[T int16 | int32 | int64 | uint | uint8 | uint16 | uint32 | uint64 | float32 | float64](k Kernel, t *[]T) {
//
//}
//func SetArray[T int16 | int32 | int64 | uint | uint8 | uint16 | uint32 | uint64 | float32 | float64](k Kernel, t *[...]T) {
//
//}

func (k Kernel) SetArg(argIndex uint32, argSize uint64, argValue interface{}) error {
	//var argPtr unsafe.Pointer
	var argPtr unsafe.Pointer

	switch p := argValue.(type) {
	case *Buffer:
		argPtr = unsafe.Pointer(p)
	case *uint32:
		argPtr = unsafe.Pointer(p)
	case *uint64:
		argPtr = unsafe.Pointer(p)
	case *uint8:
		argPtr = unsafe.Pointer(p)
	default:
		return errors.New("Unknown type for argValue")
	}

	errInt := clError(C.clSetKernelArg(
		k.kernel,
		C.cl_uint(argIndex),
		C.size_t(argSize),
		argPtr,
	))
	return clErrorToError(errInt)
}

func (k Kernel) Release() {
	C.clReleaseKernel(k.kernel)
}
