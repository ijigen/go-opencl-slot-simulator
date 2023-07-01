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

//func SetSlice[T int16 | int32 | int64 | uint | uint8 | uint16 | uint32 | uint64 | float32 | float64](k Kernel, t *[]T) {
//
//}
//func SetArray[T int16 | int32 | int64 | uint | uint8 | uint16 | uint32 | uint64 | float32 | float64](k Kernel, t *[...]T) {
//
//}

func (k Kernel) Set(setters ...Setter) error {
	for i, setter := range setters {
		err := setter.Set(k, uint(i))
		if err != nil {
			panic(err)
		}
	}
	return nil
}

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
