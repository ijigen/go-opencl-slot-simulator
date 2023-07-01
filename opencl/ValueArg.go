package opencl

// #include "opencl.h"
import "C"
import (
	"unsafe"
)

type Number interface {
	int16 | int32 | int64 | uint | uint8 | uint16 | uint32 | uint64 | float32 | float64
}

type ValueArg[T Number] struct {
	Arg *T
}

func Value[T Number](arg T) ValueArg[T] {
	return ValueArg[T]{&arg}
}

func (v ValueArg[T]) Size() C.size_t {
	return C.size_t(unsafe.Sizeof(v.Arg))
}

func (v ValueArg[T]) Pointer() unsafe.Pointer {
	return unsafe.Pointer(v.Arg)
}
