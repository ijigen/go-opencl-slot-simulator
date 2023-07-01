package opencl

// #include "opencl.h"
import "C"
import "unsafe"

type Arg interface {
	Size() C.size_t
	Pointer() unsafe.Pointer
}
