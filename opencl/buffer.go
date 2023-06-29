package opencl

// #include "opencl.h"
import "C"
import "unsafe"

type MemFlags uint64

const (
	MemReadWrite   MemFlags = C.CL_MEM_READ_WRITE
	MemWriteOnly            = C.CL_MEM_WRITE_ONLY
	MemReadOnly             = C.CL_MEM_READ_ONLY
	MemCopyHostPtr          = C.CL_MEM_COPY_HOST_PTR
	// ...
)

type Buffer struct {
	buffer C.cl_mem
}

func createBuffer(context Context, flags []MemFlags, size uint64) (Buffer, error) {
	// AND together all flags
	flagBitField := uint64(0)
	for _, flag := range flags {
		flagBitField |= uint64(flag)
	}

	var errInt clError
	buffer := C.clCreateBuffer(
		context.context,
		C.cl_mem_flags(flagBitField),
		C.size_t(size),
		nil,
		(*C.cl_int)(&errInt),
	)
	if errInt != clSuccess {
		return Buffer{}, clErrorToError(errInt)
	}

	return Buffer{buffer}, nil
}

func createBuffer2(context Context, flags []MemFlags, size uint64, dataPtr interface{}) (Buffer, error) {
	// AND together all flags
	flagBitField := uint64(0)
	for _, flag := range flags {
		flagBitField |= uint64(flag)
	}

	var ptr unsafe.Pointer
	switch p := dataPtr.(type) {
	case []uint64:
		ptr = unsafe.Pointer(&p[0])
	case []uint32:
		ptr = unsafe.Pointer(&p[0])
	case []uint16:
		ptr = unsafe.Pointer(&p[0])
	case []uint8:
		ptr = unsafe.Pointer(&p[0])
	case *uint64:
		ptr = unsafe.Pointer(p)
	case *uint32:
		ptr = unsafe.Pointer(p)
	case *uint16:
		ptr = unsafe.Pointer(p)
	case *uint8:
		ptr = unsafe.Pointer(p)
	}

	var errInt clError
	buffer := C.clCreateBuffer(
		context.context,
		C.cl_mem_flags(flagBitField),
		C.size_t(size),
		ptr,
		(*C.cl_int)(&errInt),
	)
	if errInt != clSuccess {
		return Buffer{}, clErrorToError(errInt)
	}

	return Buffer{buffer}, nil
}

func (b Buffer) Size() uint64 {
	return uint64(C.sizeof_cl_mem)
}

func (b Buffer) Release() {
	C.clReleaseMemObject(b.buffer)
}
