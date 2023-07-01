package opencl

type Setter interface {
	Set(kernel Kernel, index uint) error
}
