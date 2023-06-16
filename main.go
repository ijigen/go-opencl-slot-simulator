package main

import (
	"fmt"
	"strings"

	"go-opencl/opencl"
)

const (
	deviceType = opencl.DeviceTypeAll

	//dataSize = 3

	programCode = `
kernel void kern(global const uchar* wheels,global uchar* out)
{
	size_t i=get_global_id(0);
	uchar wheel0=i;
	uchar wheel1=wheel0/6;
	uchar wheel2=wheel1/6;


	size_t w=i*3*3;
	
	out[w+0] = wheels[(wheel0+0)%6];
	out[w+1] = wheels[(wheel0+1)%6];
	out[w+2] = wheels[(wheel0+2)%6];

	out[w+3] = wheels[(wheel1+0)%6];
	out[w+4] = wheels[(wheel1+1)%6];
	out[w+5] = wheels[(wheel1+2)%6];

	out[w+6] = wheels[(wheel2+0)%6];
	out[w+7] = wheels[(wheel2+1)%6];
	out[w+8] = wheels[(wheel2+2)%6];
}

//kernel void wheel(global const uchar* wheel,global const uchar screen_length,global uchar* screen)
//{
//	size_t spin_index = get_global_id(0);
//	size_t offset = get_global_id(1);
//
//
//}
`
)

var test = "" +
	"12" +
	"12" +
	"12" +
	"" +
	"111" +
	"112" +
	"121" +
	"122" +
	"211" +
	"212" +
	"221" +
	"222"

var wheels = []uint8{
	1, 2, 3, 4, 5, 6,
	1, 2, 3, 4, 5, 6,
	1, 2, 3, 4, 5, 6,
}
var screenSize = [...]uint8{
	3,
	3,
	3,
}
var X = false
var O = true
var lines = func() [][][]bool {
	result := [][][]bool{
		{
			{O, O, O},
			{X, X, X},
			{X, X, X},
		},

		{
			{X, X, X},
			{O, O, O},
			{X, X, X},
		},

		{
			{X, X, X},
			{X, X, X},
			{O, O, O},
		},

		{
			{O, X, X},
			{X, O, X},
			{X, X, O},
		},

		{
			{X, X, O},
			{X, O, X},
			{O, X, X},
		},
	}
	return result
}()

//	var total = func() uint64 {
//		i := 0
//		for _, wheel := range wheels {
//			i += len(wheel)
//		}
//		return uint64(i)
//	}()
var screen = make([][][][][][]uint8, 6*6*6)

func printHeader(name string) {
	fmt.Println(strings.ToUpper(name))
	for _ = range name {
		fmt.Print("=")
	}
	fmt.Println()
}

func printInfo(platform opencl.Platform, device opencl.Device) {
	var platformName string
	err := platform.GetInfo(opencl.PlatformName, &platformName)
	if err != nil {
		panic(err)
	}

	var vendor string
	err = device.GetInfo(opencl.DeviceVendor, &vendor)
	if err != nil {
		panic(err)
	}

	fmt.Println()
	printHeader("Using")
	fmt.Println("Platform:", platformName)
	fmt.Println("Vendor:  ", vendor)
}

func main() {
	platforms, err := opencl.GetPlatforms()
	if err != nil {
		panic(err)
	}

	printHeader("Platforms")

	foundDevice := false

	var platform opencl.Platform
	var device opencl.Device
	var name string
	for _, curPlatform := range platforms {
		err = curPlatform.GetInfo(opencl.PlatformName, &name)
		if err != nil {
			panic(err)
		}

		var devices []opencl.Device
		devices, err = curPlatform.GetDevices(deviceType)
		if err != nil {
			panic(err)
		}

		// Use the first available device
		if len(devices) > 0 && !foundDevice {
			var available bool
			err = devices[0].GetInfo(opencl.DeviceAvailable, &available)
			if err == nil && available {
				platform = curPlatform
				device = devices[0]
				foundDevice = true
			}
		}

		version := curPlatform.GetVersion()
		fmt.Printf("Name: %v, devices: %v, version: %v\n", name, len(devices), version)
	}

	if !foundDevice {
		panic("No device found")
	}

	printInfo(platform, device)

	var context opencl.Context
	context, err = device.CreateContext()
	if err != nil {
		panic(err)
	}
	defer context.Release()

	var commandQueue opencl.CommandQueue
	commandQueue, err = context.CreateCommandQueue(device)
	if err != nil {
		panic(err)
	}
	defer commandQueue.Release()

	var program opencl.Program
	program, err = context.CreateProgramWithSource(programCode)
	if err != nil {
		panic(err)
	}
	defer program.Release()

	var log string
	err = program.Build(device, &log)
	if err != nil {
		fmt.Println(log)
		panic(err)
	}

	kernel, err := program.CreateKernel("kern")
	if err != nil {
		panic(err)
	}
	defer kernel.Release()

	//xx := make([]uint8, dataSize)
	//for i := range xx {
	//	xx[i] = uint8(rand.Uint32() % 256)
	//}

	wheelsBuffer, err := context.CreateBuffer2([]opencl.MemFlags{opencl.MemReadOnly, opencl.MemCopyHostPtr}, 6*3, wheels)
	if err != nil {
		panic(err)
	}
	defer wheelsBuffer.Release()

	buffer, err := context.CreateBuffer([]opencl.MemFlags{opencl.MemWriteOnly}, 6*6*6*3*3)
	if err != nil {
		panic(err)
	}
	defer buffer.Release()

	//err = kernel.SetArg(0, x.Size(), &x)
	//if err != nil {
	//	panic(err)
	//}

	err = kernel.SetArg(0, wheelsBuffer.Size(), &wheelsBuffer)
	if err != nil {
		panic(err)
	}

	err = kernel.SetArg(1, buffer.Size(), &buffer)
	if err != nil {
		panic(err)
	}

	err = commandQueue.EnqueueNDRangeKernel(kernel, 1, []uint64{6 * 6 * 6})
	if err != nil {
		panic(err)
	}

	commandQueue.Flush()
	commandQueue.Finish()

	//data := func() [][]uint8 {
	//	matrix := make([][]uint8, 3)
	//	for i := range matrix {
	//		matrix[i] = make([]uint8, 3)
	//	}
	//	return matrix
	//}()
	data := [6 * 6 * 6][3][3]uint8{}
	err = commandQueue.EnqueueReadBuffer(buffer, true, &data)
	if err != nil {
		panic(err)
	}

	fmt.Println()
	printHeader("Output")
	for _, item := range &data {
		fmt.Printf("%v ", item)
	}
	fmt.Println()

}
