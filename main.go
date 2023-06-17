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
//kernel void kern(global const uchar* wheels,global uchar* out)
//{
//	size_t i=get_global_id(0);
//	uchar wheel0=i;
//	uchar wheel1=wheel0/6;
//	uchar wheel2=wheel1/6;
//
//
//	size_t w=i*3*3;
//	
//	out[w+0] = wheels[(wheel0+0)%6];
//	out[w+1] = wheels[(wheel0+1)%6];
//	out[w+2] = wheels[(wheel0+2)%6];
//
//	out[w+3] = wheels[(wheel1+0)%6];
//	out[w+4] = wheels[(wheel1+1)%6];
//	out[w+5] = wheels[(wheel1+2)%6];
//
//	out[w+6] = wheels[(wheel2+0)%6];
//	out[w+7] = wheels[(wheel2+1)%6];
//	out[w+8] = wheels[(wheel2+2)%6];
//}

kernel void wheel(
			const	uint	wheel_length,
	global	const	uchar*	wheel,
			const	uchar	wheel_offset,
			const	uchar	screen_size,
	global		 	uchar*	screens)
{
	size_t spin_index = get_global_id(0);
	size_t offset = get_global_id(1);
	
	screens[spin_index*screen_size+offset]=wheel[(spin_index/wheel_offset+offset)%wheel_length];
}
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

var wheels = [][]uint8{
	{1, 2, 3, 4, 5, 6, 6, 5, 5, 6, 6, 8, 1, 6, 7, 1, 4},
	{1, 2, 3, 4, 5, 6, 7},
	{1, 2, 3, 4, 5, 6},
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

	var wheel0Length = uint32(len(wheels[0]))

	wheel0Buffer, err := context.CreateBuffer2([]opencl.MemFlags{opencl.MemReadOnly, opencl.MemCopyHostPtr}, uint64(len(wheels[0])), wheels[0])
	if err != nil {
		panic(err)
	}
	defer wheel0Buffer.Release()

	wheel0Offset := uint64(1)

	screen0Size := screenSize[0]

	var spin_total uint64 = 1
	for _, innerSlice := range wheels {
		spin_total *= uint64(len(innerSlice))
	}
	screen0Buffer, err := context.CreateBuffer([]opencl.MemFlags{opencl.MemWriteOnly}, spin_total*uint64(screenSize[0]))
	if err != nil {
		panic(err)
	}
	defer screen0Buffer.Release()

	wheelKernel, err := program.CreateKernel("wheel")
	if err != nil {
		panic(err)
	}
	defer wheelKernel.Release()

	err = wheelKernel.SetArg(0, 4, &wheel0Length)
	if err != nil {
		panic(err)
	}
	err = wheelKernel.SetArg(1, wheel0Buffer.Size(), &wheel0Buffer)
	if err != nil {
		panic(err)
	}
	err = wheelKernel.SetArg(2, 8, &wheel0Offset)
	if err != nil {
		panic(err)
	}
	err = wheelKernel.SetArg(3, 1, &screen0Size)
	if err != nil {
		panic(err)
	}
	err = wheelKernel.SetArg(4, screen0Buffer.Size(), &screen0Buffer)
	if err != nil {
		panic(err)
	}

	err = commandQueue.EnqueueNDRangeKernel(wheelKernel, 2, []uint64{spin_total, uint64(screenSize[0])})
	if err != nil {
		panic(err)
	}

	commandQueue.Flush()
	commandQueue.Finish()

	a := spin_total * uint64(screenSize[0])
	data := make([]uint8, a)

	//data := [6 * 6 * 6][3]uint8{}
	err = commandQueue.EnqueueReadBuffer(screen0Buffer, true, data)
	if err != nil {
		panic(err)
	}

	fmt.Println()
	printHeader("Output")
	for _, item := range data {
		fmt.Printf("%v ", item)
	}
	fmt.Println()

}
