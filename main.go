package main

import (
	"fmt"
	"strings"

	"go-opencl/opencl"
)

const (
	deviceType  = opencl.DeviceTypeAll
	programCode = `
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

var wheels = [][]uint8{
	{1, 2, 3, 4, 5, 6},
	{1, 2, 3, 4, 5, 6},
	{1, 2, 3, 4, 5, 6},
}
var screensSize = [...]uint8{
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

	var spin_total uint64 = 1
	for _, innerSlice := range wheels {
		spin_total *= uint64(len(innerSlice))
	}
	screensBuffer := make([]opencl.Buffer, len(wheels))

	var wheelsOffset uint64 = 1

	for i, wheel := range wheels {

		var wheelLength = uint32(len(wheel))

		wheelBuffer, err := context.CreateBuffer2([]opencl.MemFlags{opencl.MemReadOnly, opencl.MemCopyHostPtr}, uint64(wheelLength), wheel)
		if err != nil {
			panic(err)
		}
		defer wheelBuffer.Release()

		wheelOffset := wheelsOffset
		wheelsOffset *= uint64(wheelLength)

		screenSize := screensSize[i]

		screenBuffer, err := context.CreateBuffer([]opencl.MemFlags{opencl.MemWriteOnly}, spin_total*uint64(screenSize))
		if err != nil {
			panic(err)
		}
		defer screenBuffer.Release()
		screensBuffer[i] = screenBuffer

		wheelKernel, err := program.CreateKernel("wheel")
		if err != nil {
			panic(err)
		}
		defer wheelKernel.Release()

		err = wheelKernel.SetArg(0, 4, &wheelLength)
		if err != nil {
			panic(err)
		}
		err = wheelKernel.SetArg(1, wheelBuffer.Size(), &wheelBuffer)
		if err != nil {
			panic(err)
		}

		err = wheelKernel.SetArg(2, 8, &wheelOffset)
		if err != nil {
			panic(err)
		}
		err = wheelKernel.SetArg(3, 1, &screenSize)
		if err != nil {
			panic(err)
		}

		err = wheelKernel.SetArg(4, screenBuffer.Size(), &screenBuffer)
		if err != nil {
			panic(err)
		}

		err = commandQueue.EnqueueNDRangeKernel(wheelKernel, 2, []uint64{spin_total, uint64(screenSize)})
		if err != nil {
			panic(err)
		}

	}

	commandQueue.Flush()
	commandQueue.Finish()

	datas := make([][]uint8, len(wheels))

	for i, _ := range datas {
		datas[i] = make([]uint8, spin_total*uint64(screensSize[i]))
		err = commandQueue.EnqueueReadBuffer(screensBuffer[i], true, datas[i])
		if err != nil {
			panic(err)
		}
	}

	fmt.Println()
	printHeader("Output")
	for _, item := range datas {
		fmt.Printf("%v ", item)
	}
	fmt.Println()

}
