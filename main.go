package main

import (
	"fmt"
	"go-opencl/opencl"
	"strings"
	"time"
)

const (
	deviceType  = opencl.DeviceTypeAll
	programCode = `

__kernel void sum(__global ulong* input, __global ulong* output, const ulong dataSize) {
    __local ulong localSum[512];
    ulong globalId = get_global_id(0);
    ulong localId = get_local_id(0);
    
    localSum[localId] = (globalId < dataSize) ? input[globalId] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(ulong size = get_local_size(0) / 2; size>0; size/=2) {
        if(localId < size) {
            localSum[localId] += localSum[localId + size];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(localId == 0) {
        output[get_group_id(0)] = localSum[0];
    }
}



//kernel void wheel(
//			const	uint	wheel_length,
//			const	uchar	wheel_offset,
//	global		 	ushort*	screens
//){
//	//size_t index = get_global_id(0);
//	//screens[index]=index%wheel_length;
//	//
//	//
//	//size_t index = get_global_id(0);
//	//size_t temp=index;
//	//while(temp>wheel_length){
//	//	temp-=wheel_length
//	//}
//	//screens[index]=temp;
//
//}


//kernel void spin(
//		const	ulong	spin_offset,	//spin序列號位移
//		const	ushort	wheels_count,	//輪帶數目
//global	const	ulong*	wheel_lengths,	//每輪長度
//global	const	ulong*	wheels_offset,	//每輪位移量
//global	const	ushort*	wheels,			//輪帶資料(symbols)
//		const	ushort	lines_count,	//線獎數
//global	const	ushort*	lines,			//線獎資料
//global			ulong*	score			//得分
//){
//	size_t index = get_global_id(0);			//序列號
//	ulong spin_index = index + spin_offset;		//spin序列號
//
//   	for(ushort line_index=0 ; line_index<lines_count ; line_index++ ){	//計算每一條線獎
//		global const ushort* line = &lines[line_index*wheels_count];	//線
//		ulong wheel_offset = 1;
//		global const ushort* wheel = &wheels[0];				//輪帶		
//        for(ushort wheel_index = 0 ; wheel_index < wheels_count ; wheel_index++ ){	
//			
//			
//			wheel_offset *= wheels_offset[wheel_index];
//			wheel = &wheels[wheel_offset];						//換輪
//		}
//    }
//	
//}

//ushort wheel0(
//					size_t	index,
//			const	uint	wheel_length,
//			const	uchar	wheel_offset
//){
//	return (index/wheel_offset)%wheel_length;
//}
//
//kernel void wheel(
//			const	uint	wheel_length,
//			const	uchar	wheel_offset,
//	global		 	ushort*	screens
//){
//	size_t index = get_global_id(0);
//	screens[index]=wheel0(index,wheel_length,wheel_offset,screens);
//}



//kernel void wheel(
//			const	uint	wheel_length,
//			const	uchar	wheel_offset,
//	global		 	ushort*	screens
//){
//	size_t index = get_global_id(0);
//	screens[index]=index%wheel_length;
//	
//
//	size_t index = get_global_id(0);
//	size_t temp=index;
//	while(temp>wheel_length){
//		temp-=wheel_length
//	}
//	screens[index]=temp;
//
//}



//kernel void line(
//			const	uint	line_length,
//	global	const	uchar*	line,
//	global	const 	uchar*	screens,
//	global	const 	uint*	score
//)
//{
//
//}
`
)

var wheels = [][]uint{
	{7, 5, 6, 18, 6, 9, 1, 10, 8, 4, 4, 4, 8, 9, 1, 10, 8, 18, 6, 8, 10, 1, 1, 10, 6, 10, 18, 6, 10, 5, 2, 2, 2, 7, 1, 7, 3, 8, 9, 4, 4, 7, 9, 4, 8, 10, 4, 5, 18, 9, 2, 2, 5, 9, 8, 0, 0, 0, 6, 10, 2, 7, 6, 4, 4, 8, 10, 6, 18, 10, 8, 4, 6, 7, 10, 7, 2, 7, 6, 3, 3, 9, 2, 5, 5, 18, 7, 8, 6, 3, 18, 0, 5, 10, 8, 6, 3, 9, 10, 1, 1, 1, 8, 18, 5, 3, 6, 4, 8, 5, 18, 7, 6, 3, 9, 8, 3, 6, 9, 18, 10, 7, 9, 10, 7, 4, 4, 8, 7, 5, 6, 7, 2, 8, 9, 10, 18, 9, 8, 3, 7, 4, 4, 9, 10, 18, 6, 8, 1, 1, 5, 4, 9, 10, 18, 7, 7, 4, 4, 10, 8, 4, 7, 9, 4, 18, 5, 9, 0, 0, 7, 8, 1, 5, 18, 6, 10, 9, 2, 2, 4, 6, 7, 10, 3, 7, 6, 9, 2, 3},
	{10, 7, 5, 9, 4, 3, 5, 10, 16, 7, 9, 3, 3, 3, 7, 16, 4, 4, 4, 8, 5, 3, 3, 10, 5, 18, 0, 0, 0, 10, 10, 3, 2, 3, 9, 7, 16, 4, 9, 2, 4, 6, 0, 9, 2, 2, 18, 8, 8, 3, 0, 2, 8, 18, 9, 8, 6, 8, 4, 16, 10, 7, 10, 1, 1, 1, 5, 6, 2, 6, 18, 10, 7, 8, 6, 16, 9, 8, 1, 8, 7, 1, 1, 16, 10, 4, 7, 6, 2, 9, 18, 10, 8, 1, 10, 9, 3, 7, 5, 4, 6, 9, 3, 6, 18, 6, 4, 4, 4, 9, 8, 10, 4, 5, 10, 7, 9, 16, 4, 4, 8, 5, 18, 5, 8, 10, 3, 3, 5, 9, 16, 7, 1, 5, 10, 18, 7, 10, 3, 5, 7, 10, 0, 0, 0, 9, 8, 2, 16, 6, 9, 8, 10, 18, 10, 4, 6, 3, 9, 2, 2, 2, 10, 6, 2, 3, 7, 8, 18, 9, 2, 10, 6, 2, 10, 5, 2, 16, 3, 7, 3, 8, 1, 10, 6, 10, 6, 2, 10, 6},
	{7, 9, 3, 3, 3, 4, 10, 2, 5, 15, 8, 5, 2, 2, 2, 8, 3, 7, 15, 8, 4, 5, 8, 2, 8, 2, 9, 0, 0, 0, 10, 5, 18, 8, 10, 1, 1, 1, 15, 2, 4, 6, 7, 9, 4, 4, 0, 0, 15, 8, 5, 9, 5, 15, 3, 9, 8, 3, 8, 6, 8, 18, 10, 7, 9, 10, 4, 9, 10, 7, 1, 9, 15, 4, 6, 10, 3, 3, 9, 10, 5, 3, 3, 8, 10, 16, 6, 7, 5, 6, 4, 5, 8, 4, 4, 15, 5, 10, 4, 9, 10, 0, 16, 3, 10, 5, 3, 10, 9, 2, 10, 7, 9, 4, 7, 8, 3, 3, 15, 9, 9, 5, 5, 7, 16, 10, 7, 9, 4, 8, 8, 5, 15, 3, 5, 7, 10, 9, 7, 2, 7, 10, 0, 0, 0, 10, 4, 5, 1, 1, 8, 9, 10, 18, 9, 8, 6, 15, 4, 4, 9, 6, 10, 8, 2, 6, 2, 10, 6, 5, 3, 10, 7, 9, 10, 3, 2, 2, 15, 10, 4, 4, 10, 9, 10, 9, 7, 10, 10, 7},
	{10, 7, 5, 9, 4, 3, 10, 5, 16, 7, 9, 3, 3, 3, 7, 5, 16, 4, 4, 10, 8, 3, 3, 8, 5, 16, 0, 0, 0, 10, 18, 10, 2, 3, 3, 7, 8, 16, 9, 2, 4, 6, 0, 9, 2, 2, 18, 3, 3, 3, 8, 2, 0, 16, 9, 8, 6, 8, 4, 4, 16, 7, 10, 1, 16, 1, 5, 6, 10, 6, 18, 10, 7, 8, 7, 9, 16, 8, 1, 8, 16, 1, 1, 1, 10, 18, 7, 6, 2, 9, 18, 10, 8, 1, 10, 7, 3, 16, 5, 4, 6, 9, 3, 6, 18, 6, 4, 4, 4, 9, 8, 10, 4, 4, 10, 7, 16, 4, 4, 4, 8, 5, 8, 16, 8, 10, 3, 3, 5, 9, 18, 7, 1, 16, 10, 5, 3, 8, 10, 18, 7, 7, 0, 0, 0, 9, 8, 9, 2, 6, 3, 8, 10, 18, 10, 4, 6, 9, 6, 2, 2, 2, 16, 6, 2, 3, 7, 8, 18, 9, 2, 10, 6, 2, 16, 4, 10, 4, 10, 7, 3, 8, 1, 10, 9, 10, 9, 10, 4, 9},
	//{7, 5, 5, 18, 6, 9, 1, 10, 10, 4, 4, 4, 8, 9, 4, 4, 6, 4, 7, 8, 10, 1, 8, 6, 10, 4, 4, 6, 10, 5, 2, 2, 2, 7, 1, 7, 18, 8, 9, 4, 4, 7, 9, 18, 8, 10, 4, 5, 9, 9, 2, 18, 5, 9, 8, 0, 0, 0, 6, 10, 2, 7, 6, 4, 4, 8, 10, 6, 18, 10, 8, 4, 6, 7, 18, 7, 2, 7, 6, 3, 3, 9, 2, 5, 5, 18, 7, 8, 6, 3, 18, 0, 5, 10, 8, 6, 3, 9, 10, 1, 1, 1, 8, 18, 5, 3, 6, 4, 8, 5, 18, 7, 6, 3, 9, 8, 8, 6, 9, 18, 10, 7, 9, 10, 7, 4, 4, 8, 7, 5, 6, 7, 2, 8, 9, 10, 18, 8, 9, 10, 7, 4, 4, 9, 10, 18, 6, 8, 1, 1, 5, 8, 9, 10, 18, 7, 7, 4, 4, 10, 8, 4, 7, 9, 10, 18, 5, 9, 0, 0, 7, 8, 1, 5, 18, 7, 4, 9, 2, 2, 8, 6, 7, 4, 2, 8, 6, 2, 8, 3},
	//{1, 2, 3},
	//{1, 2, 3, 4, 5, 6},
	//{1, 2, 3, 4, 5, 6},
	//{1, 2, 3, 4, 5, 6},
}

var screensSize = [...]uint8{
	3,
	3,
	3,
	3,
	3,
}

var lines = [][]uint8{
	{0, 0, 0},
	{1, 1, 1},
	{2, 2, 2},
	{0, 1, 2},
	{2, 1, 0},
}

//func main() {
//	m := make([]map[uint]uint, len(wheels))
//	for i, wheel := range wheels {
//		m[i] = map[uint]uint{}
//		for _, symbol := range wheel {
//			m[i][symbol] = m[i][symbol] + 1
//		}
//	}
//
//	fmt.Println(m)
//}

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

	var memMax uint64
	err = device.GetInfo(opencl.DriverMaxMemAllocSize, &memMax)
	if err != nil {
		panic(err)
	}

	var groupMax uint64
	err = device.GetInfo(opencl.DriverMaxWorkGroupSize, &groupMax)
	if err != nil {
		panic(err)
	}

	itemMax := [3]uint64{}
	err = device.GetInfo(opencl.DriverMaxWorkItemSizes, &itemMax)
	if err != nil {
		panic(err)
	}

	fmt.Println()
	printHeader("Using")
	fmt.Println("Platform:", platformName)
	fmt.Println("Vendor:  ", vendor)
	fmt.Println("DriverMaxMemAllocSize:  ", memMax)
}

func uniqueSlice[T comparable](slice []T) []T {
	keys := make(map[T]bool)
	result := []T{}

	for _, entry := range slice {
		if _, value := keys[entry]; !value {
			keys[entry] = true
			result = append(result, entry)
		}
	}
	return result
}
func toBool[T comparable](slice []T, equals T) []bool {
	result := make([]bool, len(slice))
	for i, s := range slice {
		result[i] = s == equals
	}
	return result
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

	var totalSpin uint64 = 1
	for _, innerSlice := range wheels {
		totalSpin *= uint64(len(innerSlice))
	}
	fmt.Println("totalSpin:", totalSpin)
	start := time.Now() // 获取当前时间
	screensBuffer := make([]opencl.Buffer, len(wheels))

	unit := uint64(134217728)
	for i := uint64(0); i < totalSpin; i += unit {
		count := unit
		if totalSpin-i < unit {
			count = totalSpin - i
		}
		fmt.Println(count)
	}

	///////////////////
	length := uint64(129)
	input := make([]uint64, length)
	output := make([]uint64, length)
	for i := range input {
		input[i] = 2
	}

	inputBuffer, err := context.CreateBuffer2([]opencl.MemFlags{opencl.MemReadOnly, opencl.MemCopyHostPtr}, uint64(length)*8, input)
	if err != nil {
		panic(err)
	}
	defer inputBuffer.Release()

	outputBuffer, err := context.CreateBuffer([]opencl.MemFlags{opencl.MemWriteOnly}, uint64(length)*8)
	if err != nil {
		panic(err)
	}
	defer outputBuffer.Release()

	sum, err := program.CreateKernel("sum")
	if err != nil {
		panic(err)
	}
	defer sum.Release()

	err = sum.SetArg(0, inputBuffer.Size(), &inputBuffer)
	if err != nil {
		panic(err)
	}

	err = sum.SetArg(1, outputBuffer.Size(), &outputBuffer)
	if err != nil {
		panic(err)
	}

	//err = sum.SetArg(2, 8, &length)
	//if err != nil {
	//	panic(err)
	//}

	//err = opencl.SetValueArg(sum, 2, 8, length)
	//if err != nil {
	//	panic(err)
	//}
	//a := opencl.SetValue[]{TValue: length}

	err = opencl.Value(length).Set(sum, 2)
	if err != nil {
		panic(err)
	}

	err = commandQueue.EnqueueNDRangeKernel(sum, 1, []uint64{uint64(length)})
	if err != nil {
		panic(err)
	}
	///////////////////

	commandQueue.Flush()
	commandQueue.Finish()

	elapsed := time.Since(start)
	fmt.Println("OpenCL執行完成耗时：", elapsed)

	err = commandQueue.EnqueueReadBuffer(outputBuffer, true, output)
	if err != nil {
		panic(err)
	}

	fmt.Println()
	printHeader("Output")
	for _, item := range output {
		fmt.Printf("%v ", item)
	}
	fmt.Println()

	var wheelsOffset uint64 = 1
	for i, wheel := range wheels {

		var wheelLength = uint32(len(wheel))

		//wheelBuffer, err := context.CreateBuffer2([]opencl.MemFlags{opencl.MemReadOnly, opencl.MemCopyHostPtr}, uint64(wheelLength), wheel)
		//if err != nil {
		//	panic(err)
		//}
		//defer wheelBuffer.Release()

		wheelOffset := wheelsOffset
		wheelsOffset *= uint64(wheelLength)

		screenBuffer, err := context.CreateBuffer([]opencl.MemFlags{opencl.MemReadWrite}, totalSpin*2)
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
		//err = sum.SetArg(2, wheelBuffer.Size(), &wheelBuffer)
		//if err != nil {
		//	panic(err)
		//}

		err = wheelKernel.SetArg(1, 8, &wheelOffset)
		if err != nil {
			panic(err)
		}
		//err = sum.SetArg(4, 1, &screenSize)
		//if err != nil {
		//	panic(err)
		//}

		err = wheelKernel.SetArg(2, screenBuffer.Size(), &screenBuffer)
		if err != nil {
			panic(err)
		}

		err = commandQueue.EnqueueNDRangeKernel(wheelKernel, 1, []uint64{totalSpin})
		if err != nil {
			panic(err)
		}

	}

	commandQueue.Flush()
	commandQueue.Finish()

	elapsed = time.Since(start)
	fmt.Println("OpenCL執行完成耗时：", elapsed)

	datas := make([][]uint16, len(wheels))

	for i, _ := range datas {
		datas[i] = make([]uint16, totalSpin)
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
