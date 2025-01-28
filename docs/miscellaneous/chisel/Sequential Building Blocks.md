## Register
使能Reg
```scala
val enableReg = Reg(UInt(4.W))
when (enable) {
	enableReg := inVal
}

val enableReg2 = RegEnable(inVal, enable)

val resetEnableReg = RegInit(0.U(4.W))
when (enable) {
	resetEnableReg := inVal
}

val resetEnableReg2 = RegEnable(inVal, 0.U(4.W), enable)
```
表达式Reg
```scala
val risingEdge = din & !RegNext(din)
// RegNext(d)
// t时刻 RegNext(din) 为 t-1时刻 din 值
```
## 计数器
![image.png](https://s2.loli.net/2024/09/20/X2wxB64lYHGZVWM.png)
```scala
val cntReg = RegInit(0.U(4.W))
cntReg := cntReg + 1.U
```
数event
```scala
val cntEventsReg = RegInit(0.U(4.W))
when(event) {
	cntEventsReg := cntEventsReg + 1.U
}
```
范围计数 $[0,N]$
```scala
val cntReg = RegInit(0.U(8.W))

cntReg := cntReg + 1.U
when(cntReg === N) {
	cntReg := 0.U
}

cntReg := Mux(cntReg === N, 0.U, cntReg + 1.U)
```

```scala
val cntReg = RegInit(N)

cntReg := cntReg - 1.U
when(cntReg === 0.U) {
	cntReg := N
}
```
生成函数
```scala
def genCounter(n: Int) = {
	val cntReg = RegInit(0.U(8.W))
	cntReg := Mux(cntReg === n.U, 0.U, cntReg + 1.U)
	cntReg  // return value
}

val count10 = genCounter(10)
```
### 计时器
原理：数$f_{clock}/f_{tick}$个周期，为其它器件<u>提供使能信号</u>(而不是次级时钟信号)
```scala
val tickCounterReg = RegInit(0.U(32.W))
val tick = tickCounterReg === (N-1).U

tickCounterReg := tickCounterReg + 1.U
when (tick) {
	tickCounterReg := 0.U
}

val lowFrequCntReg = RegInit(0.U(4.W))
when (tick) {
	lowFrequCntReg := lowFrequCntReg + 1.U
}
```
### 书呆子计数器
优化：通过判断符号位确定计数值是否为负，从而判断是否达到计数终点
```scala
val MAX = (N-2).S(8.W)
val cntReg = RegInit(MAX)
io.tick := false.B

cntReg := cntReg - 1.S
when(cntReg(7)) {
	cntReg := MAX
	io.tick := ture.B
}
```
### 定时器
```scala
val cntReg = RegInit(0.U(8.W))
val done = cntReg === 0.U

val next = WireDefault(0.U)
when (load) {
	next := din
} .elsewhen (!done) {
	next := cntReg - 1.U
}
cntReg := next
```
### 脉宽控制
每nrCycles CC有din CC高电平
```scala
import chisel3.util.unsignedBitLength

def pwm(nrCycles: Int, din: UInt) = {
	val cntReg = RegInit(0.U(unsignedBitLength(nrCycles-1).W))
	cntReg := Mux(cntReg === (nrCycles-1).U, 0.U, cntReg + 1.U)
	din > cntReg
}

val din = 3.U
val dout = pwm(10, din)
```

+ `unsignedBitLength(n)` = $\lfloor\log_2(n)\rfloor+1$

应用：呼吸灯
```scala
val FREQ = 100000000
val MAX = FREQ/1000

val modulationReg = RegInit(0.U(32.W))

val upReg = RegInit(true.B)

when (modulationReg < FREQ.U && upReg) {
	modulationReg := modulationReg + 1.U
}.elsewhen (modulationReg == FREQ.U && upReg) {
	upReg := false.B
}.elsewhen (modulationReg > 0.U && !upReg) {
	modulationReg := modulationReg - 1.U
}.otherwise {
	upReg := true.B
}

val sig = pwm(MAX, modulationReg >> 10)
```

+ 每周期中高电平占比上下浮动
+ `modulationReg / 1000`成本高，使用右移实现

## 移位寄存器
```scala
val shiftReg = Reg(UInt(4.W))
shiftReg := shiftReg(2, 0) ## din
val dout = shiftReg(3)
```

+ 左移，每次最高位被移出，向低位输入

### 串行转并行
```scala
val outReg = RegInit(0.U(4.W))
outReg := serIn ## outReg(3, 1)
val q = outReg
```

+ 右移，每次向高位输入

### 并行转串行
```scala
val loadReg = RegInit(0.U(4.W))
when (load) {
	loadReg := d
}.otherwise {
	loadReg := 0.U ## loadReg(3, 1)
}
val serOut = loadReg(0)
```
## Memory
内存模型：
![image.png](https://s2.loli.net/2024/10/03/DdUv4YpxAcKuire.png)
实现：使用内置类型`SyncReadMem`
```scala
class Memory() extends Module {
	val io = IO(new Bundle {
		val rdAddr = Input(UInt(10.W))
		val rdData = Output(UInt(8.W))
		val wrAddr = Input(UInt(10.W))
		val wrData = Input(UInt(8.W))
		val wrEna = Input(Bool())
	})

	val mem = SyncReadMem(1024, UInt(8.W))

	io.rdData := mem.read(io.rdAddr)

	when(io.wrEna) {
		mem.write(io.wrAddr, io.wrData)
	}
}
```

+ `SyncReadMem`本质：`reg [7:0] Memory[0:1023];`

!!! faq
	在同一周期对同一地址读写
	在同一周期写入并读取同一地址，读出值<u>未定义</u>

### Forward memory
读取到当前正在对该地址写入的值
原理图：
	![image.png](https://s2.loli.net/2024/10/03/lYg1bPHincBkJIq.png)
实现：
```scala
class ForwardingMemory extends Module {
	val io = IO(new Bundle {
		val rdAddr = Input(UInt(10.W))
		val rdData = Output(UInt(8.W))
		val wrAddr = Input(UInt(10.W))
		val wrData = Input(UInt(8.W))
		val wrEna = Input(Bool())
	})

	val mem = SyncReadMem(1024, UInt(8.W))

	val wrDataReg = RegNext(io.wrData)
	val doForwardReg = RegNext(io.wrAddr === io.rdAddr && io.wrEna)

	val memData = mem.read(io.rdAddr)

	when(io.wrEna) {
		mem.write(io.wrAddr, io.wrData)
	}

	io.rdData := Mux(doForwardReg, wrDataReg, memData)
}
```

+ 将当前周期需要写入的值先缓存在`wtDataReg`中
+ 读取时判断上一个周期是否发生了forward, 若发生了forward，则这个周期直接给出上个周期存入的`wtDataReg`的值

### 同步写非同步读Mem
语法：`Mem`
实现：触发器

+ 缺陷：消耗大，尽量使用同步Mem
+ 针对具体板卡，<u>将相应的Mem作黑盒</u>

设置Memory初始值：`loadMemoryFromFile`
```scala
loadMemoryFromFile(memory, fp)
```
