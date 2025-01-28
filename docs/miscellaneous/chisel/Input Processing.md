## 非同步输入
!!! note
	亚稳态(metastability)
	+ 当D触发器试图捕获一个信号的逻辑高或低状态时，如果输入信号的变化时间恰好在触发器的采样边沿附近，输入信号可能无法立即确定0 / 1
	+ 在亚稳态期间，输出信号可能不会立即稳定在高电平或低电平之间，而是处于某种不确定的中间状态

解决：加一个触发器(同步化)
![image.png](https://s2.loli.net/2024/10/03/tPgS1mWH62sRD9B.png)
```scala
val btnSync = RegNext(RegNext(btn))
```
## 去抖动
思路：慢采样
```scala
val fac = 100000000/100

val btnDebReg = Reg(Bool())

val btnSync = RegNext(RegNext(btn))
val cntReg = RegInit(0.U(32.W))
val tick = cntReg === (fac-1).U
cntReg := cntReg + 1.U
when (tick) {
	cntReg := 0.U
	btnDebReg := btnSync
}
```

+ 每`fac`个周期采样一次

## 输入信号过滤
目的：消除信号中的尖刺

方法：投票电路

+ 过滤掉小于采样周期的信号

![image.png](https://s2.loli.net/2024/10/03/BHK2ym57dLcpb3M.png)
```scala
val shiftReg = RegInit(0.U(3.W))
when (tick) {
	shiftReg := shiftReg(1, 0) ## btnDebReg
}
val btnClean = (shiftReg(2) & shiftReg(1)) | (shiftReg(2) & shiftReg(0)) | (shiftReg(1) & shiftReg(0))

// 上升沿检测
val risingEdge = btnClean & !RegNext(btnClean)

val reg = RegInit(0.U(8.W))
when (risingEdge) {
	reg := reg + 1.U
}
```
### 函数整合
```scala
def sync(v: Bool) = RegNext(RegNext(v))

def rising(v: Bool) = v & !RegNext(v)

def tickGen() = {
	val reg = RegInit(0.U(log2Up(fac).W))
	val tick = reg === (fac-1).U
	reg := Mux(tick, 0.U, reg + 1.U)
	tick
}

def filter(v: Bool, t: Bool) = {
	val reg = RegInit(0.U(3.W))
	when (t) {
		reg := reg(1, 0) ## v
	}
	(reg(2) & reg(1)) | (reg(2) & reg(0)) | (reg(1) & reg(0))
}

def btnSync = sync(io.btnU)

val tick = tickGen() 
val btnDeb = Reg(Bool()) 
when (tick) {
	btnDeb := btnSync 
} 

val btnClean = filter(btnDeb, tick) 
val risingEdge = rising(btnClean)

val reg = RegInit(0.U(8.W))
when (risingEdge) { 
	reg := reg + 1.U 
}
```
## 同步复位
方法：模块具有隐藏复位信号`reset`，将其连接到同步复位信号即可
```scala
class SyncRest extend Module {
	val io = IO(new Bundle {
		val value = Output(UInt())
	})

	val syncReset = RegNext(RegNext(reset))
	val cnt = Module(new WhenCounter(5))
	cnt.reset := syncReset

	io.value = cnt.io.cnt
}
```

+ 在顶层模块进行复位同步
+ 将`reset`同步化，并与子模块`reset`信号连接

