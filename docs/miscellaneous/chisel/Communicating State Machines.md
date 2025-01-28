自动机通信：

+ 一个自动机的输出是另一个自动机的输入
+ 一个自动机观察另一个自动机的输出

自动机分解：将一个大自动机分解成多个小自动机
## 例——闪光灯
输入：`start`
输出：`light`
功能：

+ 当`start`拉高1CC, 闪光灯开闪
+ 每次闪3次
+ 每次闪时拉高6CC, 拉低4CC
+ 闪完后，关灯，等待下次闪

分解1：
	![image.png](https://s2.loli.net/2024/10/04/C3t4YaVryO2vuNB.png)

+ 主FSM：控制闪烁
+ 计时FSM：负责等待
	+ `timerLoad`拉高，计时器载入被递减的初始值
	+ `timerSelect`选择load = 5 / 3
	+ `timerDone`当计数结束时被拉高
	+ 其余时间计数器自己往下走

实现：
```scala
class LightFlasher extends Module {
  val io = IO(new Bundle {
    val start = Input(Bool())
    val light = Output(Bool())
  })

  io.light := false.B

  val timerLoad = WireDefault(false.B)
  val timerSelect = WireDefault(true.B)
  val timerDone = Wire(Bool())

  val timerReg = RegInit(0.U)
  timerDone := timerReg === 0.U
  when(!timerDone) {
    timerReg := timerReg - 1.U
  }
  when(timerLoad) {
    when(timerSelect) {
      timerReg := 5.U
    }.otherwise {
      timerReg := 3.U
    }
  }

  timerLoad := timerDone

  object State extends ChiselEnum {
    val off, flash1, space1, flash2, space2, flash3 = Value
  }
  import State._

  val stateReg = RegInit(off)

  switch(stateReg) {
    is(off) {
	  timerLoad := true.B
      when(io.start) {
        stateReg := flash1
      }
    }
    is(flash1) {
      timerSelect := false.B
      io.light := true.B
      when(timerDone) {
        stateReg := space1
      }
    }
    is(space1) {
      when(timerDone) {
        stateReg := flash2
      }
    }
    is(flash2) {
      timerSelect := false.B
      io.light := true.B
      when(timerDone) {
        stateReg := space2
      }
    }
    is(space2) {
      when(timerDone) {
        stateReg := flash3
      }
    }
    is(flash3) {
      timerSelect := false.B
      io.light := true.B
      when(timerDone) { stateReg := off }
    }
  }
}
```

+ 默认状态：当自动机未达到改变该值的状态时，<u>Wire值不会锁存，而是会变为默认状态</u>
	+ 可以将"multi-driven wire"的信号理解为不同条件下的不同取值
	+ 当多次设置默认状态时，后设置的覆盖之前设置的
	+ Reg值会锁存
+ `timerLoad := timerDone` 覆盖 `val timerLoad = WireDefault(false.B)` 成为 `timerLoad` 的默认值，即主自动机不对`timerLoad`进行设置(条件触发)时，`timerLoad`将由`timerDone`决定
+ 主自动机解读：
	+ 初始时，位于`off`态，按住`timerLoad`
		+ 若不按住`timerLoad`，则会被`timerDone`影响而自动向下走
		+ `start`拉高后，转移到`flash1`态
	+ 进入`flash1`态后，`timer`开始从5到0倒计时，倒计时期间始终处于`flash1`态(stateReg锁存)，按住`light`, `timerSelect`
		+ 若不按住`light`，则会被默认值关闭
		+ 按住`timerSelect`，为下一次倒计时选择3..0
			+ 事实上，不按住`timerSelect`，而是在转移时拉低行为上也正确，但会导致拉低的条件更复杂
			+ 只需保证在`timerLoad`为高即`timerDone`为高时，`timerSelect`为低即可为`timerReg`选择正确的输入值3

问题：存在冗余性，`flash1`, `flash2`, `flash3`功能相同，`space1`, `space2`功能相同

分解2：引入计数器，主自动机简化为3个态
![image.png](https://s2.loli.net/2024/10/06/HmW4kshRKYBQTgi.png)
实现：
```scala
class LightFlasher extends Module {
  val io = IO(new Bundle {
    val start = Input(Bool())
    val light = Output(Bool())
  })

  io.light := false.B

  val timerLoad = WireDefault(false.B)
  val timerSelect = WireDefault(true.B)
  val timerDone = Wire(Bool())

  val timerReg = RegInit(0.U)
  timerDone := timerReg === 0.U
  when(!timerDone) {
    timerReg := timerReg - 1.U
  }
  when(timerLoad) {
    when(timerSelect) {
      timerReg := 5.U
    }.otherwise {
      timerReg := 3.U
    }
  }

  timerLoad := timerDone

  val cntLoad = WireDefault(false.B)
  val cntDecr = WireDefault(false.B)
  val cntDone = Wire(Bool())

  val cntReg = RegInit(0.U)
  cntDone := cntReg === 0.U
  when(cntLoad) { cntReg := 2.U }
  when(cntDecr) { cntReg := cntReg - 1.U }

  object State extends ChiselEnum {
    val off, flash, space = Value
  }
  import State._

  val stateReg = RegInit(off)

  switch(stateReg) {
    is(off) {
      timerLoad := true.B
      timerSelect := true.B
      cntLoad := true.B
      when(io.start) { stateReg := flash }
    }
    is(flash) {
      timerSelect := false.B
      io.light := true.B
      when(timerDone & !cntDone) { stateReg := space }
      when(timerDone & cntDone) { stateReg := off }
    }
    is(space) {
      cntDecr := timerDone
      when(timerDone) { stateReg := flash }
    }
  }
}
```

+ 在每次`space`要发生转移时，`cntDecr`被拉高，即第一次space -> flash后，cntReg = 1; 第二次space -> flash后，cntReg = 0，之后flash无法进入space

## 带Datapath的自动机
Datapath：用于对自动机间通信的信号做运算，而不是仅仅只通信控制信号
