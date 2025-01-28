## 信号类型和常量
宽度：`n.W`

类型：`<Type>(Width)`
```scala
Bits(8.W)  // Bit's vector
UInt(8.W)  // Unsigned int
SInt(10.W) // Signed Int, 2' complement

Bits(n.W)
```
常量：`c.<T>[(Width)]` 自动宽度推断
```scala
0.U
-3.S

3.U(4.W)
```
> [!important] Bit extraction VS Width
> `1.U(32)`：1的第32位
> 
> `1.U(32.W)`：32'b1

进制：`"Str".<T>`
```scala
"hff".U
"0377".U

"b1111_1111".U  // _ is ignored
```
字符：`'Char'.<T>` ASCII 编码
```scala
val aChar = 'A'.U
```
逻辑值：`[true|false].B`
```scala
Bool() // Type

true.B
false.B
```
## 组合电路
逻辑操作：自动宽度推断
```scala
val and = a & b
val or = a | b
val xor = a ^ b
val not = ~a
```
算数操作：自动宽度推断
```scala
val add = a + b
val sub = a - b  // max(|a|, |b|)
val neg = -a
val mul = a * b  // |a| + |b|
val div = a / b 
val mod = a % b  // |a|
```
延迟绑定(非定义初始化)：`:=` 用于值<u>更新</u>
```scala
val w = Wire(UInt())

w := a & b
```
位选：`x(n[, m])`
```scala
val sign = x(31)
val lowByte = largeWord(7, 0)
```
连接：`Cat(a, b[, ...])` `a ## b`
```scala
val word = highByte ## lowByte
```
操作符：
![image.png](https://s2.loli.net/2024/09/17/1Jo4x5KVeAtLwhB.png)
操作函数：
![image.png](https://s2.loli.net/2024/09/17/1h56wCIS8itATav.png)
Mux：`Mux(S, a, b)`
```scala
val result = Mux(sel, a, b)  // result = sel ? a : b
```
## 寄存器
> [!note] 编程模型 
> 默认全局时钟 + 同步复位
> ![image.png](https://s2.loli.net/2024/09/17/qLsJwFRG76Mudc3.png)

定义：`RegInit(c)` `RegInit(d[, c])`
```scala
val reg = RegInit(0.U(8.W))   // 8-bit reg initialized with 0
val nextReg = RegNext(d)      // connected to input at definition
val bothReg = RegNext(d, 0.U) // connected to input & initial value
```
操作：
```scala
reg := d    // connect to input
val q = reg // connect to output
```
> [!example] 计数器
> ```scala
> val cntReg = RegInit(0.U(8.W))
> cntReg := Mux(cntReg === 9.U, 0.U, cntReg + 1.U)
> ```

## 结构
### Bundle
Bundle：组合不同类型信号
+ 子信号称作域(field)

定义：类继承
```scala
class Channel extends Bundle {
	val data = UInt(32.W)
	val valid = Bool()
}
```
实例化：`Wire(new C())`
```scala
val ch = Wire(new Channel())
```
成员访问：
```scala
ch.data := 123.U
ch.vaild := true.B

val b = ch.valid
val channel = ch  // referenced as a whole
```
### Vec
Vec：组合<u>相同类型</u>信号

+ 通过下标访问

> [!faq] Vec VS Seq
> Vec使用场景：
> + 硬件动态寻址(Mux)
> + 寄存器堆
> + 模块端口
> 
> 其余描述一系列事物应选择Seq

#### 组合Vec
组合Vec：Mux

定义：`Wire(Vec(n, Type))` `VecInit([c|Sig][, ...])`
```scala
val v = Wire(Vec(3, UInt(4.W)))

val defVec = VecInit(1.U(3.W), 2.U, 3.U)
val defVecSig = VecInit(d, e, f)
```
操作：
```scala
val vec = Wire(Vec(3, UInt(8.W)))
vec(0) := x
vec(1) := y
vec(2) := z
val muxOut = vec(sel)

when (cond) {
	defVec(0) := 4.U
	defVec(1) := 5.U
	defVec(2) := 6.U
}
val vecOut = defVec(sel)
```
#### 寄存器Vec
组合Vec：RegFile

定义：`Reg(Vec(n, Type))` `RegInit(VecInit([c|Sig][, ...]))` 指定寄存器初始值
```scala
val regVec = Reg(Vec(3, UInt(8.W)))

val InitReg = RegInit(VecInit(0.U(3.W), 1.U, 2.U))

// use seq to do batch initialize
val resetRegFile = RegInit(VecInit(Seq.fill(32)(0.U(32.W))))
```
操作：
```scala
val dout = regVec(rdIdx)
regVec(wrIdx) := din

val resetReg = initReg(sel)
initReg(0) := d
initReg(1) := e
initReg(2) := f
```
> [!example] RV32 寄存器堆
> ```scala
> val registerFile = Reg(Vec(32, UInt(32.W)))
> registerFile(index) := dIn
> val dOut = registerFile(index)
>``` 

### Bundle, Vec组合
Vec套Bundle：
```scala
val vecBundel = Wire(Vec(8, new Channel()))
```
Bundle套Vec：
```scala
class BundleVec extends Bundle {
	val field = UInt(8.W)
	val vector = Vec(4, UInt(8.W))
}
```
Bundle寄存器：先创建初始值的Wire，再转Reg
```scala
val initVal = Wire(new Channel())

initVal.data := 0.U
initVal.valid := false.B

val channelReg = RegInit(initVal)
```
> [!example] 部分赋值
> 对于
> ```verilog
> reg [15:0] assignWord;
> assignWord[7:0] <= lowByte;
> assignWord[15, 8] <= highByte;
> ```
> Chisel3中需使用Bundel实现
> ```scala
> val assignWord = Wire(UInt(16.W))
> 
> class Split extends Bundle {
> 	// 高位 -> 低位
> 	val high = UInt(8.W)
> 	val low = UInt(8.W)
> }
> 
> val split = Wire(new Split())
> split.low := lowByte
> split.high := highByte
> 
> assignWord := split.asUInt
> ```
> 问题：需知道Bundle -> Bits 时的顺序
> 解决：使用Vec\[Bool]
> ```scala
> val vecResult = Wire(Vec(4, Bool()))
> 
> vecResult(0) := data(0)
> vecResult(1) := data(1)
> vecResult(2) := data(2)
> vecResult(3) := data(3)
> 
> val uintResult = vecResult.asUInt
> ```

### Wire, Reg & IO
Chisel类型 VS 硬件：

+ `UInt`, `SInt`, `Bits` 不代表硬件
+ 只有将Chisel类型用`Wire`, `Reg`, `IO`包裹才生成硬件
	+ Wire：组合逻辑
	+ Reg：寄存器(D 触发器的集合)
	+ IO：模块引脚

`=` VS `:=`：

+ <u>创建</u>硬件(命名)：`val <name> = [Wire|Reg|IO](<Type>())`
	+ 使用<u>常值变量</u>(val)创建
	+ 带初始值硬件：`WireDefault`, `RegInit`, `VecInit`
		+ 尽管Chisel进行自动宽度推断，<u>定义时最好指定宽度</u>
+ 已有硬件<u>赋值</u>：`:=`
	+ 条件赋值：每个条件均需赋值
		+ Chisel不允许锁存器

> [!important] Chisel 是硬件语言
> ==Chisel生成硬件==，所有代码生成硬件，并行执行

