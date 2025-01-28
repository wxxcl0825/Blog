> [!important] val 不可重新赋值
> val 无法用`=`, `:=`重新赋值
> 
> 要表达Mux, 需使用`Wire`

## When
`when`生成 Mux：
```scala
val w = Wire(UInt())
w := 0.U

// val w = WireDefault(0.U)  //  assign default value for complex circuit

when (cond) {
    w := 1.U
}
```
![image.png](https://s2.loli.net/2024/09/18/1lJt6UAm5E4QIxk.png)
```scala
val w = Wire(UInt())
when(cond) {
    w := 1.U
} .elsewhen(cond2) {
    w := 2.U
} .otherwise {
    w := 3.U
}
```
> [!note] elsewhen
> `elsewhen`可接无数个
> 当所有条件与某个变量相关时，最好使用`switch`

## Switch
`switch`生成 decoder：
```scala
result := 0.U  // always need to assign default value

switch(sel) {
	is (0.U) { result := 1.U }
    is (1.U) { result := 2.U }
    is (2.U) { result := 4.U }
    is (3.U) { result := 8.U }
}

switch(sel) {
	is ("b00".U) { result := "b0001".U }
    is ("b01".U) { result := "b0010".U }
    is ("b10".U) { result := "b0100".U }
    is ("b11".U) { result := "b1000".U }
}

result := 1.U << sel
```
## Encoder
### 小Encode
```scala
b := "b00".U  // assign default value
switch (a) {
	is ("b0001".U) { b:= "b00".U }
	is ("b0010".U) { b:= "b01".U }
	is ("b0100".U) { b:= "b10".U }
	is ("b1000".U) { b:= "b11".U }
}
```
### 循环
```scala 
for (i <- 0 until 10) {  // [0, 9]
	// ...
}
```
### 硬件生成器
```scala
val v = Wire(Vec(16, UInt(4.W)))
v(0) := 0.U
for (i <- 1 until 16) {
	v(i) := Mux(hotIn(i), i.U, 0.U) | v(i - 1)
}
val encOnt = v(15)
```

+ 向量v的每个元素在对应的整数值和0中选择，选择信号为输入hotIn的对应位
+ 对向量v做约化，求或相当于求前缀或，约化结果为`v(15)`

生成结果
```verilog
module Encoder(
  input         clock,
                reset,
  input  [15:0] io_hotIn,
  output [3:0]  io_encOut
);

  wire [2:0] _v_5_T_1 = io_hotIn[5] ? 3'h5 : 3'h0;
  wire [2:0] _v_6_T_1 = io_hotIn[6] ? 3'h6 : 3'h0;
  assign io_encOut =
    {4{io_hotIn[15]}} | (io_hotIn[14] ? 4'hE : 4'h0) | (io_hotIn[13] ? 4'hD : 4'h0)
    | (io_hotIn[12] ? 4'hC : 4'h0) | (io_hotIn[11] ? 4'hB : 4'h0)
    | (io_hotIn[10] ? 4'hA : 4'h0) | (io_hotIn[9] ? 4'h9 : 4'h0)
    | {io_hotIn[8],
       io_hotIn[7] | _v_6_T_1[2] | _v_5_T_1[2] | io_hotIn[4],
       {2{io_hotIn[7]}} | _v_6_T_1[1:0] | _v_5_T_1[1:0] | {2{io_hotIn[3]}}
         | io_hotIn[2:1]};
endmodule
```
## 优先仲裁器
+ 优先放行低位信号
+ 每次只放行一个信号

### 小仲裁器
```scala
val grant = VecInit(false.B, false.B, false.B)
val notGranted = VecInit(false.B, false.B)

grant(0) := request(0)
notGranted(0) := !grant(0)
grant(1) := request(1) && notGranted(0)
notGranted(1) := !grant(1) && notGranted(0)
grant(2) := request(2) && notGranted(1)
```

```scala
val grant = WireDefault("b000".U(3.W))
switch (request) {
	is ("b000".U) { grant := "b000".U }
	is ("b001".U) { grant := "b001".U }
	is ("b010".U) { grant := "b010".U }
	is ("b011".U) { grant := "b001".U }
	is ("b100".U) { grant := "b100".U }
	is ("b101".U) { grant := "b001".U }
	is ("b110".U) { grant := "b010".U }
	is ("b111".U) { grant := "b001".U }
}
```
### 硬件生成器
```scala
val grant = VecInit.fill(n)(false.B)
val notGranted = VecInit.fill(n)(false.B)

grant(0) := request(0)
notGranted(0) := !grant(0)

for (i <- 1 until n) {
	grant(i) := request(i) && notGranted(i - 1)
    notGranted(i) := !grant(i) && notGranted(i - 1)
}
```
生成结果
```verilog
module Arbiter(
  input        clock,
               reset,
  input  [2:0] io_request,
  output [2:0] io_grant
);

  wire grant_1 = io_request[1] & ~(io_request[0]);
  assign io_grant = {io_request[2] & ~grant_1 & ~(io_request[0]), grant_1, io_request[0]};
endmodule
```

> [!note] 优先编码器
> 优先编码器 = 仲裁器 + 编码器
> ![image.png](https://s2.loli.net/2024/09/19/Ko1MXzcGwgVriF5.png)

## 比较器
```scala
val equ = a === b
val gt = a > b
```
