Components: Module
## 计数器
![image.png](https://s2.loli.net/2024/09/18/hDkbPgEmr8lCxS1.png)
```scala
class Adder extends Module {
	val io = IO(new Bundle {
		val a = Input(UInt(8.W))
		val b = Input(UInt(8.W))
		val y = Output(UInt(8.W))
	})

	io.y := io.a + io.b
}

class Register extends Module {
	val io = IO(new Bundle {
		val d = Input(UInt(8.W))
		val q = Input(UInt(8.W))
	})

	val reg = RegInit(0.U)
	reg := io.d
	io.q := reg
}

class Count10 extends Module {
	val io = IO(new Bundle {
		val dout = Output(UInt(8.W))
	})

	val add = Module(new Adder())
	val reg = Module(new Register())

	val count = reg.io.q  // UInt

	add.io.a := 1.U
	add.io.b := count
	val result = add.io.y

	val next = Mux(count === 9.U, 0.U, result)
	reg.io.d := next

	io.dout := count
}
```
## 嵌套模块
![image.png](https://s2.loli.net/2024/09/18/3aZvhUsQMJKHqdb.png)
```scala
class CompA extends Module {
	val io = IO(new Bundle {
		val a = Input(UInt(8.W))
		val b = Input(UInt(8.W))
		val x = Output(UInt(8.W))
		val y = Output(UInt(8.W))
	})

	// function of A
}

class CompB extends Module {
	val io = IO(new Bundle {
		val in1 = Input(UInt(8.W))
		val in2 = Input(UInt(8.W))
		val out = Output(UInt(8.W))
	})

	// function of B
}

class CompC extends Module {
	val io = IO(new Bundle {
		val inA = Input(UInt(8.W))
		val inB = Input(UInt(8.W))
		val inC = Input(UInt(8.W))
		val outX = Output(UInt(8.W))
		val outY = Output(UInt(8.W))
	})

	// create A & B
	val compA = Module(new CompA())
	val compB = Module(new CompB())

	// connect A & B
	compA.io.a := io.inA
	compA.io.b := io.inB
	io.outX := compA.io.x

	compB.io.in1 := compA.io.y
	compB.io.in2 := io.inC
	io.outY := compB.io.out
}

class CompD extends Module {
	val io = IO(new Bundle {
		val in = Input(UInt(8.W))
		val out = Output(UInt(8.W))
	})

	// function of D
}

class TopLevel extends Module {
	val io = IO(new Bundle {
		val inA = Input(UInt(8.W))
		val inB = Input(UInt(8.W))
		val inC = Input(UInt(8.W))
		val outM = Output(UInt(8.W))
		val outN = Output(UInt(8.W))
	})

	// create C & D
	val c = Module(new CompC())
	val d = Module(new CompD())

	// connect C & D
	c.io.inA := io.inA
	c.io.inB := io.inB
	c.io.inC := io.inC
	io.outM := c.io.outX

	d.io.in := c.io.outY
	io.outN := d.io.out
}
```
## ALU
```scala
import chisel3._
import chisel3.util._  // switch

class Alu extends Module {
	val io = IO(new Bundle {
		val a = Input(UInt(16.W))
		val b = Input(UInt(16.W))
		val fn = Input(UInt(2.W))
		val y = Output(UInt(16.W))
	})

	io.y := 0.U  // default value
	
	switch(io.fn) {
		is(0.U) { io.y := io.a + io.b }
		is(1.U) { io.y := io.a - io.b }
		is(2.U) { io.y := io.a | io.b }
		is(3.U) { io.y := io.a & io.b }
	}
}
```
## 批量连接
批量连接：`<>`
```scala
class Fetch extends Module { 
    val io = IO(new Bundle { 
        val instr = Output(UInt(32.W)) 
        val pc = Output(UInt(32.W)) 
    }) 
    // Implementation of fetch 
}

class Decode extends Module { 
    val io = IO(new Bundle { 
        val instr = Input(UInt(32.W)) 
        val pc = Input(UInt(32.W)) 
        val aluOp = Output(UInt(5.W)) 
        val regA = Output(UInt(32.W))
        val regB = Output(UInt(32.W)) 
    }) 
    // Implementation of decode 
}

class Execute extends Module { 
    val io = IO(new Bundle {
        val aluOp = Input(UInt(5.W)) 
        val regA = Input(UInt(32.W)) 
        val regB = Input(UInt(32.W)) 
        val result = Output(UInt(32.W)) 
    }) 
    // Implementation of execute 
}

class Top extends Module {
    val io = IO(new Bundle {
        val result = Output(UInt(32.W))
    })

    val fetch = Module(new Fetch())
    val decode = Module(new Decode())
    val execute = Module(new Execute())

    fetch.io <> decode.io
    decode.io <> execute.io
    io <> execute.io
}
```
## 使用Verilog模块
### 直接使用外部模块
直接使用外部模块：指定`Verilog`模块所需参数
```scala
class BUFGCE extends BlackBox(Map("SIM_DEVICE" -> "7SERIES")) {
    val io = IO(new Bundle {
        val I = Input(Clock())
        val CE = Input(Bool())
        val O = Input(Clock())
    })
}

class alt_inbuf extends ExtModule(Map("io_standard" -> "1.0V",
                                       "location" -> "IOBANK_1",
                                       "enable_bus_hold" -> "on",
                                       "weak_pull_up_resistor" -> "off",
                                       "termination" -> "parallel 50 ohms")
                                       ) {
    val io = IO(new Bundle {
        val i = Input(Bool())
        val o = Input(Bool())
    })
}
```
生成结果(需在外面套一层`Module`)：
```verilog
BUFGCE #(
    .SIM_DEVICE("7SERIES")
  ) bufgce (
    .I  (io_I),
    .CE (io_CE),
    .O  (io_O)
  );

alt_inbuf #(
    .enable_bus_hold("on"),
    .io_standard("1.0V"),
    .location("IOBANK_1"),
    .termination("parallel 50 ohms"),
    .weak_pull_up_resistor("off")
  ) alt_inbuf (
    .io_i (io_i),
    .io_o (io_o)
  );
 ```
### 内联
```scala
class BlackBoxAdderIO extends Bundle {
    val a = Input(UInt(32.W))
    val b = Input(UInt(32.W))
    val cin = Input(Bool())
    val c = Output(UInt(32.W))
    val cout = Output(Bool())
}

class InlineBlackBoxAdder extends HasBlackBoxInline {
    val io = IO(new BlackBoxAdderIO)
    setInline("InlineBlackBoxAdder.v",
    s"""
    |module InlineBlackBoxAdder(a, b, cin, c, cout);
    |input  [31:0] a, b;
    |input  cin;
    |output [31:0] c;
    |output cout;
    |wire   [32:0] sum;
    |
    |assign sum  = a + b + cin;
    |assign c    = sum[31:0];
    |assign cout = sum[32];
    |
    |endmodule
    """.stripMargin)
}

class InlineAdder extends Module {
    val io = IO(new BlackBoxAdderIO)
    val adder = Module(new InlineBlackBoxAdder)
    io <> adder.io
}
```
生成结果：
```verilog
module InlineAdder(
  input         clock,
                reset,
  input  [31:0] io_a,
                io_b,
  input         io_cin,
  output [31:0] io_c,
  output        io_cout
);

  InlineBlackBoxAdder adder (
    .a    (io_a),
    .b    (io_b),
    .cin  (io_cin),
    .c    (io_c),
    .cout (io_cout)
  );
endmodule

module InlineBlackBoxAdder(a, b, cin, c, cout);
input  [31:0] a, b;
input  cin;
output [31:0] c;
output cout;
wire   [32:0] sum;

assign sum  = a + b + cin;
assign c    = sum[31:0];
assign cout = sum[32];

endmodule
```
### 导入文件
```scala
class ResourceBlackBoxAdder extends HasBlackBoxResource {
    val io = IO(new BlackBoxAdderIO)
    addResource("/ResourceBlackBoxAdder.v")  // ./src/main/resource
}

class PathBlackBoxAdder extends HasBlackBoxPath {
    val io = IO(new BlackBoxAdderIO)
    addPath("./src/main/resource/PathBlackBoxAdder.v")
}
```
