// RUN: rm -rf %t && mkdir %t
// RUN: split-file %s %t
// RUN: fir-opt %t/all.fir --cfg-conversion | FileCheck %t/all.fir
// RUN: fir-opt %t/loop.fir --cfg-conversion="transform-if=0 transform-iterate-while=0" | FileCheck %t/loop.fir
// RUN: fir-opt %t/if.fir --cfg-conversion="transform-iterate-while=0" | FileCheck %t/if.fir
// RUN: fir-opt %t/while.fir --cfg-conversion="transform-do-loop=0 transform-if=0" | FileCheck %t/while.fir

//--- all.fir
// CHECK-LABEL: @x
// CHECK-NOT: fir.if
// CHECK-NOT: fir.do_loop
// CHECK-NOT: fir.iterate_while
func @x(%lb : index, %ub : index, %step : index, %b : i1, %addr : !fir.ref<index>) {
  fir.do_loop %iv = %lb to %ub step %step unordered {
    fir.if %b {
      fir.store %iv to %addr : !fir.ref<index>
    } else {
      %zero = arith.constant 0 : index
      fir.store %zero to %addr : !fir.ref<index>
    }
  }
  return
}

// CHECK-LABEL: @x2
func @x2(%lo : index, %up : index, %ok : i1) {
  %c1 = arith.constant 1 : index
  %unused = fir.iterate_while (%i = %lo to %up step %c1) and (%ok1 = %ok) {
    %ok2 = fir.call @f2() : () -> i1
    fir.result %ok2 : i1
  }
  return
}

//--- loop.fir
// CHECK-LABEL: @x
// CHECK: fir.if
// CHECK-NOT: fir.do_loop
func @x(%lb : index, %ub : index, %step : index, %b : i1, %addr : !fir.ref<index>) {
  fir.do_loop %iv = %lb to %ub step %step unordered {
    fir.if %b {
      fir.store %iv to %addr : !fir.ref<index>
    } else {
      %zero = arith.constant 0 : index
      fir.store %zero to %addr : !fir.ref<index>
    }
  }
  return
}

// CHECK-LABEL: @x2
// CHECK: fir.iterate_while
func @x2(%lo : index, %up : index, %ok : i1) {
  %c1 = arith.constant 1 : index
  %unused = fir.iterate_while (%i = %lo to %up step %c1) and (%ok1 = %ok) {
    %ok2 = fir.call @f2() : () -> i1
    fir.result %ok2 : i1
  }
  return
}

//--- if.fir
// CHECK-LABEL: @x
// CHECK-NOT: fir.if
// CHECK-NOT: fir.do_loop
func @x(%lb : index, %ub : index, %step : index, %b : i1, %addr : !fir.ref<index>) {
  fir.do_loop %iv = %lb to %ub step %step unordered {
    fir.if %b {
      fir.store %iv to %addr : !fir.ref<index>
    } else {
      %zero = arith.constant 0 : index
      fir.store %zero to %addr : !fir.ref<index>
    }
  }
  return
}

// CHECK-LABEL: @x2
// CHECK: fir.iterate_while
func @x2(%lo : index, %up : index, %ok : i1) {
  %c1 = arith.constant 1 : index
  %unused = fir.iterate_while (%i = %lo to %up step %c1) and (%ok1 = %ok) {
    %ok2 = fir.call @f2() : () -> i1
    fir.result %ok2 : i1
  }
  return
}

//--- while.fir
// CHECK-LABEL: @x
// CHECK: fir.do_loop
// CHECK: fir.if
func @x(%lb : index, %ub : index, %step : index, %b : i1, %addr : !fir.ref<index>) {
  fir.do_loop %iv = %lb to %ub step %step unordered {
    fir.if %b {
      fir.store %iv to %addr : !fir.ref<index>
    } else {
      %zero = arith.constant 0 : index
      fir.store %zero to %addr : !fir.ref<index>
    }
  }
  return
}

// CHECK-LABEL: @x2
// CHECK-NOT: fir.iterate_while
func @x2(%lo : index, %up : index, %ok : i1) {
  %c1 = arith.constant 1 : index
  %unused = fir.iterate_while (%i = %lo to %up step %c1) and (%ok1 = %ok) {
    %ok2 = fir.call @f2() : () -> i1
    fir.result %ok2 : i1
  }
  return
}
