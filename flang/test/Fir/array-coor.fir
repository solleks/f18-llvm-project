// RUN: tco %s | FileCheck %s

func @array_coor_box_value(%29 : !fir.box<!fir.array<2xf64>>,
     			   %33 : index) -> f64 {
  %34 = fir.array_coor %29 %33 : (!fir.box<!fir.array<2xf64>>, index) ->
      		       	          !fir.ref<f64>
  %35 = fir.load %34 : !fir.ref<f64>
  return %35 : f64
}

// CHECK-LABEL: define double @array_coor_box_value
// CHECK: %[[t3:.*]] = sub i64 %{{.*}}, 1
// CHECK: %[[t4:.*]] = mul i64 %[[t3]], 1
// CHECK: %[[t5:.*]] = getelementptr { [2 x double]*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, { [2 x double]*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }* %{{.*}}, i32 0, i32 7, i64 0, i32 2
// CHECK: %[[t6:.*]] = load i64, i64* %[[t5]]
// CHECK: %[[t7:.*]] = mul i64 %[[t4]], %[[t6]]
// CHECK: %[[t8:.*]] = add i64 %[[t7]], 0
// CHECK: %[[t9:.*]] = getelementptr { [2 x double]*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, { [2 x double]*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }* %{{.*}}, i32 0, i32 0
// CHECK: %[[t10:.*]] = load [2 x double]*, [2 x double]** %[[t9]]
// CHECK: %[[t11:.*]] = bitcast [2 x double]* %[[t10]] to i8*
// CHECK: %[[t12:.*]] = getelementptr i8, i8* %[[t11]], i64 %[[t8]]
// CHECK: %[[t13:.*]] = bitcast i8* %[[t12]] to double*
// CHECK: %[[t14:.*]] = load double, double* %[[t13]]
// CHECK: ret double %[[t14]]
