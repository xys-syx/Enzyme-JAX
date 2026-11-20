// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(split-branched-accesses,canonicalize)" | FileCheck %s

// An access at an index a branch chose is done in each arm instead, at the
// value that arm chose: a constant, or a value in hand at the access. The
// index arithmetic between the branch and the access is redone in each arm.

#set = affine_set<()[s0] : (s0 >= 1)>
module {

  func.func @scf_load(%m: memref<?xi32>, %c: i1) -> i32 {
    %c3 = arith.constant 3 : index
    %c7 = arith.constant 7 : index
    %i = scf.if %c -> index { scf.yield %c3 : index } else { scf.yield %c7 : index }
    %v = memref.load %m[%i] : memref<?xi32>
    return %v : i32
  }
  func.func @affine_store(%m: memref<?xf64>, %s: index, %val: f64) {
    %c3 = arith.constant 3 : index
    %c7 = arith.constant 7 : index
    %i = affine.if #set()[%s] -> index { affine.yield %c3 : index } else { affine.yield %c7 : index }
    affine.store %val, %m[%i] : memref<?xf64>
    return
  }
  func.func @scf_store(%m: memref<?xi32>, %c: i1, %val: i32) {
    %c3 = arith.constant 3 : index
    %c7 = arith.constant 7 : index
    %i = scf.if %c -> index { scf.yield %c3 : index } else { scf.yield %c7 : index }
    memref.store %val, %m[%i] : memref<?xi32>
    return
  }

  // an arm that hands back a value in hand at the access is done at it
  func.func @dynamic_arm(%m: memref<?xi32>, %c: i1, %d: index) -> i32 {
    %c3 = arith.constant 3 : index
    %i = scf.if %c -> index {
      scf.yield %c3 : index
    } else {
      scf.yield %d : index
    }
    %v = memref.load %m[%i] : memref<?xi32>
    return %v : i32
  }

  // an arm that computes its value inside is left alone
  func.func @computed_arm(%m: memref<?xi32>, %c: i1, %d: index) -> i32 {
    %c3 = arith.constant 3 : index
    %i = scf.if %c -> index {
      scf.yield %c3 : index
    } else {
      %e = arith.addi %d, %c3 : index
      scf.yield %e : index
    }
    %v = memref.load %m[%i] : memref<?xi32>
    return %v : i32
  }

  // the index computed from the branch's result: a layout's row chosen by
  // a flag, q + NQ * (row + 3 * e) with row 2 or 3
  func.func @index_chain(%m: memref<?xf64>, %s: index, %nq: index, %q: index, %e: index, %val: f64) {
    %c3 = arith.constant 3 : index
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %row = affine.if #set()[%s] -> i32 { affine.yield %c2_i32 : i32 } else { affine.yield %c3_i32 : i32 }
    %rowi = arith.index_cast %row : i32 to index
    %e3 = arith.muli %e, %c3 : index
    %r = arith.addi %rowi, %e3 : index
    %off = arith.muli %r, %nq : index
    %i = arith.addi %q, %off : index
    memref.store %val, %m[%i] : memref<?xf64>
    return
  }

  // the memref and the value stored are computed after the branch: the branch
  // is asked again where the access stands, so nothing has to move
  func.func @operands_after_branch(%p: !llvm.ptr, %s: index, %x: f64) {
    %c3 = arith.constant 3 : index
    %c7 = arith.constant 7 : index
    %i = affine.if #set()[%s] -> index { affine.yield %c3 : index } else { affine.yield %c7 : index }
    %m = "enzymexla.pointer2memref"(%p) : (!llvm.ptr) -> memref<?xf64>
    %v = arith.addf %x, %x : f64
    memref.store %v, %m[%i] : memref<?xf64>
    return
  }

  // an affine access whose index arithmetic stays in the index type has a
  // valid index in each arm and is split
  func.func @affine_index_chain(%m: memref<?xi32>, %s: index, %d: index) -> i32 {
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %x = affine.if #set()[%s] -> index { affine.yield %c3 : index } else { affine.yield %d : index }
    %i = arith.addi %x, %c1 : index
    %v = affine.load %m[symbol(%i)] : memref<?xi32>
    return %v : i32
  }

  // index arithmetic through i32 gives no valid affine index inside an arm,
  // as a symbol or as a dimension, so these accesses are left alone
  func.func @affine_index_from_i32(%m: memref<?xi32>, %s: index, %a: i32) -> i32 {
    %c3_i32 = arith.constant 3 : i32
    %x = affine.if #set()[%s] -> i32 { affine.yield %a : i32 } else { affine.yield %c3_i32 : i32 }
    %e = arith.extui %x : i32 to i64
    %i = arith.index_cast %e : i64 to index
    %v = affine.load %m[symbol(%i)] : memref<?xi32>
    %w = affine.load %m[%i] : memref<?xi32>
    %r = arith.addi %v, %w : i32
    return %r : i32
  }
}

// CHECK:  func.func @scf_load(%[[v1:.+]]: memref<?xi32>, %[[v2:.+]]: i1) -> i32 {
// CHECK-NEXT:  %[[v3:.+]] = arith.constant 3 : index
// CHECK-NEXT:  %[[v4:.+]] = arith.constant 7 : index
// CHECK-NEXT:  %[[v5:.+]] = scf.if %[[v2]] -> (i32) {
// CHECK-NEXT:  %[[v6:.+]] = memref.load %[[v1]][%[[v3]]] : memref<?xi32>
// CHECK-NEXT:  scf.yield %[[v6]] : i32
// CHECK-NEXT:  } else {
// CHECK-NEXT:  %[[v7:.+]] = memref.load %[[v1]][%[[v4]]] : memref<?xi32>
// CHECK-NEXT:  scf.yield %[[v7]] : i32
// CHECK-NEXT:  }
// CHECK-NEXT:  return %[[v5]] : i32
// CHECK-NEXT:  }

// CHECK:  func.func @affine_store(%[[v1:.+]]: memref<?xf64>, %[[v2:.+]]: index, %[[v3:.+]]: f64) {
// CHECK-NEXT:  affine.if #set()[%[[v2]]] {
// CHECK-NEXT:  affine.store %[[v3]], %[[v1]][3] : memref<?xf64>
// CHECK-NEXT:  } else {
// CHECK-NEXT:  affine.store %[[v3]], %[[v1]][7] : memref<?xf64>
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT:  }

// CHECK:  func.func @scf_store(%[[v1:.+]]: memref<?xi32>, %[[v2:.+]]: i1, %[[v3:.+]]: i32) {
// CHECK-NEXT:  %[[v4:.+]] = arith.constant 3 : index
// CHECK-NEXT:  %[[v5:.+]] = arith.constant 7 : index
// CHECK-NEXT:  scf.if %[[v2]] {
// CHECK-NEXT:  memref.store %[[v3]], %[[v1]][%[[v4]]] : memref<?xi32>
// CHECK-NEXT:  } else {
// CHECK-NEXT:  memref.store %[[v3]], %[[v1]][%[[v5]]] : memref<?xi32>
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT:  }

// CHECK:  func.func @dynamic_arm(%[[v1:.+]]: memref<?xi32>, %[[v2:.+]]: i1, %[[v3:.+]]: index) -> i32 {
// CHECK-NEXT:  %[[v4:.+]] = arith.constant 3 : index
// CHECK-NEXT:  %[[v5:.+]] = scf.if %[[v2]] -> (i32) {
// CHECK-NEXT:  %[[v6:.+]] = memref.load %[[v1]][%[[v4]]] : memref<?xi32>
// CHECK-NEXT:  scf.yield %[[v6]] : i32
// CHECK-NEXT:  } else {
// CHECK-NEXT:  %[[v7:.+]] = memref.load %[[v1]][%[[v3]]] : memref<?xi32>
// CHECK-NEXT:  scf.yield %[[v7]] : i32
// CHECK-NEXT:  }
// CHECK-NEXT:  return %[[v5]] : i32
// CHECK-NEXT:  }

// CHECK:  func.func @computed_arm(%[[v1:.+]]: memref<?xi32>, %[[v2:.+]]: i1, %[[v3:.+]]: index) -> i32 {
// CHECK-NEXT:  %[[v4:.+]] = arith.constant 3 : index
// CHECK-NEXT:  %[[v5:.+]] = scf.if %[[v2]] -> (index) {
// CHECK-NEXT:  scf.yield %[[v4]] : index
// CHECK-NEXT:  } else {
// CHECK-NEXT:  %[[v6:.+]] = arith.addi %[[v3]], %[[v4]] : index
// CHECK-NEXT:  scf.yield %[[v6]] : index
// CHECK-NEXT:  }
// CHECK-NEXT:  %[[v7:.+]] = memref.load %[[v1]][%[[v5]]] : memref<?xi32>
// CHECK-NEXT:  return %[[v7]] : i32
// CHECK-NEXT:  }

// CHECK:  func.func @index_chain(%[[v1:.+]]: memref<?xf64>, %[[v2:.+]]: index, %[[v3:.+]]: index, %[[v4:.+]]: index, %[[v5:.+]]: index, %[[v6:.+]]: f64) {
// CHECK-NEXT:  %[[v7:.+]] = arith.constant 2 : index
// CHECK-NEXT:  %[[v8:.+]] = arith.constant 3 : index
// CHECK-NEXT:  %[[v9:.+]] = arith.muli %[[v5]], %[[v8]] : index
// CHECK-NEXT:  affine.if #set()[%[[v2]]] {
// CHECK-NEXT:  %[[v10:.+]] = arith.addi %[[v9]], %[[v7]] : index
// CHECK-NEXT:  %[[v11:.+]] = arith.muli %[[v10]], %[[v3]] : index
// CHECK-NEXT:  %[[v12:.+]] = arith.addi %[[v4]], %[[v11]] : index
// CHECK-NEXT:  memref.store %[[v6]], %[[v1]][%[[v12]]] : memref<?xf64>
// CHECK-NEXT:  } else {
// CHECK-NEXT:  %[[v13:.+]] = arith.addi %[[v9]], %[[v8]] : index
// CHECK-NEXT:  %[[v14:.+]] = arith.muli %[[v13]], %[[v3]] : index
// CHECK-NEXT:  %[[v15:.+]] = arith.addi %[[v4]], %[[v14]] : index
// CHECK-NEXT:  memref.store %[[v6]], %[[v1]][%[[v15]]] : memref<?xf64>
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT:  }

// CHECK:  func.func @operands_after_branch(%[[o1:.+]]: !llvm.ptr, %[[o2:.+]]: index, %[[o3:.+]]: f64) {
// CHECK-NEXT:  %[[o4:.+]] = arith.constant 3 : index
// CHECK-NEXT:  %[[o5:.+]] = arith.constant 7 : index
// CHECK-NEXT:  %[[o6:.+]] = "enzymexla.pointer2memref"(%[[o1]]) : (!llvm.ptr) -> memref<?xf64>
// CHECK-NEXT:  %[[o7:.+]] = arith.addf %[[o3]], %[[o3]] : f64
// CHECK-NEXT:  affine.if #set()[%[[o2]]] {
// CHECK-NEXT:  memref.store %[[o7]], %[[o6]][%[[o4]]] : memref<?xf64>
// CHECK-NEXT:  } else {
// CHECK-NEXT:  memref.store %[[o7]], %[[o6]][%[[o5]]] : memref<?xf64>
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT:  }

// CHECK:  func.func @affine_index_chain(%[[v1:.+]]: memref<?xi32>, %[[v2:.+]]: index, %[[v3:.+]]: index) -> i32 {
// CHECK-NEXT:  %[[v4:.+]] = arith.constant 1 : index
// CHECK-NEXT:  %[[v5:.+]] = affine.if #set()[%[[v2]]] -> i32 {
// CHECK-NEXT:  %[[v6:.+]] = affine.load %[[v1]][4] : memref<?xi32>
// CHECK-NEXT:  affine.yield %[[v6]] : i32
// CHECK-NEXT:  } else {
// CHECK-NEXT:  %[[v7:.+]] = arith.addi %[[v3]], %[[v4]] : index
// CHECK-NEXT:  %[[v8:.+]] = affine.load %[[v1]][symbol(%[[v7]])] : memref<?xi32>
// CHECK-NEXT:  affine.yield %[[v8]] : i32
// CHECK-NEXT:  }
// CHECK-NEXT:  return %[[v5]] : i32
// CHECK-NEXT:  }

// CHECK:  func.func @affine_index_from_i32(%[[v1:.+]]: memref<?xi32>, %[[v2:.+]]: index, %[[v3:.+]]: i32) -> i32 {
// CHECK-NEXT:  %[[v4:.+]] = arith.constant 3 : i32
// CHECK-NEXT:  %[[v5:.+]] = affine.if #set()[%[[v2]]] -> i32 {
// CHECK-NEXT:  affine.yield %[[v3]] : i32
// CHECK-NEXT:  } else {
// CHECK-NEXT:  affine.yield %[[v4]] : i32
// CHECK-NEXT:  }
// CHECK-NEXT:  %[[v6:.+]] = arith.extui %[[v5]] : i32 to i64
// CHECK-NEXT:  %[[v7:.+]] = arith.index_cast %[[v6]] : i64 to index
// CHECK-NEXT:  %[[v8:.+]] = affine.load %[[v1]][symbol(%[[v7]])] : memref<?xi32>
// CHECK-NEXT:  %[[v9:.+]] = affine.load %[[v1]][symbol(%[[v7]])] : memref<?xi32>
// CHECK-NEXT:  %[[v10:.+]] = arith.addi %[[v8]], %[[v9]] : i32
// CHECK-NEXT:  return %[[v10]] : i32
// CHECK-NEXT:  }
