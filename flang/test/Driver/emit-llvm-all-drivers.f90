! RUN: %flang_fc1 -emit-llvm %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-mlir %s -o - | tco -o - | FileCheck %s
! RUN: bbc %s -o - | tco -o - | FileCheck %s

! This is added as a regression test for a failure reported in
! https://github.com/flang-compiler/f18-llvm-project/issues/1280. It makes sure
! that this example can be lowered to LLVM IR by all major drivers in Flang.
! Ultimately, we should be verifying that the generated code is identical in all
! the tested cases. Currently, there are some discrepancies in the generated
! debug info.
!TODO: Make sure that the generated code is identical in all cases

! CHECK: ; ModuleID = 'FIRModule'

MODULE test_module
    REAL(8_4), ALLOCATABLE, DIMENSION(:) :: mu
    CONTAINS
    SUBROUTINE test_subr ( istat )

        INTEGER(4_4), INTENT(INOUT) :: istat
        IF ( istat > 0 ) RETURN
        w = zero

    END SUBROUTINE test_subr
END MODULE test_module
