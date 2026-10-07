//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Non-temporal store instructions in LLVM IR (!nontemporal metadata) are
/// documented as optimization hints. However, on X86, regular stores follow
/// the TSO (Total Store Order) memory model, whereas non-temporal stores
/// (MOVNT*) write to weakly-ordered write-combining (WC) memory buffers that
/// bypass cache hierarchies.
///
/// Without explicit memory fences, write-combining stores are not ordered with
/// respect to other WC stores or normal WB stores. As a result, cross-thread
/// synchronization that follows a non-temporal store may allow other threads to
/// observe out-of-order writes, unless an SFENCE is executed.
///
/// This pass performs forward dataflow analysis to insert `llvm.x86.sse.sfence`
/// intrinsics between non-temporal stores and subsequent potential
/// synchronization points (such as atomic operations, function calls, exception
/// unwinding, and function returns), guaranteeing that weakly-ordered
/// non-temporal stores are serialized before synchronization, while still
/// allowing the performance benefits of non-temporal stores in a loop.
///
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86Subtarget.h"
#include "X86TargetMachine.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsX86.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include <optional>

using namespace llvm;

#define DEBUG_TYPE "x86-fence-nontemporal-stores"

static bool isNonTemporalStore(const Instruction &I) {
  return I.mayWriteToMemory() && I.hasMetadata(LLVMContext::MD_nontemporal);
}

/// Returns true if I is a hardware fence that serializes write-combining
/// stores:
/// - SFENCE (llvm.x86.sse.sfence) flushes and serializes WC stores.
/// - MFENCE (llvm.x86.sse2.mfence) serializes all memory operations.
///
/// Note: Atomic fences (such as `fence seq_cst` or `fence release`) do NOT
/// necessarily serialize write-combining stores.
static bool isNonTemporalFence(const Instruction &I) {
  if (const auto *CB = dyn_cast<CallBase>(&I)) {
    if (auto Intrin = CB->getIntrinsicID()) {
      return Intrin == Intrinsic::x86_sse_sfence ||
             Intrin == Intrinsic::x86_sse2_mfence;
    }
  }
  return false;
}

/// Returns true if I is a potential cross-thread synchronization point, or
/// which might return to a caller (either via return or throwing).
///
/// Returns true if I may give another thread a synchronizes-with edge that
/// makes this thread's earlier stores visible to it. This is
/// Instruction::maySynchronize, except that monotonic (and unordered) atomics
/// don't count, since on their own they can't do that (it takes a fence or a
/// release operation, and those still count), and neither do calls that don't
/// access memory.
static bool mayPublishStores(const Instruction &I) {
  if (const auto *CB = dyn_cast<CallBase>(&I))
    if (CB->doesNotAccessMemory())
      return false;
  AtomicOrdering Ordering = AtomicOrdering::NotAtomic;
  if (const auto *LI = dyn_cast<LoadInst>(&I))
    Ordering = LI->getOrdering();
  else if (const auto *SI = dyn_cast<StoreInst>(&I))
    Ordering = SI->getOrdering();
  else if (const auto *RMW = dyn_cast<AtomicRMWInst>(&I))
    Ordering = RMW->getOrdering();
  else if (const auto *CX = dyn_cast<AtomicCmpXchgInst>(&I))
    Ordering = CX->getMergedOrdering();
  if (Ordering != AtomicOrdering::NotAtomic &&
      !isStrongerThanMonotonic(Ordering))
    return false;
  return I.maySynchronize();
}

/// InvokeInst and CleanupReturnInst are handled as special cases, because
/// I.mayThrow() may return false for them, yet true for a CatchSwitchInst
/// unwind destination, which cannot have non-PHI instructions preceding it.
/// Fencing before the unwinding terminator keeps the catchswitch block clean.
static bool isSyncOrReturnPoint(const Instruction &I) {
  return isa<ReturnInst, InvokeInst, CleanupReturnInst>(&I) ||
         mayPublishStores(I) || I.mayThrow();
}

static bool containsNonTemporalStores(Function &F) {
  for (const BasicBlock &BB : F) {
    for (const Instruction &I : BB) {
      if (isNonTemporalStore(I))
        return true;
    }
  }
  return false;
}

static bool runImpl(Function &F, const X86Subtarget *ST) {
  // If the target lacks SSE1, non-temporal stores cannot use MOVNT instructions
  // (they lower to regular TSO stores), and SFENCE is not supported by
  // hardware.
  if (ST && !ST->hasSSE1())
    return false;

  // Fast path: if the function contains no non-temporal stores, do nothing.
  if (!containsNonTemporalStores(F))
    return false;

  // Compute the "dirty" (currently outstanding unfenced non-temporal stores)
  // state at entry of each basic block. Note that this is conservative: if a
  // block has multiple predecessors, we will insert a sfence if _any_ of the
  // predecessors had a nontemporal store, instead of splitting the critical
  // edge.
  //
  // The Flags bitfield keeps track of the state:
  // - If the block contains no state-altering instructions, Passthrough = true,
  //   and we'll then propagate DirtyOut based on DirtyIn
  // - If the last state-altering instruction is an NT store, DirtyOut = true
  // - If the last state-altering instruction is a fence or sync point (before
  //   which a fence would be added), DirtyOut = false.
  enum BlockFlags {
    Passthrough = 1,
    DirtyIn = 2,
    DirtyOut = 4,
  };
  DenseMap<const BasicBlock *, int> Flags;
  // Forward dataflow analysis to compute DirtyIn for all basic blocks.
  SmallSetVector<const BasicBlock *, 16> Worklist;

  for (const BasicBlock &BB : F) {
    int Flag = Passthrough;
    for (const Instruction &I : BB) {
      if (isNonTemporalStore(I)) {
        Flag = DirtyOut;
      } else if (isNonTemporalFence(I) || isSyncOrReturnPoint(I)) {
        Flag = 0;
      }
    }
    Flags[&BB] = Flag;

    if ((Flag & DirtyOut)) {
      for (const BasicBlock *Succ : successors(&BB)) {
        Worklist.insert(Succ);
      }
    }
  }

  // Iterate the worklist and update DirtyIn/DirtyOut as required, adding
  // successors to worklist whenever DirtyOut is modified.
  while (!Worklist.empty()) {
    const BasicBlock *BB = Worklist.pop_back_val();

    bool NewDirtyIn = false;
    for (const BasicBlock *Pred : predecessors(BB)) {
      if (Flags[Pred] & DirtyOut) {
        NewDirtyIn = true;
        break;
      }
    }
    if (NewDirtyIn && !(Flags[BB] & DirtyIn)) {
      Flags[BB] |= DirtyIn;
      if (Flags[BB] & Passthrough) {
        Flags[BB] |= DirtyOut;
        for (const BasicBlock *Succ : successors(BB))
          Worklist.insert(Succ);
      }
    }
  }

  // Finally, walk each block and insert sfence before any sync point reached in
  // a dirty state.
  bool Changed = false;
  IRBuilder<> Builder(F.getContext());

  // A remark per fence names the instruction that needed it and says when it
  // is inside a loop, where an sfence per iteration is expensive. Loop info is
  // only computed when remarks are enabled for this pass.
  OptimizationRemarkEmitter ORE(&F);
  std::optional<DominatorTree> DT;
  std::optional<LoopInfo> LI;
  if (ORE.allowExtraAnalysis(DEBUG_TYPE)) {
    DT.emplace(F);
    LI.emplace(*DT);
  }

  for (BasicBlock &BB : F) {
    bool CurrentDirty = Flags[&BB] & DirtyIn;
    for (Instruction &I : llvm::make_early_inc_range(BB)) {
      if (isNonTemporalStore(I)) {
        CurrentDirty = true;
      } else if (isNonTemporalFence(I)) {
        CurrentDirty = false;
      } else if (isSyncOrReturnPoint(I)) {
        if (CurrentDirty) {
          Builder.SetInsertPoint(&I);
          Builder.SetCurrentDebugLocation(I.getDebugLoc());
          Builder.CreateIntrinsic(Intrinsic::x86_sse_sfence, {});
          ORE.emit([&]() {
            OptimizationRemark R(DEBUG_TYPE, "SFenceInserted", &I);
            R << "inserted sfence before ";
            if (const auto *CB = dyn_cast<CallBase>(&I)) {
              if (const Function *Callee = CB->getCalledFunction())
                R << "call to " << ore::NV("Callee", Callee);
              else
                R << (CB->isInlineAsm() ? "inline asm" : "indirect call");
            } else {
              R << ore::NV("Inst", I.getOpcodeName());
            }
            R << " to order earlier non-temporal stores";
            if (LI && LI->getLoopFor(&BB))
              R << "; it is inside a loop, so it can run on every iteration";
            return R;
          });
          CurrentDirty = false;
          Changed = true;
        }
      }
    }
  }

  return Changed;
}

namespace {

class X86FenceNonTemporalStoresLegacy : public FunctionPass {
public:
  static char ID;

  X86FenceNonTemporalStoresLegacy() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override {
    // No skipFunction: the fences are needed for correctness, including in
    // optnone functions.
    auto *TPC = getAnalysisIfAvailable<TargetPassConfig>();
    const X86Subtarget *ST =
        TPC ? &TPC->getTM<X86TargetMachine>().getSubtarget<X86Subtarget>(F)
            : nullptr;
    return runImpl(F, ST);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
  }

  StringRef getPassName() const override {
    return "X86 Fence Non-Temporal Stores";
  }
};

} // end anonymous namespace

char X86FenceNonTemporalStoresLegacy::ID = 0;

INITIALIZE_PASS(X86FenceNonTemporalStoresLegacy, DEBUG_TYPE,
                "X86 Fence Non-Temporal Stores", false, false)

FunctionPass *llvm::createX86FenceNonTemporalStoresLegacyPass() {
  return new X86FenceNonTemporalStoresLegacy();
}

PreservedAnalyses
X86FenceNonTemporalStoresPass::run(Function &F, FunctionAnalysisManager &FAM) {
  const X86Subtarget *ST = TM ? TM->getSubtargetImpl(F) : nullptr;
  if (!runImpl(F, ST))
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
