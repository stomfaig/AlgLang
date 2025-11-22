#ifndef ALG_FRONTEND_LOWER_CPP
#define ALG_FRONTEND_LOWER_CPP


#include "lower.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "Alg/AlgOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Pass/Pass.h"


#include "Alg/AlgTypes.h"

std::unique_ptr<mlir::Pass> createLowerToLLVMPass();

/// Rewrite pattern for lowring alg::ElementCreateOp to `arith` and `vector` dialects.
struct ElementCreateOpLowering : public mlir::OpConversionPattern<mlir::alg::ElementCreateOp> {
    using OpConversionPattern<mlir::alg::ElementCreateOp>::OpConversionPattern;

    virtual mlir::LogicalResult matchAndRewrite(mlir::alg::ElementCreateOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) const final {
        auto Loc = op->getLoc();

        assert(op->getNumResults() == 1);

        auto resultType = *op->result_type_begin();

        auto elemType = mlir::dyn_cast<mlir::alg::ElementType>(resultType);

        if (!elemType) {
            llvm::errs() << "Not an Alg_Element type\n";
            return llvm::failure();
        }

        auto dataType = elemType.getRepr();
        
        auto data = op.getDataAttr().getValue();

        llvm::SmallVector<mlir::Value> elements;
        elements.reserve(data.size());

        for (mlir::Attribute attr : data) {
            auto ConstantOp = mlir::arith::ConstantOp::create(rewriter, Loc, mlir::cast<mlir::TypedAttr>(attr));
            elements.push_back(ConstantOp);
        }

        auto CreateOp =  mlir::vector::FromElementsOp::create(rewriter, Loc, dataType, elements);

        rewriter.replaceOp(op, CreateOp->getResults());

        return llvm::success();
    }
};

struct AddOpLowering : public mlir::OpConversionPattern<mlir::alg::AddOp> {
    using mlir::OpConversionPattern<mlir::alg::AddOp>::OpConversionPattern;

    virtual mlir::LogicalResult matchAndRewrite(mlir::alg::AddOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) const final {
        
        auto Loc = op->getLoc();

        auto elem1 = adaptor.getElem1();
        auto elem2 = adaptor.getElem2();

        elem1.print(llvm::outs());
        elem2.print(llvm::outs());

        auto AddOp = mlir::arith::AddIOp::create(rewriter, Loc, elem1, elem2);

        rewriter.replaceOp(op, AddOp);

        return mlir::success();
    }
};

/// "Lower" UnaryNeg, by rewriting it to an IntMul operation.
/// There is no actual lowering here, we simply "lower" to operations that can be properly lowered.
// This might be removed later in favour of doing this in mlirGen.
struct UnaryNegLowering : public mlir::OpConversionPattern<mlir::alg::UnaryNeg> {
    using mlir::OpConversionPattern<mlir::alg::UnaryNeg>::OpConversionPattern;

    virtual mlir::LogicalResult matchAndRewrite(mlir::alg::UnaryNeg op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) const final {
        auto Loc = op->getLoc();

        auto Type = rewriter.getI32Type();

        auto ConstOp = mlir::arith::ConstantIntOp::create(rewriter, Loc, Type, -1);
        auto ConstVal = ConstOp.getResult();
        auto IntMulOp = mlir::alg::IntMul::create(rewriter, Loc, op.getElem().getType(), op.getElem(), ConstVal);

        rewriter.replaceOp(op, IntMulOp);

        return mlir::success();
    }
};

struct IntMulLowering : public mlir::OpConversionPattern<mlir::alg::IntMul> {
    using mlir::OpConversionPattern<mlir::alg::IntMul>::OpConversionPattern;

    virtual mlir::LogicalResult matchAndRewrite(mlir::alg::IntMul op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) const final {
        
        auto Loc = op->getLoc();

        auto BroadcastOp = mlir::vector::BroadcastOp::create(rewriter, Loc, adaptor.getElem().getType(), adaptor.getMult());
        auto BroadcastVec = BroadcastOp.getResult();

        auto MulIOp = mlir::arith::MulIOp::create(rewriter, Loc, BroadcastVec, adaptor.getElem());

        rewriter.replaceOp(op, MulIOp);

        return mlir::success();
    }   
};

void AlgLoweringPass::runOnOperation() {

    // This defines a vanilla conversion target, that we will use.
    // We could define a class inheriting from `ConversionTarget`, which
    // "self-populates", but for testing this is fine.
    mlir::ConversionTarget target(getContext());
    
    target.addLegalDialect<mlir::arith::ArithDialect, mlir::vector::VectorDialect>();
    //target.addIllegalDialect<mlir::alg::AlgDialect>();

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<
        ElementCreateOpLowering,
        AddOpLowering,
        IntMulLowering,
        UnaryNegLowering
    >(&getContext());

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns)))) {
        signalPassFailure();
    }
    
}

#endif // ALG_FRONTEND_LOWER_CPP