include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Alg_Dialect : Dialect {
    let name = "alg";
    let summary = "alg";
    let description = "alg";
    let cppNamespace = "::mlir::alg";

    let extraClassDeclaration = [{
        void registerTypes();
    }];

    let useDefaultTypePrinterParser = 1;
    let useDefaultAttributePrinterParser = 1;
}

class Alg_Op<string mnemonic, list<NativeOpTrait> traits = []> : Op<Alg_Dialect, mnemonic, traits>;

class Alg_Type<string name, string typeMnemonic, list<Trait> traits = []> : TypeDef<Alg_Dialect, name, traits> {
    let mnemonic = typeMnemonic;
}

class Alg_Attr<string name, string typeMnemonic, list<Trait> traits = []> : AttrDef<Alg_Dialect, name, traits> {
    let mnemonic = typeMnemonic;
} 

def Alg_GroupAttr : Alg_Attr<"Group", "grp"> {
    let summary="Algebraic group type.";
    let description = [{
        Represents an abstract algebraic group. 
        May optionally carry parameters, e.g., group name or dimension.
    }];

    let parameters = (ins "StringAttr":$name);

    let assemblyFormat = "`<` $name `>`";
}

def Alg_Element : Alg_Type<"Element", "elem"> {
    let summary = "An element of a group";
    let description = [{
        Represents an element of an algebraic group, by persisting
        the group the element belongs to, and a representation in
        in terms of the generators.
    }];

    let parameters = (ins
        Alg_GroupAttr:$group,
        Builtin_Vector:$repr
    );

    let assemblyFormat = "`<` $group `:` $repr `>`";
}

def ElementCreateOp : Alg_Op<"element.create"> {
    let arguments = (ins
        ArrayAttr:$data,
        Alg_GroupAttr:$group
    );

    let results = (outs
        Alg_Element:$elem
    );
}

def AddOp : Alg_Op<"add"> {
    let summary = "add";

    let description = [{
        Adds two elements of the same group.
    }];

    let arguments = (ins
        Alg_Element:$elem1,
        Alg_Element:$elem2
    );

    let results = (outs
        Alg_Element:$result
    );

    let hasVerifier = 1;
}

def IntMul : Alg_Op<"sc_mul"> {
    let summary = "Multiply a group elem by an integer.";

    let description = [{
        Multiply an element of a group by an integer, as shorthand for
        repeated addition.
    }];

    let arguments = (ins
        Alg_Element:$elem,
        Builtin_Integer:$mult
    );

    let results = (outs
        Alg_Element:$result
    );
}

def UnaryNeg : Alg_Op<"un_neg"> {
    let summary = "Unary negation of group elements (i.e. group inverse)";

    let description = [{
        Construct the inverse of an element.
    }];

    let arguments = (ins
        Alg_Element:$elem
    );

    let results = (outs
        Alg_Element:$result
    );
}

def UnaryConstrOp : Alg_Op<"constr"> {
    let summary = "Introduces constraints on group elements";

    let description = [{
        Introduces constraints on group elements. From that point on,
        the constraint will be uses for any further simplifications.
    }];

    let arguments = (ins Alg_Element:$elem);
}