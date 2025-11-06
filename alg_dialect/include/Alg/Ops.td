include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/AttrTypeBase.td"

def Alg_Dialect : Dialect {
    let name = "alg";
    let summary = "alg";
    let description = "alg";
    let cppNamespace = "::mlir::alg";

    let extraClassDeclaration = [{
        void registerTypes();
    }];
}

class Alg_Type<string name, string typeMnemonic, list<Trait> traits = []> : TypeDef<Alg_Dialect, name, traits> {
    let mnemonic = typeMnemonic;
}

def Alg_Group : Alg_Type<"Group", "grp"> {
    let summary="Algebraic group type.";
    let description = [{
        Represents an abstract algebraic group. 
        May optionally carry parameters, e.g., group name or dimension.
    }];

    let parameters = (ins OptionalParameter<"StringAttr">:$name);

    let assemblyFormat = "`<` $name `>`";
}

def Alg_Element : Alg_Type<"Element", "elt"> {
    let summary = "An element of a specific group";

    let description = [{
        Represents an element belonging to a specific Alg_Group.
    }];

    let parameters = (ins TypeParameter<"GroupType", "grp">:$group);

    let assemblyFormat = "`<` $group `>`";
}

// This simply specifies a new `Op` class, that inherits from Op.
class Alg_Op<string mnemonic, list<NativeOpTrait> traits = []> : Op<Alg_Dialect, mnemonic, traits>;

// Now, we simply define an instance of the class we have just created, with no traits.
def IdentityOp : Alg_Op<"id"> {
    let summary = "Return the identity of a group.";

    let description = [{
        Obtain the identity of a group.
    }];
}

def CompositionOp : Alg_Op<"composition"> {
    let summary = "Combine two elements of a group.";

    let description = [{
        Describes the combination of to Alg_Element objects.
    }];

    let arguments = (ins
        Alg_Element:$elt1,
        Alg_Element:$elt2
    );
}