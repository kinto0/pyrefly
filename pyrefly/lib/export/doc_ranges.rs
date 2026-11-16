/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Building the `DocRanges` of a definition from its AST.

use pyrefly_python::ast::Ast;
use pyrefly_python::docstring::DocRanges;
use pyrefly_python::docstring::Docstring;
use pyrefly_python::dunder;
use ruff_python_ast::Expr;
use ruff_python_ast::Parameters;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtClassDef;
use ruff_python_ast::StmtFunctionDef;
use ruff_text_size::TextRange;

use crate::export::special::SpecialExport;

pub fn function_doc_ranges(x: &StmtFunctionDef) -> DocRanges {
    doc_ranges_from_parts(Docstring::range_from_stmts(&x.body), &x.parameters)
}

/// A class's parameters are those of its `__init__`.
pub fn class_doc_ranges(x: &StmtClassDef) -> DocRanges {
    let docstring = Docstring::range_from_stmts(&x.body);
    match x.body.iter().rev().find_map(|stmt| match stmt {
        Stmt::FunctionDef(init) if init.name.id == dunder::INIT => Some(init),
        _ => None,
    }) {
        Some(init) => doc_ranges_from_parts(docstring, &init.parameters),
        None => DocRanges::of_docstring(docstring),
    }
}

/// The documentation of a function with the given docstring and parameters.
pub fn doc_ranges_from_parts(docstring: Option<TextRange>, parameters: &Parameters) -> DocRanges {
    DocRanges::new(
        docstring,
        parameters
            .iter()
            .filter_map(|param| {
                Some((param.name().id.clone(), annotated_doc(param.annotation()?)?))
            })
            .collect(),
    )
}

/// The range of the string literal in the `Doc("...")` metadata of an `Annotated[...]`
/// annotation, as in `name: Annotated[str, Doc("The user's name")]`. This is the
/// [annotated-doc](https://github.com/fastapi/annotated-doc) convention that FastAPI uses.
/// `Annotated` and `Doc` are matched with `SpecialExport::from_syntax`, because this runs before
/// bindings exist. When there is more than one `Doc`, the last applies, as PEP 727 specifies. An
/// implicitly concatenated string is skipped, because it has no single literal to read back.
fn annotated_doc(annotation: &Expr) -> Option<TextRange> {
    let Expr::Subscript(annotated) = annotation else {
        return None;
    };
    if SpecialExport::from_syntax(&annotated.value) != Some(SpecialExport::Annotated) {
        return None;
    }
    Ast::unpack_slice(&annotated.slice)
        .iter()
        .skip(1)
        .rev()
        .find_map(|metadata| match metadata {
            Expr::Call(call)
                if SpecialExport::from_syntax(&call.func) == Some(SpecialExport::Doc)
                    && let [Expr::StringLiteral(text)] = &*call.arguments.args =>
            {
                (!text.value.is_implicit_concatenated()).then_some(text.range)
            }
            _ => None,
        })
}

#[cfg(test)]
mod tests {
    use pyrefly_python::ast::Ast;
    use pyrefly_python::docstring::clean_annotated_doc;
    use ruff_python_ast::PySourceType;
    use ruff_python_ast::Stmt;

    use crate::export::doc_ranges::class_doc_ranges;
    use crate::export::doc_ranges::function_doc_ranges;

    /// The `Doc` text of each documented parameter of the first statement of `source`.
    fn annotated_docs(source: &str) -> Vec<(String, String)> {
        let (module, _, _) = Ast::parse(source, PySourceType::Python);
        let doc_ranges = match module.body.first() {
            Some(Stmt::FunctionDef(x)) => function_doc_ranges(x),
            Some(Stmt::ClassDef(x)) => class_doc_ranges(x),
            _ => panic!("expected a function or class definition"),
        };
        doc_ranges
            .parameters()
            .iter()
            .map(|(name, range)| {
                let doc = clean_annotated_doc(&source[*range]).expect("non-blank `Doc` text");
                (name.to_string(), doc)
            })
            .collect()
    }

    #[test]
    fn test_doc_ranges_parameters() {
        let source = r#"
def f(
    a: Annotated[int, Doc("plain name")],
    b: typing.Annotated[int, annotated_doc.Doc('attribute name')],
    c: Annotated[int, Doc("first"), Doc("last")],
    *args: Annotated[int, Doc("varargs")],
    d: Annotated[int, "not a Doc"],
    e: int,
    f: Annotated[int, Doc(f"not a literal")],
    g: list[Doc("not Annotated")],
    h: Annotated[int, Doc("implicitly " "concatenated")],
): ...
"#;
        assert_eq!(
            annotated_docs(source),
            vec![
                ("a".to_owned(), "plain name".to_owned()),
                ("b".to_owned(), "attribute name".to_owned()),
                ("c".to_owned(), "last".to_owned()),
                ("args".to_owned(), "varargs".to_owned()),
            ]
        );
    }

    #[test]
    fn test_doc_ranges_class_parameters_come_from_init() {
        let source = r#"
class C:
    """Docstring."""
    def __init__(self, a: Annotated[int, Doc("from init")]) -> None: ...
    def other(self, b: Annotated[int, Doc("not from init")]) -> None: ...
"#;
        assert_eq!(
            annotated_docs(source),
            vec![("a".to_owned(), "from init".to_owned())]
        );
    }

    #[test]
    fn test_doc_ranges_parameters_from_other_modules() {
        let source = r#"
def f(
    a: other.Annotated[int, Doc("other Annotated")],
    b: Annotated[int, other.Doc("other Doc")],
    c: typing_extensions.Annotated[int, typing_extensions.Doc("typing_extensions")],
): ...
"#;
        assert_eq!(
            annotated_docs(source),
            vec![("c".to_owned(), "typing_extensions".to_owned())]
        );
    }
}
