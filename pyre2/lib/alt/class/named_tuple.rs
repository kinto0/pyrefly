/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use ruff_python_ast::name::Name;
use starlark_map::smallmap;

use crate::alt::answers::AnswersSolver;
use crate::alt::answers::LookupAnswer;
use crate::alt::types::class_metadata::ClassSynthesizedField;
use crate::alt::types::class_metadata::ClassSynthesizedFields;
use crate::dunder;
use crate::types::callable::Callable;
use crate::types::callable::FuncId;
use crate::types::callable::Function;
use crate::types::callable::FunctionKind;
use crate::types::callable::Param;
use crate::types::callable::ParamList;
use crate::types::callable::Required;
use crate::types::class::Class;
use crate::types::class::ClassType;
use crate::types::literal::Lit;
use crate::types::tuple::Tuple;
use crate::types::types::Type;
use crate::util::prelude::SliceExt;

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    pub fn get_named_tuple_elements(&self, cls: &Class) -> Vec<Name> {
        let mut elements = Vec::new();
        for name in cls.fields() {
            if let Some(range) = cls.field_decl_range(name) {
                elements.push((name.clone(), range));
            }
        }
        elements.sort_by_key(|e| e.1.start());
        elements.iter().map(|e| e.0.clone()).collect()
    }

    pub fn named_tuple_element_types(&self, cls: &ClassType) -> Option<Vec<Type>> {
        let class_metadata = self.get_metadata_for_class(cls.class_object());
        let named_tuple_metadata = class_metadata.named_tuple_metadata()?;
        Some(
            named_tuple_metadata
                .elements
                .iter()
                .filter_map(|name| {
                    let attr = self.try_lookup_attr(&Type::ClassType(cls.clone()), name)?;
                    self.resolve_as_instance_method(attr)
                })
                .collect(),
        )
    }

    fn get_named_tuple_field_params(&self, cls: &Class, elements: &[Name]) -> Vec<Param> {
        elements.map(|name| {
            let member = &*self.get_class_member(cls, name).unwrap().value;
            Param::Pos(
                name.clone(),
                member.as_named_tuple_type(),
                member.as_named_tuple_requiredness(),
            )
        })
    }

    fn get_named_tuple_new(&self, cls: &Class, elements: &[Name]) -> ClassSynthesizedField {
        let mut params = vec![Param::Pos(
            Name::new("cls"),
            Type::Type(Box::new(cls.self_type())),
            Required::Required,
        )];
        params.extend(self.get_named_tuple_field_params(cls, elements));
        let ty = Type::Function(Box::new(Function {
            signature: Callable::list(ParamList::new(params), cls.self_type()),
            metadata: FunctionKind::Def(Box::new(FuncId {
                module: self.module_info().name(),
                cls: Some(cls.name().clone()),
                func: dunder::NEW,
            })),
        }));
        ClassSynthesizedField::new(ty)
    }

    fn get_named_tuple_init(&self, cls: &Class, elements: &[Name]) -> ClassSynthesizedField {
        let mut params = vec![cls.self_param()];
        params.extend(self.get_named_tuple_field_params(cls, elements));
        let ty = Type::Function(Box::new(Function {
            signature: Callable::list(ParamList::new(params), cls.self_type()),
            metadata: FunctionKind::Def(Box::new(FuncId {
                module: self.module_info().name(),
                cls: Some(cls.name().clone()),
                func: dunder::INIT,
            })),
        }));
        ClassSynthesizedField::new(ty)
    }

    fn get_named_tuple_iter(&self, cls: &Class, elements: &[Name]) -> ClassSynthesizedField {
        let params = vec![cls.self_param()];
        let element_types: Vec<Type> = elements
            .iter()
            .map(|name| (*self.get_class_member(cls, name).unwrap().value).as_named_tuple_type())
            .collect();
        let ty = Type::Function(Box::new(Function {
            signature: Callable::list(
                ParamList::new(params),
                Type::ClassType(self.stdlib.iterable(self.unions(element_types))),
            ),
            metadata: FunctionKind::Def(Box::new(FuncId {
                module: self.module_info().name(),
                cls: Some(cls.name().clone()),
                func: dunder::ITER,
            })),
        }));
        ClassSynthesizedField::new(ty)
    }

    fn get_named_tuple_match_args(&self, elements: &[Name]) -> ClassSynthesizedField {
        let ty = Type::Tuple(Tuple::Concrete(
            elements
                .iter()
                .map(|e| Type::Literal(Lit::String(e.as_str().into())))
                .collect(),
        ));
        ClassSynthesizedField::new(ty)
    }

    pub fn get_named_tuple_synthesized_fields(
        &self,
        cls: &Class,
    ) -> Option<ClassSynthesizedFields> {
        let metadata = self.get_metadata_for_class(cls);
        let named_tuple = metadata.named_tuple_metadata()?;
        Some(ClassSynthesizedFields::new(smallmap! {
            dunder::NEW => self.get_named_tuple_new(cls, &named_tuple.elements),
            dunder::INIT => self.get_named_tuple_init(cls, &named_tuple.elements),
            dunder::MATCH_ARGS => self.get_named_tuple_match_args(&named_tuple.elements),
            dunder::ITER => self.get_named_tuple_iter(cls, &named_tuple.elements)
        }))
    }
}
