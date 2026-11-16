/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use quote::ToTokens;
use quote::format_ident;
use quote::quote;
use quote::quote_spanned;
use syn::Data;
use syn::DeriveInput;
use syn::Fields;
use syn::GenericParam;
use syn::Generics;
use syn::parse_macro_input;
use syn::spanned::Spanned;

pub(crate) fn derive_type_ord(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match derive_type_ord_impl(&input) {
        Ok(x) => x.into_token_stream().into(),
        Err(e) => e.to_compile_error().into(),
    }
}

fn generics(
    generics: &Generics,
) -> syn::Result<(proc_macro2::TokenStream, proc_macro2::TokenStream)> {
    let mut lifetimes = Vec::new();
    let mut ts = Vec::new();
    for param in &generics.params {
        match param {
            GenericParam::Lifetime(l) => {
                lifetimes.push(&l.lifetime);
            }
            GenericParam::Type(t) if t.bounds.is_empty() => {
                ts.push(&t.ident);
            }
            _ => {
                return Err(syn::Error::new_spanned(
                    param,
                    "Unsupported generic parameter",
                ));
            }
        }
    }
    let before = quote_spanned! { generics.span() => < #(#lifetimes,)* #(#ts: crate::type_ord::TypeOrd),* > };
    let after = quote_spanned! { generics.span() => < #(#lifetimes,)* #(#ts),* > };
    Ok((before, after))
}

/// Compare the pairs in order, returning the first ordering that is not `Equal`, like the derived
/// `Ord` does.
fn lexicographic(
    pairs: impl Iterator<Item = (proc_macro2::TokenStream, proc_macro2::TokenStream)>,
) -> proc_macro2::TokenStream {
    let type_ord = quote! { crate::type_ord::TypeOrd };
    let steps = pairs.map(|(lhs, rhs)| {
        quote! {
            match #type_ord::type_cmp(#lhs, #rhs, ctx) {
                ::std::cmp::Ordering::Equal => {}
                ord => return ord,
            }
        }
    });
    quote! { #(#steps)* ::std::cmp::Ordering::Equal }
}

fn derive_type_ord_impl(input: &DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let name = &input.ident;
    let (generics_before, generics_after) = generics(&input.generics)?;
    let type_ord = quote! { crate::type_ord::TypeOrd };
    let type_ord_ctx = quote! { crate::type_ord::TypeOrdCtx };
    let body = match &input.data {
        Data::Struct(data_struct) => match &data_struct.fields {
            Fields::Named(fields_named) => lexicographic(fields_named.named.iter().map(|field| {
                let field_name = &field.ident;
                (quote! { &self.#field_name }, quote! { &other.#field_name })
            })),
            Fields::Unnamed(fields_unnamed) => {
                lexicographic(fields_unnamed.unnamed.iter().enumerate().map(|(i, _)| {
                    let index = syn::Index::from(i);
                    (quote! { &self.#index }, quote! { &other.#index })
                }))
            }
            Fields::Unit => quote! { ::std::cmp::Ordering::Equal },
        },
        Data::Enum(data_enum) => {
            let indices = data_enum.variants.iter().enumerate().map(|(i, variant)| {
                let variant_name = &variant.ident;
                let pattern = match &variant.fields {
                    Fields::Named(_) => quote! { #name::#variant_name { .. } },
                    Fields::Unnamed(_) => quote! { #name::#variant_name(..) },
                    Fields::Unit => quote! { #name::#variant_name },
                };
                quote! { #pattern => #i }
            });
            let variants = data_enum.variants.iter().map(|variant| {
                let variant_name = &variant.ident;
                match &variant.fields {
                    Fields::Named(fields_named) => {
                        let field_names: Vec<_> = fields_named.named.iter().map(|f| &f.ident).collect();
                        let lhs: Vec<_> = (0..fields_named.named.len()).map(|i| format_ident!("lhs{i}")).collect();
                        let rhs: Vec<_> = (0..fields_named.named.len()).map(|i| format_ident!("rhs{i}")).collect();
                        let bind_lhs = field_names.iter().zip(&lhs).map(|(name, var)| quote! {#name: #var});
                        let bind_rhs = field_names.iter().zip(&rhs).map(|(name, var)| quote! {#name: #var});
                        let body = lexicographic(lhs.iter().zip(&rhs).map(|(l, r)| (quote! { #l }, quote! { #r })));
                        quote_spanned! { variant.span() =>
                            (#name::#variant_name { #(#bind_lhs),*  }, #name::#variant_name {  #(#bind_rhs),*  }) => {
                                #body
                            }
                        }
                    }
                    Fields::Unnamed(fields_unnamed) => {
                        let lhs: Vec<_> = (0..fields_unnamed.unnamed.len()).map(|i| format_ident!("lhs{i}")).collect();
                        let rhs: Vec<_> = (0..fields_unnamed.unnamed.len()).map(|i| format_ident!("rhs{i}")).collect();
                        let body = lexicographic(lhs.iter().zip(&rhs).map(|(l, r)| (quote! { #l }, quote! { #r })));
                        quote_spanned! { variant.span() =>
                            (#name::#variant_name(#(#lhs),*), #name::#variant_name(#(#rhs),*)) => {
                                #body
                            }
                        }
                    }
                    Fields::Unit => {
                        quote_spanned! { variant.span() =>
                            (#name::#variant_name, #name::#variant_name) => ::std::cmp::Ordering::Equal
                        }
                    }
                }
            });
            quote! {
                match (self, other) {
                    #(#variants),*,
                    _ => {
                        fn variant_index #generics_before (x: &#name #generics_after) -> usize {
                            match x {
                                #(#indices),*
                            }
                        }
                        variant_index(self).cmp(&variant_index(other))
                    }
                }
            }
        }
        _ => {
            return Err(syn::Error::new_spanned(
                input,
                "Can't deal with this type of data",
            ));
        }
    };
    Ok(quote! {
        impl #generics_before #type_ord for #name #generics_after {
            fn type_cmp(&self, other: &Self, ctx: &#type_ord_ctx) -> ::std::cmp::Ordering {
                #body
            }
        }
    })
}
