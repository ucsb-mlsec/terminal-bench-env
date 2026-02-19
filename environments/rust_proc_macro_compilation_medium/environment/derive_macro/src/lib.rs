use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

#[proc_macro_derive(MyTrait)]
pub fn my_trait_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    
    let name = input.ident;
    
    let expanded = quote! {
        impl MyTrait for #name {
            fn describe() -> String {
                format!("This is a {} struct", stringify!(#name))
            }
        }
    };
    
    TokenStream::from(expanded)
}