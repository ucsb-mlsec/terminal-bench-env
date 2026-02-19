use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, Fields};

#[proc_macro_derive(Builder)]
pub fn builder_derive(input: TokenStream2) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    
    let name = input.ident;
    let builder_name = syn::Ident::new(&format!("{}Builder", name), name.span());
    
    let fields = match input.data {
        Data::Struct(data_struct) => {
            match data_struct.fields {
                Fields::Named(fields_named) => fields_named.named,
                _ => panic!("Builder only supports structs with named fields"),
            }
        }
        _ => panic!("Builder only supports structs"),
    };
    
    let builder_fields = fields.iter().map(|f| {
        let field_name = &f.ident;
        let field_type = &f.ty;
        quote! {
            #field_name: std::option::Option<#field_type>
        }
    });
    
    let builder_init = fields.iter().map(|f| {
        let field_name = &f.ident;
        quote! {
            #field_name: std::option::None
        }
    });
    
    let builder_methods = fields.iter().map(|f| {
        let field_name = &f.ident;
        let field_type = &f.ty;
        quote! {
            pub fn #field_name(&mut self, #field_name: #field_type) -> &mut Self {
                self.#field_name = std::option::Some(#field_name);
                self
            }
        }
    });
    
    let builder_build_fields = fields.iter().map(|f| {
        let field_name = &f.ident;
        quote! {
            #field_name: self.#field_name.clone().ok_or(concat!(stringify!(#field_name), " is not set"))?
        }
    });
    
    let expanded = quote! {
        impl #name {
            pub fn builder() -> #builder_name {
                #builder_name {
                    #(#builder_init),*
                }
            }
        }
        
        pub struct #builder_name {
            #(#builder_fields),*
        }
        
        impl #builder_name {
            #(#builder_methods)*
            
            pub fn build(&self) -> std::result::Result<#name, std::boxed::Box<dyn std::error::Error>> {
                std::result::Ok(#name {
                    #(#builder_build_fields),*
                })
            }
        }
    };
    
    TokenStream::from(expanded)
}

pub fn helper_function(input: TokenStream) -> String {
    format!("Processing: {:?}", input)
}

#[proc_macro]
pub fn custom_macro(input: TokenStream) -> TokenStream {
    let output = quote! {
        fn generated_function() {
            println!("Hello from generated code!");
        }
    };
    
    output.into()
}

fn internal_helper(tokens: TokenStream2) -> proc_macro2::TokenStream {
    quote! {
        struct InternalStruct;
    }
}

#[derive(Debug)]
struct BuilderConfig {
    enabled: bool,
    default_values: Vec<String>,
}

impl BuilderConfig {
    fn new() -> Self {
        BuilderConfig {
            enabled: true,
            default_values: Vec::new(),
        }
    }
    
    fn apply(&self, tokens: &TokenStream) -> TokenStream {
        tokens.clone()
    }
}