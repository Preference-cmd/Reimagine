use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream};
use syn::{
    Expr, ExprArray, ExprLit, FnArg, GenericArgument, Ident, ItemFn, Lit, LitStr, PathArguments,
    ReturnType, Token, Type, parse_quote,
};

pub fn expand(attr: TokenStream, item: TokenStream) -> TokenStream {
    match expand_inner(attr, item.clone()) {
        Ok(tokens) => tokens,
        Err(err) => {
            let err = err.to_compile_error();
            quote! {
                #item
                #err
            }
        }
    }
}

fn expand_inner(attr: TokenStream, item: TokenStream) -> syn::Result<TokenStream> {
    let args = syn::parse2::<AgentToolArgs>(attr)?;
    args.validate()?;

    let input_fn = syn::parse2::<ItemFn>(item)?;
    validate_async_fn(&input_fn)?;

    let fn_name = &input_fn.sig.ident;
    let vis = &input_fn.vis;
    let wrapper_ident = format_ident!("{}AgentTool", to_pascal_case(&fn_name.to_string()));
    let constructor_ident = format_ident!("{}_agent_tool", fn_name);

    let input_ty = extract_input_type(&input_fn)?;
    let output_ty = extract_output_type(&input_fn)?;

    let name = args.name.as_ref().unwrap();
    let description = args.description.as_ref().unwrap();
    let permission = args.permission.as_ref().unwrap();
    let mode_tokens = args
        .modes
        .iter()
        .map(|mode| match mode.as_str() {
            "agent" => quote!(::reimagine_agent::AgentMode::Agent),
            "build" => quote!(::reimagine_agent::AgentMode::Build),
            _ => unreachable!("validated mode"),
        })
        .collect::<Vec<_>>();
    let risk_token = match args.risk.as_deref().unwrap_or("read") {
        "read" => quote!(::reimagine_agent::ToolRiskLevel::Read),
        "editor" => quote!(::reimagine_agent::ToolRiskLevel::Editor),
        "external" => quote!(::reimagine_agent::ToolRiskLevel::External),
        _ => unreachable!("validated risk"),
    };

    Ok(quote! {
        #input_fn

        #[derive(Debug, Clone, Copy, Default)]
        #vis struct #wrapper_ident;

        #vis fn #constructor_ident() -> #wrapper_ident {
            #wrapper_ident
        }

        #[::async_trait::async_trait]
        impl ::reimagine_agent::AgentTool for #wrapper_ident {
            fn spec(&self) -> ::reimagine_agent::ToolSpec {
                let input_schema = ::serde_json::to_value(
                    ::schemars::schema_for!(#input_ty)
                )
                .expect("agent tool input schema must serialize to JSON");
                let output_schema = ::serde_json::to_value(
                    ::schemars::schema_for!(#output_ty)
                )
                .expect("agent tool output schema must serialize to JSON");

                ::reimagine_agent::ToolSpec::new(
                    ::reimagine_agent::ToolName::new(#name),
                    #description,
                    [#(#mode_tokens),*],
                    ::reimagine_agent::ToolPermission::new(#permission),
                    #risk_token,
                )
                .with_input_schema(input_schema)
                .with_output_schema(output_schema)
            }

            async fn invoke(
                &self,
                ctx: &::reimagine_agent::ToolContext,
                input: ::reimagine_agent::ToolInput,
            ) -> ::reimagine_agent::ToolResult {
                let typed_input: #input_ty = ::serde_json::from_value(input).map_err(|err| {
                    ::reimagine_agent::ToolError::new(
                        ::reimagine_agent::ToolErrorCode::InvalidInput,
                        format!("failed to deserialize tool input: {err}"),
                    )
                    .with_tool(::reimagine_agent::ToolName::new(#name))
                })?;
                let typed_output = #fn_name(ctx.clone(), typed_input).await?;
                ::serde_json::to_value(typed_output).map_err(|err| {
                    ::reimagine_agent::ToolError::new(
                        ::reimagine_agent::ToolErrorCode::ExecutionFailed,
                        format!("failed to serialize tool output: {err}"),
                    )
                    .with_tool(::reimagine_agent::ToolName::new(#name))
                })
            }
        }
    })
}

#[derive(Default)]
struct AgentToolArgs {
    name: Option<String>,
    description: Option<String>,
    modes: Vec<String>,
    permission: Option<String>,
    risk: Option<String>,
}

impl Parse for AgentToolArgs {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let mut args = Self::default();
        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![=]>()?;
            match key.to_string().as_str() {
                "name" => args.name = Some(parse_string_value(input)?),
                "description" => args.description = Some(parse_string_value(input)?),
                "modes" => args.modes = parse_modes(input)?,
                "permission" => args.permission = Some(parse_string_value(input)?),
                "risk" => args.risk = Some(parse_string_value(input)?),
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!("unsupported agent_tool metadata `{other}`"),
                    ));
                }
            }
            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }
        Ok(args)
    }
}

impl AgentToolArgs {
    fn validate(&self) -> syn::Result<()> {
        require_present(self.name.as_deref(), "name")?;
        require_present(self.description.as_deref(), "description")?;
        require_present(self.permission.as_deref(), "permission")?;
        if self.modes.is_empty() {
            return Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                "`modes` must include at least one mode",
            ));
        }
        for mode in &self.modes {
            if mode != "agent" && mode != "build" {
                return Err(syn::Error::new(
                    proc_macro2::Span::call_site(),
                    "`modes` values must be `agent` or `build`",
                ));
            }
        }
        if let Some(risk) = &self.risk
            && risk != "read"
            && risk != "editor"
            && risk != "external"
        {
            return Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                "`risk` must be `read`, `editor`, or `external`",
            ));
        }
        Ok(())
    }
}

fn require_present(value: Option<&str>, key: &str) -> syn::Result<()> {
    match value {
        Some(value) if !value.trim().is_empty() => Ok(()),
        _ => Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            format!("missing required agent_tool metadata `{key}`"),
        )),
    }
}

fn parse_string_value(input: ParseStream<'_>) -> syn::Result<String> {
    let lit: LitStr = input.parse()?;
    Ok(lit.value())
}

fn parse_modes(input: ParseStream<'_>) -> syn::Result<Vec<String>> {
    let expr: Expr = input.parse()?;
    let Expr::Array(ExprArray { elems, .. }) = expr else {
        return Err(syn::Error::new_spanned(
            expr,
            "`modes` must be a string array",
        ));
    };
    elems
        .into_iter()
        .map(|expr| match expr {
            Expr::Lit(ExprLit {
                lit: Lit::Str(value),
                ..
            }) => Ok(value.value()),
            other => Err(syn::Error::new_spanned(
                other,
                "`modes` must contain string literals",
            )),
        })
        .collect()
}

fn validate_async_fn(input_fn: &ItemFn) -> syn::Result<()> {
    if input_fn.sig.asyncness.is_none() {
        return Err(syn::Error::new_spanned(
            input_fn.sig.fn_token,
            "`#[agent_tool]` requires an async function",
        ));
    }
    if input_fn.sig.inputs.len() != 2 {
        return Err(syn::Error::new_spanned(
            &input_fn.sig.inputs,
            "`#[agent_tool]` functions must accept `(ToolContext, Input)`",
        ));
    }

    let first = input_fn.sig.inputs.first().unwrap();
    let FnArg::Typed(first) = first else {
        return Err(syn::Error::new_spanned(
            first,
            "`#[agent_tool]` does not support method receivers",
        ));
    };
    if !type_ends_with(&first.ty, "ToolContext") {
        return Err(syn::Error::new_spanned(
            &first.ty,
            "first argument must be `ToolContext`",
        ));
    }
    Ok(())
}

fn extract_input_type(input_fn: &ItemFn) -> syn::Result<Type> {
    let Some(FnArg::Typed(second)) = input_fn.sig.inputs.iter().nth(1) else {
        return Err(syn::Error::new_spanned(
            &input_fn.sig.inputs,
            "`#[agent_tool]` functions must accept a typed input argument",
        ));
    };
    Ok((*second.ty).clone())
}

fn extract_output_type(input_fn: &ItemFn) -> syn::Result<Type> {
    let ReturnType::Type(_, ty) = &input_fn.sig.output else {
        return Err(syn::Error::new_spanned(
            &input_fn.sig.output,
            "`#[agent_tool]` functions must return `ToolResult<T>`",
        ));
    };
    extract_success_type(ty)
}

fn extract_success_type(ty: &Type) -> syn::Result<Type> {
    let Type::Path(type_path) = ty else {
        return Err(syn::Error::new_spanned(
            ty,
            "`#[agent_tool]` return type must be `ToolResult<T>` or `Result<T, E>`",
        ));
    };
    let Some(segment) = type_path.path.segments.last() else {
        return Err(syn::Error::new_spanned(ty, "unsupported return type"));
    };
    match segment.ident.to_string().as_str() {
        "ToolResult" | "Result" => first_generic_type(ty, &segment.arguments),
        _ => Err(syn::Error::new_spanned(
            ty,
            "`#[agent_tool]` return type must be `ToolResult<T>` or `Result<T, E>`",
        )),
    }
}

fn first_generic_type(ty: &Type, args: &PathArguments) -> syn::Result<Type> {
    let PathArguments::AngleBracketed(args) = args else {
        return Err(syn::Error::new_spanned(
            ty,
            "return type must specify a success output type",
        ));
    };
    let Some(GenericArgument::Type(output_ty)) = args.args.first() else {
        return Err(syn::Error::new_spanned(
            ty,
            "return type must specify a success output type",
        ));
    };
    Ok(output_ty.clone())
}

fn type_ends_with(ty: &Type, expected: &str) -> bool {
    match ty {
        Type::Path(type_path) => type_path
            .path
            .segments
            .last()
            .is_some_and(|segment| segment.ident == expected),
        Type::Reference(reference) => type_ends_with(&reference.elem, expected),
        _ => false,
    }
}

fn to_pascal_case(value: &str) -> String {
    let mut out = String::new();
    let mut upper_next = true;
    for ch in value.chars() {
        if ch == '_' {
            upper_next = true;
            continue;
        }
        if upper_next {
            out.extend(ch.to_uppercase());
            upper_next = false;
        } else {
            out.push(ch);
        }
    }
    if out.is_empty() {
        "Generated".to_owned()
    } else {
        out
    }
}

#[allow(dead_code)]
fn _assert_paths_compile() {
    let _: Type = parse_quote!(::reimagine_agent::ToolContext);
}
