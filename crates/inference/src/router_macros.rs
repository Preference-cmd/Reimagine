/// Macro to generate typed capability dispatch methods for `InferenceRouter`.
///
/// These methods all follow the same pattern:
/// 1. Resolve the backend from the request
/// 2. Call the `*_with_invocation` method on the resolved backend
///
/// # Usage
///
/// ```rust,ignore
/// impl_dispatch_methods! {
///     load_bundle_with_invocation: load_bundle_with_invocation(LoadBundleRequest) -> LoadBundleResponse,
///     text_encode_with_invocation: text_encode_with_invocation(TextEncodeRequest) -> TextEncodeResponse,
///     // ...
/// }
/// ```
macro_rules! impl_dispatch_methods {
    ($(
        $method:ident : $backend_method:ident($request_ty:ty) -> $response_ty:ty
    ),* $(,)?) => {
        $(
            pub async fn $method(
                &self,
                invocation: &InferenceInvocation,
                request: $request_ty,
            ) -> Result<$response_ty, InferenceError> {
                let backend = self.resolve_for_request(&request)?;
                backend
                    .backend
                    .$backend_method(invocation, request)
                    .await
            }
        )*
    };
}
