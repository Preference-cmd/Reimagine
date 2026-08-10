#[test]
fn agent_tool_ui_tests() {
    let t = trybuild::TestCases::new();
    t.pass("tests/pass/valid_async_tool.rs");
    t.compile_fail("tests/ui/missing_metadata.rs");
    t.compile_fail("tests/ui/invalid_signature.rs");
}
