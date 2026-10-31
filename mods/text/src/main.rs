//! The `text` mod binary: serve [`text_mod::TextMod`] over stdio.

fn main() -> std::io::Result<()> {
    mod_sdk::sdk::serve_main(text_mod::TextMod::new())
}
