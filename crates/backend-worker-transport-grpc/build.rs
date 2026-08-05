fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var("PROTOC").is_err() {
        if let Ok(path) = protoc_bin_vendored::protoc_bin_path() {
            // SAFETY: build scripts are single-threaded; no other thread
            // observes this env var.
            unsafe { std::env::set_var("PROTOC", path) };
        }
    }
    tonic_prost_build::compile_protos("proto/worker.proto")?;
    Ok(())
}
