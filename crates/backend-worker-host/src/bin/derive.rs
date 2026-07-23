use std::process::exit;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 { eprintln!("Usage: derive <seed-hex>"); exit(1); }
    let seed_bytes = hex::decode(&args[1]).unwrap_or_else(|e| { eprintln!("Invalid hex: {}", e); exit(1); });
    if seed_bytes.len() != 32 { eprintln!("Expected 32 bytes, got {}", seed_bytes.len()); exit(1); }
    let seed_arr: [u8; 32] = seed_bytes.try_into().unwrap();
    let key = ed25519_dalek::SigningKey::from_bytes(&seed_arr);
    println!("Seed:   {}", args[1]);
    println!("Pubkey: {}", hex::encode(key.verifying_key().to_bytes()));
}
