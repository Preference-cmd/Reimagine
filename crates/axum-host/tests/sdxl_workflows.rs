#![allow(dead_code)]

use serde_json::Value;

#[derive(Debug, Clone)]
pub struct SdxlWorkflowOptions<'a> {
    pub workflow_id: &'a str,
    pub model_id: &'a str,
    pub name: &'a str,
    pub description: &'a str,
    pub filename_prefix: &'a str,
    pub denoise: f64,
}

pub fn text_to_image(options: SdxlWorkflowOptions<'_>) -> Value {
    serde_json::json!({
        "schema_version": "reimagine.workflow.v1",
        "id": options.workflow_id,
        "version": 1,
        "metadata": {
            "name": options.name,
            "description": options.description,
            "created_by": "human"
        },
        "interface": { "inputs": [], "outputs": [] },
        "nodes": [
            checkpoint_node(options.model_id),
            positive_prompt_node(),
            negative_prompt_node(),
            positive_encode_node(),
            negative_encode_node(),
            empty_latent_node(),
            sampler_node(options.denoise),
            vae_decode_node(),
            save_image_node(options.filename_prefix),
        ],
        "edges": [
            { "id": "edge_checkpoint_model_sampler", "from": { "node": "node_checkpoint", "slot": "model" }, "to": { "node": "node_sampler", "slot": "model" } },
            { "id": "edge_checkpoint_clip_positive", "from": { "node": "node_checkpoint", "slot": "clip" }, "to": { "node": "node_positive_encode", "slot": "clip" } },
            { "id": "edge_checkpoint_clip_negative", "from": { "node": "node_checkpoint", "slot": "clip" }, "to": { "node": "node_negative_encode", "slot": "clip" } },
            { "id": "edge_positive_prompt_encode", "from": { "node": "node_positive_prompt", "slot": "value" }, "to": { "node": "node_positive_encode", "slot": "text" } },
            { "id": "edge_negative_prompt_encode", "from": { "node": "node_negative_prompt", "slot": "value" }, "to": { "node": "node_negative_encode", "slot": "text" } },
            { "id": "edge_positive_conditioning_sampler", "from": { "node": "node_positive_encode", "slot": "conditioning" }, "to": { "node": "node_sampler", "slot": "positive" } },
            { "id": "edge_negative_conditioning_sampler", "from": { "node": "node_negative_encode", "slot": "conditioning" }, "to": { "node": "node_sampler", "slot": "negative" } },
            { "id": "edge_latent_sampler", "from": { "node": "node_latent", "slot": "latent" }, "to": { "node": "node_sampler", "slot": "latent" } },
            { "id": "edge_sampler_vae_decode", "from": { "node": "node_sampler", "slot": "latent" }, "to": { "node": "node_vae_decode", "slot": "latent" } },
            { "id": "edge_checkpoint_vae_decode", "from": { "node": "node_checkpoint", "slot": "vae" }, "to": { "node": "node_vae_decode", "slot": "vae" } },
            { "id": "edge_vae_decode_save", "from": { "node": "node_vae_decode", "slot": "image" }, "to": { "node": "node_save_image", "slot": "image" } },
        ],
        "layout": { "nodes": {} }
    })
}

pub fn image_to_image(options: SdxlWorkflowOptions<'_>, input_image: &str) -> Value {
    serde_json::json!({
        "schema_version": "reimagine.workflow.v1",
        "id": options.workflow_id,
        "version": 1,
        "metadata": {
            "name": options.name,
            "description": options.description,
            "created_by": "human"
        },
        "interface": { "inputs": [], "outputs": [] },
        "nodes": [
            checkpoint_node(options.model_id),
            load_image_node(input_image),
            vae_encode_node(),
            positive_prompt_node(),
            negative_prompt_node(),
            positive_encode_node(),
            negative_encode_node(),
            sampler_node(options.denoise),
            vae_decode_node(),
            save_image_node(options.filename_prefix),
        ],
        "edges": [
            { "id": "edge_checkpoint_model_sampler", "from": { "node": "node_checkpoint", "slot": "model" }, "to": { "node": "node_sampler", "slot": "model" } },
            { "id": "edge_checkpoint_clip_positive", "from": { "node": "node_checkpoint", "slot": "clip" }, "to": { "node": "node_positive_encode", "slot": "clip" } },
            { "id": "edge_checkpoint_clip_negative", "from": { "node": "node_checkpoint", "slot": "clip" }, "to": { "node": "node_negative_encode", "slot": "clip" } },
            { "id": "edge_positive_prompt_encode", "from": { "node": "node_positive_prompt", "slot": "value" }, "to": { "node": "node_positive_encode", "slot": "text" } },
            { "id": "edge_negative_prompt_encode", "from": { "node": "node_negative_prompt", "slot": "value" }, "to": { "node": "node_negative_encode", "slot": "text" } },
            { "id": "edge_positive_conditioning_sampler", "from": { "node": "node_positive_encode", "slot": "conditioning" }, "to": { "node": "node_sampler", "slot": "positive" } },
            { "id": "edge_negative_conditioning_sampler", "from": { "node": "node_negative_encode", "slot": "conditioning" }, "to": { "node": "node_sampler", "slot": "negative" } },
            { "id": "edge_load_image_vae_encode", "from": { "node": "node_load_image", "slot": "image" }, "to": { "node": "node_vae_encode", "slot": "image" } },
            { "id": "edge_checkpoint_vae_encode", "from": { "node": "node_checkpoint", "slot": "vae" }, "to": { "node": "node_vae_encode", "slot": "vae" } },
            { "id": "edge_vae_encode_sampler", "from": { "node": "node_vae_encode", "slot": "latent" }, "to": { "node": "node_sampler", "slot": "latent" } },
            { "id": "edge_sampler_vae_decode", "from": { "node": "node_sampler", "slot": "latent" }, "to": { "node": "node_vae_decode", "slot": "latent" } },
            { "id": "edge_checkpoint_vae_decode", "from": { "node": "node_checkpoint", "slot": "vae" }, "to": { "node": "node_vae_decode", "slot": "vae" } },
            { "id": "edge_vae_decode_save", "from": { "node": "node_vae_decode", "slot": "image" }, "to": { "node": "node_save_image", "slot": "image" } },
        ],
        "layout": { "nodes": {} }
    })
}

fn checkpoint_node(model_id: &str) -> Value {
    serde_json::json!({
        "id": "node_checkpoint",
        "type_id": "builtin.checkpoint_loader",
        "label": "Checkpoint",
        "params": {
            "checkpoint": {
                "type": "model_ref",
                "value": {
                    "id": model_id,
                    "model_series": "stable_diffusion",
                    "variant": "sdxl",
                    "role": "CheckpointBundle"
                }
            }
        }
    })
}

fn load_image_node(input_image: &str) -> Value {
    serde_json::json!({
        "id": "node_load_image",
        "type_id": "builtin.load_image",
        "label": "Load Image",
        "params": {
            "image": { "type": "path", "value": input_image }
        }
    })
}

fn vae_encode_node() -> Value {
    serde_json::json!({
        "id": "node_vae_encode",
        "type_id": "builtin.vae_encode",
        "label": "VAE Encode",
        "params": {}
    })
}

fn positive_prompt_node() -> Value {
    serde_json::json!({
        "id": "node_positive_prompt",
        "type_id": "builtin.string",
        "label": "Positive Prompt",
        "params": {
            "value": { "type": "string", "value": "cinematic lake at sunrise, detailed, soft light" }
        }
    })
}

fn negative_prompt_node() -> Value {
    serde_json::json!({
        "id": "node_negative_prompt",
        "type_id": "builtin.string",
        "label": "Negative Prompt",
        "params": {
            "value": { "type": "string", "value": "low quality, blurry, distorted" }
        }
    })
}

fn positive_encode_node() -> Value {
    serde_json::json!({
        "id": "node_positive_encode",
        "type_id": "builtin.clip_text_encode",
        "label": "Positive CLIP Encode",
        "params": {}
    })
}

fn negative_encode_node() -> Value {
    serde_json::json!({
        "id": "node_negative_encode",
        "type_id": "builtin.clip_text_encode",
        "label": "Negative CLIP Encode",
        "params": {}
    })
}

fn empty_latent_node() -> Value {
    serde_json::json!({
        "id": "node_latent",
        "type_id": "builtin.empty_latent_image",
        "label": "Empty Latent",
        "params": {
            "width": { "type": "integer", "value": 512 },
            "height": { "type": "integer", "value": 512 },
            "batch_size": { "type": "integer", "value": 1 }
        }
    })
}

fn sampler_node(denoise: f64) -> Value {
    serde_json::json!({
        "id": "node_sampler",
        "type_id": "builtin.ksampler",
        "label": "KSampler",
        "params": {
            "seed": { "type": "seed", "value": 123456789 },
            "steps": { "type": "integer", "value": 5 },
            "cfg": { "type": "float", "value": 7.0 },
            "sampler": { "type": "select", "value": "euler" },
            "scheduler": { "type": "select", "value": "normal" },
            "denoise": { "type": "float", "value": denoise }
        }
    })
}

fn vae_decode_node() -> Value {
    serde_json::json!({
        "id": "node_vae_decode",
        "type_id": "builtin.vae_decode",
        "label": "VAE Decode",
        "params": {}
    })
}

fn save_image_node(filename_prefix: &str) -> Value {
    serde_json::json!({
        "id": "node_save_image",
        "type_id": "builtin.save_image",
        "label": "Save Image",
        "params": {
            "filename_prefix": { "type": "string", "value": filename_prefix }
        }
    })
}
