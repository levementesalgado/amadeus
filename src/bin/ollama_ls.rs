//! Lista modelos do Ollama com tamanho, quantização e caminho do .gguf.
//!
//! Uso:  cargo run --bin ollama_ls

use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::fs;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Manifest {
    #[serde(default)]
    layers: Vec<ManifestLayer>,
    config: Option<ManifestConfigRef>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ManifestLayer {
    media_type: String,
    digest: String,
    size: u64,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ManifestConfigRef {
    digest: String,
    size: u64,
}

#[derive(Deserialize)]
struct ModelConfig {
    model_format: Option<String>,
    model_family: Option<String>,
    model_type: Option<String>,
    file_type: Option<String>,
}

#[derive(Debug)]
struct ModelEntry {
    name: String,
    tag: String,
    family: String,
    quant: String,
    model_type: String,
    gguf_size: u64,
    gguf_blob: String,
    display_size: String,
}

fn ollama_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".into());
    PathBuf::from(home).join(".ollama").join("models")
}

fn human_size(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    for unit in UNITS {
        if size < 1024.0 {
            return format!("{:.1} {}", size, unit);
        }
        size /= 1024.0;
    }
    format!("{:.2} TB", size)
}

fn read_blob_json(blobs_dir: &Path, digest: &str) -> Result<serde_json::Value, String> {
    let path = blobs_dir.join(digest.replace(':', "-"));
    let content = fs::read_to_string(&path)
        .map_err(|e| format!("blob {}: {}", digest, e))?;
    serde_json::from_str(&content)
        .map_err(|e| format!("json {}: {}", digest, e))
}

fn find_models() -> Vec<ModelEntry> {
    let base = ollama_dir();
    let manifests_dir = base.join("manifests").join("registry.ollama.ai").join("library");
    let blobs_dir = base.join("blobs");

    let mut entries = Vec::new();

    let read_dir = match fs::read_dir(&manifests_dir) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Aviso: não foi possível ler {}: {}", manifests_dir.display(), e);
            return entries;
        }
    };

    for model_entry in read_dir.flatten() {
        let model_name = model_entry.file_name();
        let model_name = model_name.to_string_lossy().to_string();
        let model_path = model_entry.path();

        if !model_path.is_dir() {
            continue;
        }

        let Ok(tags_dir) = fs::read_dir(&model_path) else { continue; };

        for tag_entry in tags_dir.flatten() {
            let tag_name = tag_entry.file_name();
            let tag_name = tag_name.to_string_lossy().to_string();
            let manifest_path = tag_entry.path();

            let manifest_str = match fs::read_to_string(&manifest_path) {
                Ok(s) => s,
                Err(_) => continue,
            };

            let manifest: Manifest = match serde_json::from_str(&manifest_str) {
                Ok(m) => m,
                Err(_) => continue,
            };

            let model_layer = manifest.layers.iter().find(|l| l.media_type == "application/vnd.ollama.image.model");
            let gguf_size = model_layer.map(|l| l.size).unwrap_or(0);
            let gguf_digest = model_layer.map(|l| l.digest.clone()).unwrap_or_default();

            let config_digest = manifest.config.as_ref().map(|c| &c.digest);
            let config = config_digest
                .and_then(|d| read_blob_json(&blobs_dir, d).ok())
                .and_then(|v| serde_json::from_value::<ModelConfig>(v).ok());

            let family = config.as_ref().and_then(|c| c.model_family.clone()).unwrap_or_else(|| "?".into());
            let quant = config.as_ref().and_then(|c| c.file_type.clone()).unwrap_or_else(|| "?".into());
            let model_type = config.as_ref().and_then(|c| c.model_type.clone()).unwrap_or_else(|| "?".into());

            entries.push(ModelEntry {
                name: model_name.clone(),
                tag: tag_name,
                family,
                quant,
                model_type,
                gguf_size,
                gguf_blob: gguf_digest,
                display_size: human_size(gguf_size),
            });
        }
    }

    entries.sort_by(|a, b| a.name.cmp(&b.name).then(a.tag.cmp(&b.tag)));
    entries
}

fn main() {
    let models = find_models();

    if models.is_empty() {
        println!("Nenhum modelo Ollama encontrado.");
        println!("Caminho procurado: {}", ollama_dir().join("manifests/registry.ollama.ai/library").display());
        println!("Baixe um modelo com: ollama pull tinyllama");
        return;
    }

    let blobs_dir = ollama_dir().join("blobs");

    println!("╔═╗╦  ╦╔═╗╔╗╔╔═╗╔═╗  ╔╦╗╔═╗╦ ╦╔╗╔╔═╗");
    println!("╠═╝║  ║║╣ ║║║╚═╗║╣    ║ ║ ║║ ║║║║║╣ ");
    println!("╩  ╩═╝╩╚═╝╝╚╝╚═╝╚═╝   ╩ ╚═╝╚═╝╝╚╝╚═╝");
    println!();

    println!("{:<20} {:<8} {:<8} {:<6} {:<6} {:<10} {:<20}",
        "MODELO", "TAG", "FAMÍLIA", "TIPO", "QUANT", "TAMANHO", "BLOB");
    println!("{}", "-".repeat(90));

    for m in &models {
        let blob_path = blobs_dir.join(m.gguf_blob.replace(':', "-"));
        let blob_path_str = blob_path.to_string_lossy();

        println!("{:<20} {:<8} {:<8} {:<6} {:<6} {:<10} {}",
            m.name,
            m.tag,
            m.family,
            m.model_type,
            m.quant,
            m.display_size,
            blob_path_str);
    }

    println!();
    println!("Total: {} modelos", models.len());
    println!();
    println!("Para usar com o Amadeus:");
    for m in &models {
        let blob_path = blobs_dir.join(m.gguf_blob.replace(':', "-"));
        println!("  cargo run --release -- {}", blob_path.display());
        break; // só o primeiro
    }
    if models.len() > 1 {
        println!("  (... e mais {} modelos)", models.len() - 1);
    }
}
