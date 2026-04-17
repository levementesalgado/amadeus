use std::fs;
use std::path::{Path, PathBuf};

pub struct Underworld {
    root: PathBuf,
    visit_log: Vec<String>,
}

impl Underworld {
    pub fn new(root: &str) -> Self {
        Self {
            root: PathBuf::from(root),
            visit_log: Vec::new(),
        }
    }

    /// Lista todos os arquivos .md no underworld (recursivo)
    pub fn list_files(&self) -> Vec<PathBuf> {
        let mut files = Vec::new();
        if let Ok(entries) = fs::read_dir(&self.root) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    if let Ok(sub) = fs::read_dir(&path) {
                        for sub_entry in sub.flatten() {
                            if sub_entry.path().extension().map_or(false, |e| e == "md") {
                                files.push(sub_entry.path());
                            }
                        }
                    }
                } else if path.extension().map_or(false, |e| e == "md") {
                    files.push(path);
                }
            }
        }
        files.sort();
        files
    }

    /// Carrega o conteúdo de um arquivo
    pub fn read_file(&self, path: &Path) -> Option<String> {
        fs::read_to_string(path).ok()
    }

    /// Busca arquivos que contenham palavras-chave relevantes
    pub fn search(&self, keywords: &[&str]) -> Vec<PathBuf> {
        let mut results = Vec::new();
        for file in self.list_files() {
            if let Some(content) = self.read_file(&file) {
                let lower = content.to_lowercase();
                let matches = keywords.iter().filter(|&&k| {
                    let k = k.trim().to_lowercase();
                    k.len() >= 3 && lower.contains(&k)
                }).count();
                if matches >= 1 {
                    results.push((file, matches));
                }
            }
        }
        results.sort_by(|a, b| b.1.cmp(&a.1));
        results.into_iter().map(|(p, _)| p).collect()
    }

    /// Extrai [[links]] do conteúdo no estilo wiki
    pub fn extract_links(content: &str) -> Vec<String> {
        let mut links = Vec::new();
        let mut search_start = 0;
        while let Some(start) = content[search_start..].find("[[") {
            let abs_start = search_start + start + 2;
            if let Some(end) = content[abs_start..].find("]]") {
                let link = &content[abs_start..abs_start + end];
                let link = link.trim();
                if !link.is_empty() && !link.starts_with("_") {
                    links.push(link.to_string());
                }
                search_start = abs_start + end + 2;
            } else {
                break;
            }
        }
        links
    }

    /// Resolve um link [[caminho/arquivo]] para um PathBuf absoluto
    pub fn resolve_link(&self, link: &str) -> Option<PathBuf> {
        let mut path = self.root.clone();
        for part in link.split('/') {
            let part = part.trim();
            if part.is_empty() { continue; }
            path.push(part);
        }
        // Tenta com .md
        let with_ext = {
            let mut p = path.clone();
            p.set_extension("md");
            p
        };
        if with_ext.exists() {
            return Some(with_ext);
        }
        // Tenta sem extensão (já pode ter .md)
        if path.exists() {
            return Some(path);
        }
        None
    }

    /// Escolhe um arquivo aleatório para começar
    pub fn random_start(&self) -> Option<PathBuf> {
        let files = self.list_files();
        if files.is_empty() {
            return None;
        }
        let idx = fastrand::usize(0..files.len());
        Some(files[idx].clone())
    }

    /// Escolhe o próximo arquivo baseado nos [[links]] do conteúdo atual
    pub fn next_from_links(&self, content: &str) -> Option<PathBuf> {
        let links = Self::extract_links(content);
        let valid: Vec<PathBuf> = links.iter()
            .filter_map(|l| self.resolve_link(l))
            .filter(|p| p.exists())
            .collect();

        if valid.is_empty() {
            return None;
        }

        let idx = fastrand::usize(0..valid.len());
        Some(valid[idx].clone())
    }

    /// Explora um passo: lê o arquivo atual, extrai links, escolhe o próximo
    /// Retorna (conteúdo lido, próximo caminho, nome do arquivo)
    pub fn step(&mut self, current: &Path) -> Option<(String, Option<PathBuf>, String)> {
        let name = current
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("desconhecido")
            .to_string();

        let content = self.read_file(current)?;
        self.visit_log.push(name.clone());

        let next = self.next_from_links(&content);

        Some((content, next, name))
    }

    /// Exploração autônoma: N passos, treinando o n-grama a cada passo
    pub fn explore<F>(&mut self, n_steps: usize, mut train: F)
    where
        F: FnMut(&str, &str),
    {
        let mut current = match self.random_start() {
            Some(p) => p,
            None => {
                eprintln!("  Underworld vazio!");
                return;
            }
        };

        for step_n in 0..n_steps {
            let (content, next, name) = match self.step(&current) {
                Some(r) => r,
                None => break,
            };

            println!(
                "  [Underworld] Passo {}: {} ({} bytes)",
                step_n + 1,
                name,
                content.len()
            );

            train(&name, &content);

            match next {
                Some(p) => current = p,
                None => {
                    println!("  [Underworld] Sem mais links — reiniciando");
                    current = match self.random_start() {
                        Some(p) => p,
                        None => break,
                    };
                }
            }
        }

        println!(
            "  [Underworld] Exploração concluída: {} arquivos visitados",
            self.visit_log.len()
        );
    }

    pub fn visit_log(&self) -> &[String] {
        &self.visit_log
    }
}
