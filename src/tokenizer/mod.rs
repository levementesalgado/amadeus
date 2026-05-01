use std::collections::HashMap;

/// BPE tokenizer loaded from GGUF model metadata.
///
/// Vocab is a list of strings.  Merges are (left_id, right_id) pairs
/// ranked by frequency.  Encoding applies the BPE merge rules greedily.
pub struct Tokenizer {
    pub vocab: Vec<String>,
    byte_encoder: HashMap<Vec<u8>, u32>,
    merges: HashMap<(u32, u32), usize>, // pair → rank
    pub bos: u32,
    pub eos: u32,
    pub unk: u32,
    pub pad: Option<u32>,
}

impl Tokenizer {
    pub fn new(vocab: Vec<String>, merges_raw: Vec<(u32, u32)>) -> Self {
        let byte_encoder: HashMap<Vec<u8>, u32> = vocab
            .iter()
            .enumerate()
            .map(|(i, s)| (s.as_bytes().to_vec(), i as u32))
            .collect();

        let merges: HashMap<(u32, u32), usize> = merges_raw
            .into_iter()
            .enumerate()
            .map(|(rank, pair)| (pair, rank))
            .collect();

        Self {
            vocab,
            byte_encoder,
            merges,
            bos: 1,
            eos: 2,
            unk: 0,
            pad: None,
        }
    }

    /// Encode text into token ids using BPE.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        // 1. split into chars → byte-level pieces
        let pieces: Vec<u32> = text
            .as_bytes()
            .iter()
            .map(|&b| self.byte_encoder.get::<[u8]>(&[b][..]).copied().unwrap_or(self.unk))
            .collect();

        if pieces.is_empty() {
            return pieces;
        }

        // 2. greedy BPE merging
        let mut ids = pieces;
        loop {
            let mut best_rank = usize::MAX;
            let mut best_idx = None;

            for i in 0..ids.len().saturating_sub(1) {
                let pair = (ids[i], ids[i + 1]);
                if let Some(&rank) = self.merges.get(&pair) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_idx = Some(i);
                    }
                }
            }

            match best_idx {
                None => break,
                Some(idx) => {
                    let merged = self
                        .merges
                        .iter()
                        .find(|&(k, _)| *k == (ids[idx], ids[idx + 1]))
                        .and_then(|(_, &_r)| {
                            // reconstruct merged token id from vocab
                            let left = ids[idx] as usize;
                            let right = ids[idx + 1] as usize;
                            let merged_str = format!("{}{}", self.vocab[left], self.vocab[right]);
                            self.byte_encoder.get(merged_str.as_bytes()).copied()
                        })
                        .unwrap_or(ids[idx]);

                    let mut new_ids = Vec::with_capacity(ids.len() - 1);
                    new_ids.extend_from_slice(&ids[..idx]);
                    new_ids.push(merged);
                    new_ids.extend_from_slice(&ids[idx + 2..]);
                    ids = new_ids;
                }
            }
        }

        ids
    }

    /// Decode token ids back into a string.
    pub fn decode(&self, tokens: &[u32]) -> String {
        tokens
            .iter()
            .filter_map(|id| self.vocab.get(*id as usize))
            .flat_map(|s| s.chars())
            .collect()
    }

    /// Decode a single token.
    pub fn decode_token(&self, token: u32) -> &str {
        self.vocab
            .get(token as usize)
            .map(|s| s.as_str())
            .unwrap_or("<unk>")
    }
}
