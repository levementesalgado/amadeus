/// Oscilador Collatz — regula atenção entre Originalista (par) e Vanguardista (ímpar)
///
/// Sequência: se n é par → n/2 (colapso, modo conservador)
///            se n é ímpar → 3n+1 (expansão, modo criativo)
///
/// A órbita de Collatz substitui a oscilação senoidal do ExcursionState
/// com uma dinâmica determinística, caótica e universal.

#[derive(Clone, Debug, PartialEq)]
pub enum CollatzMode {
    /// n par — atenção global, conservador, segue a Constituição
    Originalist,
    /// n ímpar — atenção local, criativo, ativa Emendas
    Vanguardist,
}

#[derive(Clone)]
pub struct CollatzOscillator {
    pub n: u64,
    pub seed: u64,
    pub steps: u64,
    /// Profundidade: quantas iterações desde o último reset
    pub depth: u8,
    /// Quantos passos consecutivos no mesmo modo
    pub streak: u8,
    /// Pico já atingido nesta órbita (maior valor de n)
    pub peak: u64,
}

impl CollatzOscillator {
    pub fn new(seed: u64) -> Self {
        Self {
            n: seed,
            seed,
            steps: 0,
            depth: 0,
            streak: 0,
            peak: seed,
        }
    }

    /// Avança um passo na sequência de Collatz
    pub fn step(&mut self) {
        self.steps += 1;
        self.depth = (self.depth + 1).min(10);

        let was_even = self.n % 2 == 0;
        if self.n % 2 == 0 {
            self.n /= 2;
        } else {
            self.n = self.n.wrapping_mul(3).wrapping_add(1);
        }

        if self.n > self.peak {
            self.peak = self.n;
        }

        // Streak tracking
        let now_even = self.n % 2 == 0;
        if now_even == was_even {
            self.streak = (self.streak + 1).min(10);
        } else {
            self.streak = 1;
        }

        // Reset if we reach the 4-2-1 loop
        if self.n == 1 {
            let next = if self.steps % 2 == 0 { 3 * 1 + 1 } else { 1 };
            if next == 4 || next == 1 {
                // Stay in the loop — don't reset unless steps is large
                if self.steps > 100 {
                    let new_seed = self.seed.wrapping_add(self.steps);
                    self.n = if new_seed == 0 { 27 } else { new_seed };
                    self.depth = 0;
                    self.streak = 0;
                    self.peak = self.n;
                }
            }
        }
    }

    pub fn mode(&self) -> CollatzMode {
        if self.n % 2 == 0 {
            CollatzMode::Originalist
        } else {
            CollatzMode::Vanguardist
        }
    }

    /// Normaliza n para 0.0-1.0 (útil para modulação)
    pub fn normalized(&self) -> f32 {
        let base = self.peak.max(self.seed) as f64;
        if base > 0.0 {
            (self.n as f64 / base).clamp(0.0, 1.0) as f32
        } else {
            0.5
        }
    }

    /// Fator de exploração: Originalist reduz, Vanguardist aumenta
    pub fn explore_factor(&self) -> f32 {
        match self.mode() {
            CollatzMode::Originalist => {
                // Quanto mais próximo de 1 (ciclo 4-2-1), mais conservador
                0.5 + (1.0 - self.normalized()) * 0.3
            }
            CollatzMode::Vanguardist => {
                // Quanto maior o valor, mais criativo
                1.0 + self.normalized() * 1.5
            }
        }
    }

    /// Fator de temperatura: Vanguardist aquece, Originalist esfria
    pub fn temp_factor(&self) -> f32 {
        match self.mode() {
            CollatzMode::Originalist => 0.8 + self.normalized() * 0.3,
            CollatzMode::Vanguardist => 1.0 + self.normalized() * 1.2,
        }
    }

    /// Fator de foco: Originalist foca mais
    pub fn focus_factor(&self) -> f32 {
        match self.mode() {
            CollatzMode::Originalist => 1.0 + (1.0 - self.normalized()) * 0.5,
            CollatzMode::Vanguardist => 0.5 + (1.0 - self.normalized()) * 0.3,
        }
    }

    /// Descrição legível do estado atual
    pub fn describe(&self) -> String {
        let mode_str = match self.mode() {
            CollatzMode::Originalist => "Originalista",
            CollatzMode::Vanguardist => "Vanguardista",
        };
        format!("Collatz n={} {} peak={} streak={}", self.n, mode_str, self.peak, self.streak)
    }
}
