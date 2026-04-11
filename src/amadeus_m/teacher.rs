use std::io::{self, Write};

#[derive(Debug, Clone)]
pub struct AffectFeedback {
    pub satisfaction: f32,
    pub mood_adjustment: Vec<f32>,
    pub intensity: f32,
}

impl AffectFeedback {
    pub fn neutral(d_affect: usize) -> Self {
        Self {
            satisfaction: 0.5,
            mood_adjustment: vec![0.0; d_affect],
            intensity: 0.0,
        }
    }

    pub fn positive(d_affect: usize) -> Self {
        let mut adj = vec![0.0; d_affect];
        for v in adj.iter_mut() {
            *v = fastrand::f32() * 0.2;
        }
        Self { satisfaction: 0.8, mood_adjustment: adj, intensity: 0.5 }
    }

    pub fn negative(d_affect: usize) -> Self {
        let mut adj = vec![0.0; d_affect];
        for v in adj.iter_mut() {
            *v = -fastrand::f32() * 0.2;
        }
        Self { satisfaction: 0.2, mood_adjustment: adj, intensity: 0.7 }
    }
}

pub struct HumanTeacher {
    pub d_affect: usize,
    pub last_feedback: Option<AffectFeedback>,
}

impl HumanTeacher {
    pub fn new(d_affect: usize) -> Self {
        Self { d_affect, last_feedback: None }
    }

    pub fn auto_react(&mut self, satisfaction: f32, affect: char) -> AffectFeedback {
        let sat = satisfaction.clamp(0.0, 1.0);
        let (adj, intensity) = match affect {
            'a' | 'A' => (vec![ 0.3; self.d_affect], 0.6),
            't' | 'T' => (vec![-0.3; self.d_affect], 0.7),
            'e' | 'E' => {
                let mut a = vec![0.0; self.d_affect];
                for v in a.iter_mut() { *v = fastrand::f32() * 0.4 - 0.2; }
                (a, 0.8)
            }
            _ => (vec![0.0; self.d_affect], 0.0),
        };
        let fb = AffectFeedback { satisfaction: sat, mood_adjustment: adj, intensity };
        self.last_feedback = Some(fb.clone());
        fb
    }

    pub fn react(&mut self, output_token: u32, text: &str) -> AffectFeedback {
        println!("\n  Modelo gerou: [{}] {}", output_token, text);
        print!("  Satisfação (1-5) [3]: ");
        io::stdout().flush().ok();

        let mut buf = String::new();
        io::stdin().read_line(&mut buf).ok();
        let sat = buf.trim().parse::<f32>().unwrap_or(3.0).clamp(1.0, 5.0) / 5.0;

        print!("  Afeto: [a]legria [t]risteza [e]stranhamento [n]eutro [s]air: ");
        io::stdout().flush().ok();
        buf.clear();
        io::stdin().read_line(&mut buf).ok();

        let (adj, intensity) = match buf.trim() {
            "a" => (vec![ 0.3; self.d_affect], 0.6),
            "t" => (vec![-0.3; self.d_affect], 0.7),
            "e" => {
                let mut a = vec![0.0; self.d_affect];
                for v in a.iter_mut() { *v = fastrand::f32() * 0.4 - 0.2; }
                (a, 0.8)
            }
            "s" => std::process::exit(0),
            _ => (vec![0.0; self.d_affect], 0.0),
        };

        let fb = AffectFeedback { satisfaction: sat, mood_adjustment: adj, intensity };
        self.last_feedback = Some(fb.clone());
        fb
    }
}
