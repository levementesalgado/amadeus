use crate::amadeus_m::collatz::CollatzMode;
use crate::amadeus_m::pedagogy::DecisionStep;

pub struct FrictionReport {
    pub switches: Vec<ModeSwitch>,
    pub total_originalist: usize,
    pub total_vanguardist: usize,
    pub friction_score: f32,
}

pub struct ModeSwitch {
    pub step_index: usize,
    pub from: CollatzMode,
    pub to: CollatzMode,
    pub intensity: f32,
}

pub fn analyze_friction(trace: &[DecisionStep]) -> FrictionReport {
    let mut switches = Vec::new();
    let mut total_originalist = 0;
    let mut total_vanguardist = 0;

    if trace.is_empty() {
        return FrictionReport {
            switches,
            total_originalist: 0,
            total_vanguardist: 0,
            friction_score: 0.0,
        };
    }

    let mut prev_mode = trace[0].mode.clone();
    match prev_mode {
        CollatzMode::Originalist => total_originalist += 1,
        CollatzMode::Vanguardist => total_vanguardist += 1,
    }

    for (i, step) in trace.iter().enumerate().skip(1) {
        match &step.mode {
            CollatzMode::Originalist => total_originalist += 1,
            CollatzMode::Vanguardist => total_vanguardist += 1,
        }

        if step.mode != prev_mode {
            let intensity = if step.mode == CollatzMode::Vanguardist {
                (step.explore_at_step / 0.1).min(1.0)
            } else {
                (1.0 - step.explore_at_step / 0.1).clamp(0.0, 1.0)
            };

            switches.push(ModeSwitch {
                step_index: i,
                from: prev_mode.clone(),
                to: step.mode.clone(),
                intensity,
            });
            prev_mode = step.mode.clone();
        }
    }

    let total = trace.len() as f32;
    let friction_score = if total > 0.0 {
        let n_switches = switches.len() as f32;
        let avg_intensity = switches.iter().map(|s| s.intensity).sum::<f32>() / n_switches.max(1.0);
        (n_switches / total) * avg_intensity * 10.0
    } else {
        0.0
    };

    FrictionReport {
        switches,
        total_originalist,
        total_vanguardist,
        friction_score,
    }
}

pub fn describe_friction(report: &FrictionReport) -> String {
    let mut out = String::from("─── Fricção Cognitiva ───\n");

    if report.switches.is_empty() {
        out.push_str("  Nenhuma alternância de modo — fluxo homogêneo.\n");
        match report.total_originalist.cmp(&report.total_vanguardist) {
            std::cmp::Ordering::Greater => out.push_str("  Modo predominante: Originalista (foco global, conservador)\n"),
            std::cmp::Ordering::Less => out.push_str("  Modo predominante: Vanguardista (atenção local, criativo)\n"),
            std::cmp::Ordering::Equal => out.push_str("  Equilíbrio perfeito entre Originalista e Vanguardista\n"),
        }
        out.push_str(&format!("  Score de fricção: {:.2} (baixo — sem conflito interno)\n", report.friction_score));
        return out;
    }

    out.push_str(&format!(
        "  {} alternâncias de modo em {} passos (score: {:.2})\n",
        report.switches.len(),
        report.total_originalist + report.total_vanguardist,
        report.friction_score,
    ));

    for (_i, sw) in report.switches.iter().enumerate() {
        let from_name = match sw.from {
            CollatzMode::Originalist => "Originalista",
            CollatzMode::Vanguardist => "Vanguardista",
        };
        let to_name = match sw.to {
            CollatzMode::Originalist => "Originalista",
            CollatzMode::Vanguardist => "Vanguardista",
        };
        let description = match (&sw.from, &sw.to) {
            (CollatzMode::Originalist, CollatzMode::Vanguardist) =>
                "colapso → expansão: o modelo rompe a inércia conservadora e explode em criatividade",
            (CollatzMode::Vanguardist, CollatzMode::Originalist) =>
                "expansão → colapso: a excitação criativa se contrai em foco conservador",
            _ => "transição de modo",
        };
        let symbol = if sw.intensity > 0.7 { "⚡" } else { "·" };
        out.push_str(&format!(
            "  [{}] Passo {}: {} → {} {}\n    {}\n",
            symbol, sw.step_index, from_name, to_name, intensity_bar(sw.intensity), description
        ));
    }

    out.push_str(&format!(
        "  Balanço: {} Originalista / {} Vanguardista\n",
        report.total_originalist, report.total_vanguardist
    ));

    if report.friction_score > 5.0 {
        out.push_str("  ⚠ Fricção alta — conflito interno significativo entre modos\n");
    } else if report.friction_score > 2.0 {
        out.push_str("  Fricção moderada — tensão criativa saudável\n");
    } else {
        out.push_str("  Fricção baixa — fluxo estável\n");
    }

    out
}

fn intensity_bar(intensity: f32) -> String {
    let n = (intensity * 10.0) as usize;
    "█".repeat(n) + &"░".repeat(10 - n.min(10))
}
