use crate::samplers::Annealer;

pub struct ConstantAnnealer {
    temperature: f64,
}

impl ConstantAnnealer {
    pub fn new(temperature: f64) -> Self {
        Self { temperature }
    }
}

impl Annealer for ConstantAnnealer {
    fn temperature(&self, _sweep: usize) -> f64 {
        self.temperature
    }
}

pub struct LinearAnnealer {
    initial: f64,
    rate: f64,
    min_temperature: f64,
}

impl LinearAnnealer {
    pub fn new(initial: f64, rate: f64, min_temperature: f64) -> Self {
        Self { initial, rate, min_temperature }
    }
}

impl Annealer for LinearAnnealer {
    fn temperature(&self, sweep: usize) -> f64 {
        (self.initial - self.rate * sweep as f64).max(self.min_temperature)
    }
}

pub struct ExponentialAnnealer {
    initial: f64,
    rate: f64,
    min_temperature: f64,
}

impl ExponentialAnnealer {
    pub fn new(initial: f64, rate: f64, min_temperature: f64) -> Self {
        Self { initial, rate, min_temperature }
    }
}

impl Annealer for ExponentialAnnealer {
    fn temperature(&self, sweep: usize) -> f64 {
        (self.initial * self.rate.powi(sweep as i32)).max(self.min_temperature)
    }
}

pub struct LogarithmicAnnealer {
    initial: f64,
    min_temperature: f64
}

impl LogarithmicAnnealer {
    pub fn new(initial: f64, min_temperature: f64) -> Self {
        Self { initial, min_temperature }
    }
}

impl Annealer for LogarithmicAnnealer {
    fn temperature(&self, sweep: usize) -> f64 {
        (self.initial / (1.0 + (sweep as f64).ln_1p())).max(self.min_temperature)
    }
}