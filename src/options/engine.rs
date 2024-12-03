use ethers::types::{U256, Address};
use std::collections::HashMap;
use crate::types::{Result, PanopticError};
use crate::math::{sqrt, ln, exp};

/// Represents the Greeks for an option position
#[derive(Debug, Clone)]
pub struct Greeks {
    pub delta: f64,
    pub gamma: f64,
    pub theta: f64,
    pub vega: f64,
    pub rho: f64,
}

/// Represents market parameters for options pricing
#[derive(Debug, Clone)]
pub struct MarketParameters {
    pub spot_price: f64,
    pub strike_price: f64,
    pub time_to_expiry: f64,
    pub risk_free_rate: f64,
    pub volatility: f64,
    pub is_call: bool,
}

/// Core options engine for pricing and risk calculations
pub struct OptionsEngine {
    volatility_surface: HashMap<(u64, u64), f64>,  // (strike, expiry) -> implied vol
    risk_free_rates: HashMap<u64, f64>,           // expiry -> rate
    min_strike_spacing: u64,
    max_strike_multiplier: u64,
}

impl OptionsEngine {
    pub fn new(min_strike_spacing: u64, max_strike_multiplier: u64) -> Self {
        Self {
            volatility_surface: HashMap::new(),
            risk_free_rates: HashMap::new(),
            min_strike_spacing,
            max_strike_multiplier,
        }
    }

    /// Calculate option price using Black-Scholes-Merton model
    pub fn calculate_option_price(&self, params: &MarketParameters) -> Result<f64> {
        if params.time_to_expiry <= 0.0 {
            return Err(PanopticError::InvalidParameter("Time to expiry must be positive".into()));
        }

        let d1 = (ln(params.spot_price / params.strike_price) + 
                 (params.risk_free_rate + 0.5 * params.volatility * params.volatility) * 
                 params.time_to_expiry) / 
                (params.volatility * sqrt(params.time_to_expiry));

        let d2 = d1 - params.volatility * sqrt(params.time_to_expiry);

        let price = if params.is_call {
            params.spot_price * self.normal_cdf(d1) - 
            params.strike_price * exp(-params.risk_free_rate * params.time_to_expiry) * 
            self.normal_cdf(d2)
        } else {
            params.strike_price * exp(-params.risk_free_rate * params.time_to_expiry) * 
            self.normal_cdf(-d2) - params.spot_price * self.normal_cdf(-d1)
        };

        Ok(price)
    }

    /// Calculate option Greeks
    pub fn calculate_greeks(&self, params: &MarketParameters) -> Result<Greeks> {
        let d1 = (ln(params.spot_price / params.strike_price) + 
                 (params.risk_free_rate + 0.5 * params.volatility * params.volatility) * 
                 params.time_to_expiry) / 
                (params.volatility * sqrt(params.time_to_expiry));

        let d2 = d1 - params.volatility * sqrt(params.time_to_expiry);

        let delta = if params.is_call {
            self.normal_cdf(d1)
        } else {
            self.normal_cdf(d1) - 1.0
        };

        let gamma = self.normal_pdf(d1) / 
                   (params.spot_price * params.volatility * sqrt(params.time_to_expiry));

        let theta = if params.is_call {
            -params.spot_price * self.normal_pdf(d1) * params.volatility / 
            (2.0 * sqrt(params.time_to_expiry)) -
            params.risk_free_rate * params.strike_price * 
            exp(-params.risk_free_rate * params.time_to_expiry) * self.normal_cdf(d2)
        } else {
            -params.spot_price * self.normal_pdf(d1) * params.volatility / 
            (2.0 * sqrt(params.time_to_expiry)) +
            params.risk_free_rate * params.strike_price * 
            exp(-params.risk_free_rate * params.time_to_expiry) * self.normal_cdf(-d2)
        };

        let vega = params.spot_price * sqrt(params.time_to_expiry) * self.normal_pdf(d1);

        let rho = if params.is_call {
            params.strike_price * params.time_to_expiry * 
            exp(-params.risk_free_rate * params.time_to_expiry) * self.normal_cdf(d2)
        } else {
            -params.strike_price * params.time_to_expiry * 
            exp(-params.risk_free_rate * params.time_to_expiry) * self.normal_cdf(-d2)
        };

        Ok(Greeks {
            delta,
            gamma,
            theta,
            vega,
            rho,
        })
    }

    /// Update volatility surface
    pub fn update_volatility(&mut self, strike: u64, expiry: u64, volatility: f64) -> Result<()> {
        if volatility <= 0.0 {
            return Err(PanopticError::InvalidParameter("Volatility must be positive".into()));
        }
        self.volatility_surface.insert((strike, expiry), volatility);
        Ok(())
    }

    /// Get implied volatility for given strike and expiry
    pub fn get_implied_volatility(&self, strike: u64, expiry: u64) -> Result<f64> {
        self.volatility_surface
            .get(&(strike, expiry))
            .copied()
            .ok_or_else(|| PanopticError::NoDataAvailable)
    }

    /// Update risk-free rate for given expiry
    pub fn update_risk_free_rate(&mut self, expiry: u64, rate: f64) -> Result<()> {
        if rate < -1.0 {
            return Err(PanopticError::InvalidParameter("Invalid risk-free rate".into()));
        }
        self.risk_free_rates.insert(expiry, rate);
        Ok(())
    }

    /// Calculate optimal strike prices for given spot price
    pub fn calculate_strike_prices(&self, spot_price: f64) -> Vec<u64> {
        let mut strikes = Vec::new();
        let base_strike = (spot_price / self.min_strike_spacing as f64).floor() as u64 * 
                         self.min_strike_spacing;
        
        for i in 1..=self.max_strike_multiplier {
            strikes.push(base_strike - i * self.min_strike_spacing);
            strikes.push(base_strike + i * self.min_strike_spacing);
        }
        
        strikes.sort_unstable();
        strikes
    }

    // Helper functions for normal distribution calculations
    fn normal_cdf(&self, x: f64) -> f64 {
        0.5 * (1.0 + erf(x / sqrt(2.0)))
    }

    fn normal_pdf(&self, x: f64) -> f64 {
        exp(-x * x / 2.0) / sqrt(2.0 * std::f64::consts::PI)
    }
}

// Error function approximation
fn erf(x: f64) -> f64 {
    let a1 =  0.254829592;
    let a2 = -0.284496736;
    let a3 =  1.421413741;
    let a4 = -1.453152027;
    let a5 =  1.061405429;
    let p  =  0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x);

    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_option_pricing() {
        let engine = OptionsEngine::new(100, 10);
        let params = MarketParameters {
            spot_price: 100.0,
            strike_price: 100.0,
            time_to_expiry: 1.0,
            risk_free_rate: 0.05,
            volatility: 0.2,
            is_call: true,
        };

        let price = engine.calculate_option_price(&params).unwrap();
        assert!(price > 0.0);
    }

    #[test]
    fn test_greeks_calculation() {
        let engine = OptionsEngine::new(100, 10);
        let params = MarketParameters {
            spot_price: 100.0,
            strike_price: 100.0,
            time_to_expiry: 1.0,
            risk_free_rate: 0.05,
            volatility: 0.2,
            is_call: true,
        };

        let greeks = engine.calculate_greeks(&params).unwrap();
        assert!(greeks.delta >= 0.0 && greeks.delta <= 1.0);
        assert!(greeks.gamma >= 0.0);
        assert!(greeks.vega >= 0.0);
    }
}
