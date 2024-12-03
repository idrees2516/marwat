use super::*;
use statrs::distribution::{Normal, ContinuousCDF};
use std::f64::consts::{E, PI};

#[derive(Debug, Clone, Copy)]
pub enum PricingModel {
    BlackScholes,
    Heston,
    SABR,
    LocalVolatility,
}

pub struct HestonParameters {
    pub v0: f64,      // Initial variance
    pub kappa: f64,   // Mean reversion speed
    pub theta: f64,   // Long-term variance
    pub sigma: f64,   // Volatility of variance
    pub rho: f64,     // Correlation
}

pub struct SABRParameters {
    pub alpha: f64,   // Initial volatility
    pub beta: f64,    // CEV parameter
    pub nu: f64,      // Volatility of volatility
    pub rho: f64,     // Correlation
}

impl PricingEngine {
    pub fn price_with_model(
        &self,
        model: PricingModel,
        spot: f64,
        strike: f64,
        time: f64,
        is_call: bool,
        params: Option<&[f64]>,
    ) -> Result<f64> {
        match model {
            PricingModel::BlackScholes => {
                self.black_scholes(spot, strike, time, self.risk_free_rate, params.unwrap()[0], is_call)
            }
            PricingModel::Heston => {
                if let Some(params) = params {
                    let heston_params = HestonParameters {
                        v0: params[0],
                        kappa: params[1],
                        theta: params[2],
                        sigma: params[3],
                        rho: params[4],
                    };
                    self.heston(spot, strike, time, is_call, &heston_params)
                } else {
                    Err(PanopticError::InvalidParameters)
                }
            }
            PricingModel::SABR => {
                if let Some(params) = params {
                    let sabr_params = SABRParameters {
                        alpha: params[0],
                        beta: params[1],
                        nu: params[2],
                        rho: params[3],
                    };
                    self.sabr(spot, strike, time, is_call, &sabr_params)
                } else {
                    Err(PanopticError::InvalidParameters)
                }
            }
            PricingModel::LocalVolatility => {
                self.local_volatility(spot, strike, time, is_call, params)
            }
        }
    }

    fn heston(
        &self,
        spot: f64,
        strike: f64,
        time: f64,
        is_call: bool,
        params: &HestonParameters,
    ) -> Result<f64> {
        const N_STEPS: usize = 100;
        const N_PATHS: usize = 10000;
        
        let dt = time / N_STEPS as f64;
        let mut price_sum = 0.0;
        let mut rng = rand::thread_rng();
        
        for _ in 0..N_PATHS {
            let mut s = spot;
            let mut v = params.v0;
            
            for _ in 0..N_STEPS {
                let z1: f64 = rng.sample(rand::distributions::Standard);
                let z2: f64 = rng.sample(rand::distributions::Standard);
                let z2 = params.rho * z1 + (1.0 - params.rho * params.rho).sqrt() * z2;
                
                let ds = s * (self.risk_free_rate * dt + v.sqrt() * z1 * dt.sqrt());
                let dv = params.kappa * (params.theta - v) * dt + 
                        params.sigma * v.sqrt() * z2 * dt.sqrt();
                
                s += ds;
                v = (v + dv).max(0.0);
            }
            
            let payoff = if is_call {
                (s - strike).max(0.0)
            } else {
                (strike - s).max(0.0)
            };
            
            price_sum += payoff;
        }
        
        let price = price_sum / N_PATHS as f64 * (-self.risk_free_rate * time).exp();
        Ok(price)
    }

    fn sabr(
        &self,
        spot: f64,
        strike: f64,
        time: f64,
        is_call: bool,
        params: &SABRParameters,
    ) -> Result<f64> {
        let f = spot;
        let k = strike;
        let t = time;
        
        // Calculate SABR implied volatility
        let fk_mid = (f * k).sqrt();
        let log_fk = (f / k).ln();
        
        let gamma1 = params.beta / fk_mid;
        let gamma2 = -params.beta * (1.0 - params.beta) * log_fk * log_fk / 24.0;
        let gamma3 = params.rho * params.beta * params.nu * log_fk / 4.0;
        let gamma4 = (2.0 - 3.0 * params.rho * params.rho) * params.nu * params.nu / 24.0;
        
        let sigma_atm = params.alpha * (1.0 + (gamma2 + gamma3 + gamma4) * t);
        let sigma = sigma_atm * (fk_mid.powf(-params.beta) * 
            (1.0 + gamma1 * log_fk + (gamma2 + gamma3 + gamma4) * t));
        
        // Use Black-Scholes with SABR implied volatility
        self.black_scholes(spot, strike, time, self.risk_free_rate, sigma, is_call)
    }

    fn local_volatility(
        &self,
        spot: f64,
        strike: f64,
        time: f64,
        is_call: bool,
        params: Option<&[f64]>,
    ) -> Result<f64> {
        const N_STEPS: usize = 100;
        const N_PATHS: usize = 10000;
        
        let dt = time / N_STEPS as f64;
        let mut price_sum = 0.0;
        let mut rng = rand::thread_rng();
        
        for _ in 0..N_PATHS {
            let mut s = spot;
            
            for _ in 0..N_STEPS {
                let local_vol = self.calculate_local_volatility(s, time, params)?;
                let z: f64 = rng.sample(rand::distributions::Standard);
                
                let ds = s * (self.risk_free_rate * dt + 
                    local_vol * z * dt.sqrt());
                s += ds;
            }
            
            let payoff = if is_call {
                (s - strike).max(0.0)
            } else {
                (strike - s).max(0.0)
            };
            
            price_sum += payoff;
        }
        
        let price = price_sum / N_PATHS as f64 * (-self.risk_free_rate * time).exp();
        Ok(price)
    }

    fn calculate_local_volatility(
        &self,
        spot: f64,
        time: f64,
        params: Option<&[f64]>,
    ) -> Result<f64> {
        // Implement Dupire's formula or parametric local volatility surface
        let base_vol = if let Some(params) = params {
            params[0]
        } else {
            0.2 // Default volatility
        };
        
        let level_effect = (spot / 100.0).ln().abs() * 0.1;
        let time_effect = (1.0 + time).sqrt() * 0.05;
        
        Ok(base_vol * (1.0 + level_effect + time_effect))
    }

    pub fn calibrate_heston(
        &self,
        market_prices: &[(f64, f64, f64, bool)], // (strike, time, price, is_call)
    ) -> Result<HestonParameters> {
        // Implement Levenberg-Marquardt optimization
        let initial_params = HestonParameters {
            v0: 0.04,
            kappa: 2.0,
            theta: 0.04,
            sigma: 0.3,
            rho: -0.7,
        };
        
        // For simplicity, return initial parameters
        // In practice, implement proper calibration
        Ok(initial_params)
    }

    pub fn calibrate_sabr(
        &self,
        market_prices: &[(f64, f64, f64, bool)], // (strike, time, price, is_call)
    ) -> Result<SABRParameters> {
        // Implement calibration algorithm
        let initial_params = SABRParameters {
            alpha: 0.2,
            beta: 0.5,
            nu: 0.4,
            rho: -0.4,
        };
        
        // For simplicity, return initial parameters
        // In practice, implement proper calibration
        Ok(initial_params)
    }
}
