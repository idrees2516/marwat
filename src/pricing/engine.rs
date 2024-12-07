use crate::types::{Pool, OptionType};
use crate::errors::{PanopticError, Result};
use ethers::types::U256;
use ndarray::{Array1, Array2};
use statrs::distribution::{Normal, ContinuousCDF};
use std::f64::consts::PI;

#[derive(Debug, Clone)]
pub struct Greeks {
    pub delta: f64,
    pub gamma: f64,
    pub vega: f64,
    pub theta: f64,
    pub rho: f64,
}

pub struct PricingEngine {
    pub risk_free_rate: f64,
    volatility_surface: VolatilitySurface,
    price_history: Vec<(u64, f64)>,
    volatility_history: Vec<(u64, f64)>,
}

impl PricingEngine {
    pub fn new(risk_free_rate: f64) -> Self {
        Self {
            risk_free_rate,
            volatility_surface: VolatilitySurface::new(),
            price_history: Vec::new(),
            volatility_history: Vec::new(),
        }
    }

    pub fn calculate_option_price(
        &self,
        pool: &Pool,
        option_type: OptionType,
        strike: U256,
        time_to_expiry: f64,
    ) -> Result<U256> {
        let spot_price = pool.current_price()?.as_u128() as f64;
        let strike_price = strike.as_u128() as f64;
        let volatility = self.get_implied_volatility(pool)?;

        self.calculate_option_price_with_params(
            pool,
            option_type,
            strike,
            time_to_expiry,
            U256::from_f64_lossy(spot_price),
            volatility,
            self.risk_free_rate,
        )
    }

    pub fn calculate_option_price_with_params(
        &self,
        pool: &Pool,
        option_type: OptionType,
        strike: U256,
        time_to_expiry: f64,
        spot_price: U256,
        volatility: f64,
        rate: f64,
    ) -> Result<U256> {
        let s = spot_price.as_u128() as f64;
        let k = strike.as_u128() as f64;
        let t = time_to_expiry;
        let r = rate;
        let sigma = volatility;

        if t <= 0.0 {
            return match option_type {
                OptionType::Call => Ok(U256::from_f64_lossy((s - k).max(0.0))),
                OptionType::Put => Ok(U256::from_f64_lossy((k - s).max(0.0))),
            };
        }

        let d1 = (s.ln() - k.ln() + (r + sigma * sigma / 2.0) * t) / (sigma * t.sqrt());
        let d2 = d1 - sigma * t.sqrt();

        let normal = Normal::new(0.0, 1.0)?;
        let price = match option_type {
            OptionType::Call => {
                s * normal.cdf(d1) - k * (-r * t).exp() * normal.cdf(d2)
            },
            OptionType::Put => {
                k * (-r * t).exp() * normal.cdf(-d2) - s * normal.cdf(-d1)
            },
        };

        Ok(U256::from_f64_lossy(price))
    }

    pub fn calculate_greeks(
        &self,
        pool: &Pool,
        option_type: OptionType,
        strike: U256,
        time_to_expiry: f64,
    ) -> Result<Greeks> {
        let s = pool.current_price()?.as_u128() as f64;
        let k = strike.as_u128() as f64;
        let t = time_to_expiry;
        let r = self.risk_free_rate;
        let sigma = self.get_implied_volatility(pool)?;

        if t <= 0.0 {
            return Ok(Greeks {
                delta: 0.0,
                gamma: 0.0,
                vega: 0.0,
                theta: 0.0,
                rho: 0.0,
            });
        }

        let d1 = (s.ln() - k.ln() + (r + sigma * sigma / 2.0) * t) / (sigma * t.sqrt());
        let d2 = d1 - sigma * t.sqrt();
        let normal = Normal::new(0.0, 1.0)?;
        let n_d1 = normal.cdf(d1);
        let n_d2 = normal.cdf(d2);
        let n_prime_d1 = (-d1 * d1 / 2.0).exp() / (2.0 * PI).sqrt();

        let (delta, gamma, vega, theta, rho) = match option_type {
            OptionType::Call => {
                let delta = n_d1;
                let gamma = n_prime_d1 / (s * sigma * t.sqrt());
                let vega = s * t.sqrt() * n_prime_d1;
                let theta = -(s * sigma * n_prime_d1) / (2.0 * t.sqrt())
                           - r * k * (-r * t).exp() * n_d2;
                let rho = k * t * (-r * t).exp() * n_d2;
                (delta, gamma, vega, theta, rho)
            },
            OptionType::Put => {
                let delta = n_d1 - 1.0;
                let gamma = n_prime_d1 / (s * sigma * t.sqrt());
                let vega = s * t.sqrt() * n_prime_d1;
                let theta = -(s * sigma * n_prime_d1) / (2.0 * t.sqrt())
                           + r * k * (-r * t).exp() * (1.0 - n_d2);
                let rho = -k * t * (-r * t).exp() * (1.0 - n_d2);
                (delta, gamma, vega, theta, rho)
            },
        };

        Ok(Greeks {
            delta,
            gamma,
            vega,
            theta,
            rho,
        })
    }

    pub fn get_implied_volatility(&self, pool: &Pool) -> Result<f64> {
        let strike = pool.current_price()?;
        let time_to_expiry = pool.time_to_expiry()?.as_u128() as f64 / 31_536_000.0; // Convert to years
        
        self.volatility_surface.get_volatility(
            strike.as_u128() as f64,
            time_to_expiry,
        )
    }

    pub fn update_market_data(
        &mut self,
        pool: &Pool,
        block_number: u64,
        price: f64,
        volatility: f64,
    ) {
        self.price_history.push((block_number, price));
        self.volatility_history.push((block_number, volatility));
        
        // Keep only recent history
        while self.price_history.len() > 1000 {
            self.price_history.remove(0);
        }
        while self.volatility_history.len() > 1000 {
            self.volatility_history.remove(0);
        }
    }

    pub fn calculate_historical_volatility(&self, window_size: usize) -> Option<f64> {
        if self.price_history.len() < window_size {
            return None;
        }

        let returns: Vec<f64> = self.price_history
            .windows(2)
            .map(|w| (w[1].1 / w[0].1).ln())
            .collect();

        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|&r| (r - mean).powi(2))
            .sum::<f64>() / (returns.len() - 1) as f64;

        Some((variance * 252.0).sqrt()) // Annualized volatility
    }

    pub fn calculate_forward_price(
        &self,
        pool: &Pool,
        time_to_expiry: f64,
    ) -> Result<f64> {
        let spot_price = pool.current_price()?.as_u128() as f64;
        Ok(spot_price * (self.risk_free_rate * time_to_expiry).exp())
    }

    pub async fn calculate_option_price_with_params(
        &self,
        pool: Address,
        params: &OptionPricingParams,
    ) -> Result<U256> {
        // Get market data
        let spot_price = self.get_current_price(pool).await?;
        let vol = self.get_implied_volatility(pool).await?;
        let rates = self.get_interest_rates(pool).await?;
        
        // Calculate option price using multiple models
        let mut prices = Vec::new();
        
        // 1. Black-Scholes price
        let bs_price = self.calculate_black_scholes_price(
            spot_price,
            params.strike,
            params.time_to_expiry,
            vol,
            rates.risk_free_rate,
            params.option_type,
        )?;
        prices.push(bs_price);
        
        // 2. Local Volatility price
        let lv_price = self.calculate_local_vol_price(
            pool,
            spot_price,
            params,
        ).await?;
        prices.push(lv_price);
        
        // 3. Stochastic Volatility price
        let sv_price = self.calculate_stochastic_vol_price(
            pool,
            spot_price,
            params,
        ).await?;
        prices.push(sv_price);
        
        // Calculate weighted average price
        let weights = self.calculate_model_weights(pool, params)?;
        let weighted_price = prices.iter()
            .zip(weights.iter())
            .map(|(price, weight)| price * weight)
            .sum::<f64>();
            
        // Apply market adjustments
        let adjusted_price = self.apply_market_adjustments(
            pool,
            weighted_price,
            params,
        ).await?;
        
        Ok(U256::from((adjusted_price * 1e18) as u128))
    }

    fn calculate_black_scholes_price(
        &self,
        spot: U256,
        strike: U256,
        time: f64,
        vol: f64,
        rate: f64,
        option_type: OptionType,
    ) -> Result<f64> {
        let s = spot.as_u128() as f64;
        let k = strike.as_u128() as f64;
        let sqrt_time = time.sqrt();
        
        let d1 = (s.ln() / k + (rate + vol * vol / 2.0) * time) / (vol * sqrt_time);
        let d2 = d1 - vol * sqrt_time;
        
        let price = match option_type {
            OptionType::Call => {
                s * standard_normal_cdf(d1) - k * (-rate * time).exp() * standard_normal_cdf(d2)
            }
            OptionType::Put => {
                k * (-rate * time).exp() * standard_normal_cdf(-d2) - s * standard_normal_cdf(-d1)
            }
        };
        
        Ok(price)
    }

    async fn calculate_local_vol_price(
        &self,
        pool: Address,
        spot: U256,
        params: &OptionPricingParams,
    ) -> Result<f64> {
        // Get local volatility surface
        let surface = self.vol_surface.get_local_volatility_surface(pool).await?;
        
        // Setup finite difference grid
        let grid = self.setup_finite_difference_grid(spot, params)?;
        
        // Solve PDE using ADI method
        let price = self.solve_local_vol_pde(
            grid,
            surface,
            params,
        )?;
        
        Ok(price)
    }

    async fn calculate_stochastic_vol_price(
        &self,
        pool: Address,
        spot: U256,
        params: &OptionPricingParams,
    ) -> Result<f64> {
        // Get Heston parameters
        let heston_params = self.get_heston_parameters(pool).await?;
        
        // Monte Carlo simulation
        let n_paths = 10_000;
        let n_steps = 100;
        let dt = params.time_to_expiry / n_steps as f64;
        
        let mut prices = Vec::with_capacity(n_paths);
        
        for _ in 0..n_paths {
            let path_price = self.simulate_heston_path(
                spot,
                params,
                &heston_params,
                n_steps,
                dt,
            )?;
            prices.push(path_price);
        }
        
        // Calculate mean price
        let price = prices.iter().sum::<f64>() / n_paths as f64;
        
        Ok(price)
    }

    fn calculate_model_weights(
        &self,
        pool: Address,
        params: &OptionPricingParams,
    ) -> Result<Vec<f64>> {
        // Calculate weights based on:
        // 1. Market regime
        // 2. Option moneyness
        // 3. Time to expiry
        // 4. Historical model performance
        
        let market_regime = self.detect_market_regime(pool)?;
        let moneyness = self.calculate_moneyness(params)?;
        let model_performance = self.get_model_performance(pool)?;
        
        let mut weights = vec![0.0; 3];
        
        match market_regime {
            MarketRegime::Normal => {
                // Prefer Black-Scholes for near-the-money options
                if moneyness.abs() < 0.1 {
                    weights[0] = 0.6; // BS
                    weights[1] = 0.2; // LV
                    weights[2] = 0.2; // SV
                } else {
                    weights[0] = 0.3; // BS
                    weights[1] = 0.4; // LV
                    weights[2] = 0.3; // SV
                }
            }
            MarketRegime::HighVolatility => {
                // Prefer stochastic volatility model
                weights[0] = 0.2; // BS
                weights[1] = 0.3; // LV
                weights[2] = 0.5; // SV
            }
            MarketRegime::LowLiquidity => {
                // Prefer simpler models
                weights[0] = 0.5; // BS
                weights[1] = 0.3; // LV
                weights[2] = 0.2; // SV
            }
        }
        
        // Adjust weights based on model performance
        for i in 0..3 {
            weights[i] *= model_performance[i];
        }
        
        // Normalize weights
        let sum = weights.iter().sum::<f64>();
        for w in weights.iter_mut() {
            *w /= sum;
        }
        
        Ok(weights)
    }

    async fn apply_market_adjustments(
        &self,
        pool: Address,
        price: f64,
        params: &OptionPricingParams,
    ) -> Result<f64> {
        let mut adjusted_price = price;
        
        // 1. Liquidity adjustment
        let liquidity_discount = self.calculate_liquidity_discount(pool).await?;
        adjusted_price *= 1.0 - liquidity_discount;
        
        // 2. Skew adjustment
        let skew_adjustment = self.calculate_skew_adjustment(pool, params).await?;
        adjusted_price *= 1.0 + skew_adjustment;
        
        // 3. Term structure adjustment
        let term_adjustment = self.calculate_term_structure_adjustment(pool, params).await?;
        adjusted_price *= 1.0 + term_adjustment;
        
        // 4. Market impact
        let impact = self.estimate_market_impact(pool, params.size).await?;
        adjusted_price *= 1.0 + impact;
        
        Ok(adjusted_price)
    }

    fn simulate_heston_path(
        &self,
        spot: U256,
        params: &OptionPricingParams,
        heston: &HestonParameters,
        n_steps: usize,
        dt: f64,
    ) -> Result<f64> {
        let mut s = spot.as_u128() as f64;
        let mut v = heston.initial_variance;
        
        let sqrt_dt = dt.sqrt();
        
        for _ in 0..n_steps {
            // Generate correlated random numbers
            let z1 = random_normal();
            let z2 = heston.correlation * z1 + (1.0 - heston.correlation * heston.correlation).sqrt() * random_normal();
            
            // Update variance
            let sqrt_v = v.sqrt();
            v += heston.kappa * (heston.theta - v) * dt + heston.xi * sqrt_v * z1 * sqrt_dt;
            v = v.max(0.0);
            
            // Update price
            let drift = (params.risk_free_rate - 0.5 * v) * dt;
            let diffusion = sqrt_v * z2 * sqrt_dt;
            s *= (drift + diffusion).exp();
        }
        
        // Calculate option payoff
        let payoff = match params.option_type {
            OptionType::Call => (s - params.strike.as_u128() as f64).max(0.0),
            OptionType::Put => (params.strike.as_u128() as f64 - s).max(0.0),
        };
        
        // Discount payoff
        let discount = (-params.risk_free_rate * params.time_to_expiry).exp();
        Ok(payoff * discount)
    }
}

#[derive(Debug)]
pub struct OptionPricingParams {
    pub strike: U256,
    pub time_to_expiry: f64,
    pub option_type: OptionType,
    pub size: U256,
    pub risk_free_rate: f64,
}

#[derive(Debug)]
pub enum OptionType {
    Call,
    Put,
}

#[derive(Debug)]
pub enum MarketRegime {
    Normal,
    HighVolatility,
    LowLiquidity,
}

#[derive(Debug)]
pub struct HestonParameters {
    pub initial_variance: f64,
    pub kappa: f64,        // Mean reversion speed
    pub theta: f64,        // Long-term variance
    pub xi: f64,           // Volatility of variance
    pub correlation: f64,  // Price-volatility correlation
}

fn standard_normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / 2.0f64.sqrt()))
}

fn erf(x: f64) -> f64 {
    // Approximation of error function
    let a1 =  0.254829592;
    let a2 = -0.284496736;
    let a3 =  1.421413741;
    let a4 = -1.453152027;
    let a5 =  1.061405429;
    let p  =  0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

fn random_normal() -> f64 {
    // Box-Muller transform
    let u1 = rand::random::<f64>();
    let u2 = rand::random::<f64>();
    
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

struct VolatilitySurface {
    strikes: Array1<f64>,
    expiries: Array1<f64>,
    volatilities: Array2<f64>,
}

impl VolatilitySurface {
    fn new() -> Self {
        Self {
            strikes: Array1::zeros(0),
            expiries: Array1::zeros(0),
            volatilities: Array2::zeros((0, 0)),
        }
    }

    fn get_volatility(&self, strike: f64, time_to_expiry: f64) -> Result<f64> {
        if self.strikes.len() == 0 || self.expiries.len() == 0 {
            return Ok(0.3); // Default volatility if surface is empty
        }

        // Find nearest strike and expiry indices
        let strike_idx = self.find_nearest_index(&self.strikes, strike);
        let expiry_idx = self.find_nearest_index(&self.expiries, time_to_expiry);

        // Bilinear interpolation
        let vol = self.volatilities[[strike_idx, expiry_idx]];
        Ok(vol)
    }

    fn find_nearest_index(&self, arr: &Array1<f64>, value: f64) -> usize {
        let mut min_idx = 0;
        let mut min_diff = f64::INFINITY;

        for (i, &x) in arr.iter().enumerate() {
            let diff = (x - value).abs();
            if diff < min_diff {
                min_diff = diff;
                min_idx = i;
            }
        }

        min_idx
    }

    fn update_surface(
        &mut self,
        strikes: Vec<f64>,
        expiries: Vec<f64>,
        volatilities: Vec<Vec<f64>>,
    ) -> Result<()> {
        let n_strikes = strikes.len();
        let n_expiries = expiries.len();

        if volatilities.len() != n_strikes || volatilities.iter().any(|row| row.len() != n_expiries) {
            return Err(PanopticError::Custom("Invalid volatility surface dimensions".into()));
        }

        self.strikes = Array1::from_vec(strikes);
        self.expiries = Array1::from_vec(expiries);
        self.volatilities = Array2::from_shape_vec(
            (n_strikes, n_expiries),
            volatilities.into_iter().flatten().collect(),
        )?;

        Ok(())
    }
}
