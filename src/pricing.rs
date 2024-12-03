use crate::types::{OptionType, Pool, Strike};
use crate::errors::{PanopticError, Result};
use ethers::types::U256;
use statrs::distribution::{Normal, ContinuousCDF};
use std::f64::consts::E;

const SECONDS_PER_YEAR: f64 = 31_536_000.0; // 365 days
const BASIS_POINTS: f64 = 10_000.0;

pub struct PricingEngine {
    risk_free_rate: f64,
    volatility_adjustment: f64,
}

impl PricingEngine {
    pub fn new(risk_free_rate: f64, volatility_adjustment: f64) -> Self {
        Self {
            risk_free_rate,
            volatility_adjustment,
        }
    }

    pub fn calculate_option_price(
        &self,
        pool: &Pool,
        option_type: OptionType,
        strike: Strike,
        time_to_expiry: f64,
    ) -> Result<U256> {
        let spot_price = self.get_spot_price(pool);
        let volatility = self.calculate_implied_volatility(pool)?;
        
        let price = match option_type {
            OptionType::Call => {
                self.black_scholes(
                    spot_price,
                    strike.0.as_u128() as f64,
                    time_to_expiry,
                    self.risk_free_rate,
                    volatility,
                    true,
                )?
            }
            OptionType::Put => {
                self.black_scholes(
                    spot_price,
                    strike.0.as_u128() as f64,
                    time_to_expiry,
                    self.risk_free_rate,
                    volatility,
                    false,
                )?
            }
        };
        
        Ok(U256::from_f64_lossy(price))
    }

    pub fn calculate_implied_volatility(&self, pool: &Pool) -> Result<f64> {
        // Calculate implied volatility based on pool metrics
        let liquidity = pool.liquidity.as_u128() as f64;
        let tick_spacing = pool.tick_spacing as f64;
        let fee = pool.fee as f64 / BASIS_POINTS;
        
        // Base volatility from liquidity and tick spacing
        let base_volatility = (tick_spacing / liquidity).sqrt() * 100.0;
        
        // Adjust based on fee tier - higher fees suggest higher expected volatility
        let fee_adjustment = fee * 2.0;
        
        // Apply volatility adjustment factor
        let adjusted_volatility = base_volatility * (1.0 + fee_adjustment) * self.volatility_adjustment;
        
        Ok(adjusted_volatility.min(2.0)) // Cap at 200%
    }

    fn black_scholes(
        &self,
        spot: f64,
        strike: f64,
        time: f64,
        rate: f64,
        volatility: f64,
        is_call: bool,
    ) -> Result<f64> {
        if time <= 0.0 {
            return Ok(0.0);
        }

        let normal = Normal::new(0.0, 1.0).map_err(|_| PanopticError::Custom("Failed to create normal distribution".into()))?;
        
        let d1 = (spot.ln() + (rate + volatility * volatility / 2.0) * time)
            / (volatility * time.sqrt());
        
        let d2 = d1 - volatility * time.sqrt();
        
        let price = if is_call {
            spot * normal.cdf(d1) - strike * E.powf(-rate * time) * normal.cdf(d2)
        } else {
            strike * E.powf(-rate * time) * normal.cdf(-d2) - spot * normal.cdf(-d1)
        };
        
        Ok(price)
    }

    pub fn calculate_greeks(
        &self,
        pool: &Pool,
        option_type: OptionType,
        strike: Strike,
        time_to_expiry: f64,
    ) -> Result<Greeks> {
        let spot_price = self.get_spot_price(pool);
        let volatility = self.calculate_implied_volatility(pool)?;
        let strike_price = strike.0.as_u128() as f64;

        let normal = Normal::new(0.0, 1.0).map_err(|_| PanopticError::Custom("Failed to create normal distribution".into()))?;
        
        let d1 = (spot_price.ln() + (self.risk_free_rate + volatility * volatility / 2.0) * time_to_expiry)
            / (volatility * time_to_expiry.sqrt());
        
        let d2 = d1 - volatility * time_to_expiry.sqrt();
        
        let nd1 = normal.cdf(d1);
        let nd2 = normal.cdf(d2);
        let npd1 = normal.pdf(d1);
        
        let delta = match option_type {
            OptionType::Call => nd1,
            OptionType::Put => nd1 - 1.0,
        };
        
        let gamma = npd1 / (spot_price * volatility * time_to_expiry.sqrt());
        
        let theta = match option_type {
            OptionType::Call => {
                -spot_price * npd1 * volatility / (2.0 * time_to_expiry.sqrt())
                    - self.risk_free_rate * strike_price * E.powf(-self.risk_free_rate * time_to_expiry) * nd2
            }
            OptionType::Put => {
                -spot_price * npd1 * volatility / (2.0 * time_to_expiry.sqrt())
                    + self.risk_free_rate * strike_price * E.powf(-self.risk_free_rate * time_to_expiry) * (1.0 - nd2)
            }
        };
        
        let vega = spot_price * time_to_expiry.sqrt() * npd1;
        
        let rho = match option_type {
            OptionType::Call => {
                strike_price * time_to_expiry * E.powf(-self.risk_free_rate * time_to_expiry) * nd2
            }
            OptionType::Put => {
                -strike_price * time_to_expiry * E.powf(-self.risk_free_rate * time_to_expiry) * (1.0 - nd2)
            }
        };

        Ok(Greeks {
            delta,
            gamma,
            theta,
            vega,
            rho,
        })
    }

    fn get_spot_price(&self, pool: &Pool) -> f64 {
        (pool.sqrt_price_x96.pow(2.into()) / U256::from(2).pow(96.into())).as_u128() as f64
    }

    pub fn get_risk_free_rate(&self) -> f64 {
        self.risk_free_rate
    }

    pub fn set_risk_free_rate(&mut self, rate: f64) {
        self.risk_free_rate = rate;
    }

    pub fn get_volatility_adjustment(&self) -> f64 {
        self.volatility_adjustment
    }

    pub fn set_volatility_adjustment(&mut self, adjustment: f64) {
        self.volatility_adjustment = adjustment;
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Greeks {
    pub delta: f64,
    pub gamma: f64,
    pub theta: f64,
    pub vega: f64,
    pub rho: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Tick;
    use ethers::types::Address;

    fn create_test_pool() -> Pool {
        Pool::new(
            Address::zero(),
            Address::zero(),
            Address::zero(),
            3000,
            60,
            U256::from(2).pow(96.into()), // sqrt_price_x96 = 1.0
            U256::from(1000000),
            Tick(0),
        )
    }

    #[test]
    fn test_option_pricing() {
        let engine = PricingEngine::new(0.05, 1.0);
        let pool = create_test_pool();
        
        let price = engine.calculate_option_price(
            &pool,
            OptionType::Call,
            Strike(U256::from(1000)),
            1.0,
        ).unwrap();
        
        assert!(price > U256::zero());
    }

    #[test]
    fn test_greeks_calculation() {
        let engine = PricingEngine::new(0.05, 1.0);
        let pool = create_test_pool();
        
        let greeks = engine.calculate_greeks(
            &pool,
            OptionType::Call,
            Strike(U256::from(1000)),
            1.0,
        ).unwrap();
        
        assert!(greeks.delta >= 0.0 && greeks.delta <= 1.0);
        assert!(greeks.gamma >= 0.0);
        assert!(greeks.vega >= 0.0);
    }

    #[test]
    fn test_implied_volatility() {
        let engine = PricingEngine::new(0.05, 1.0);
        let pool = create_test_pool();
        
        let volatility = engine.calculate_implied_volatility(&pool).unwrap();
        assert!(volatility > 0.0 && volatility <= 2.0);
    }
}
