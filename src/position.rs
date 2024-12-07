use crate::types::{OptionType, Pool, Position, PositionKey, Strike, TokenId};
use crate::errors::{PanopticError, Result};
use crate::pricing::PricingEngine;
use crate::collateral::CollateralManager;
use ethers::types::{Address, U256};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug)]
pub struct PositionManager {
    positions: HashMap<PositionKey, Position>,
    greeks: HashMap<PositionKey, Greeks>,
    risk_metrics: HashMap<Address, PortfolioRisk>,
    position_limits: PositionLimits,
    pricing_engine: PricingEngine,
    collateral_manager: CollateralManager,
}

#[derive(Debug, Clone)]
struct PortfolioRisk {
    total_delta: f64,
    total_gamma: f64,
    total_vega: f64,
    total_theta: f64,
    margin_used: U256,
    margin_available: U256,
    risk_score: f64,
}

#[derive(Debug)]
struct PositionLimits {
    max_total_risk: f64,
    max_position_size: U256,
    min_collateral_ratio: f64,
    max_portfolio_vega: f64,
    max_portfolio_gamma: f64,
}

impl PositionManager {
    pub fn new(
        pricing_engine: PricingEngine,
        collateral_manager: CollateralManager,
        max_total_risk: f64,
        max_position_size: U256,
        min_collateral_ratio: f64,
        max_portfolio_vega: f64,
        max_portfolio_gamma: f64,
    ) -> Self {
        Self {
            positions: HashMap::new(),
            greeks: HashMap::new(),
            risk_metrics: HashMap::new(),
            position_limits: PositionLimits {
                max_total_risk,
                max_position_size,
                min_collateral_ratio,
                max_portfolio_vega,
                max_portfolio_gamma,
            },
            pricing_engine,
            collateral_manager,
        }
    }

    pub fn create_position(
        &mut self,
        pool: &Pool,
        owner: Address,
        option_type: OptionType,
        strike: Strike,
        amount: U256,
        collateral: U256,
        expiry: U256,
    ) -> Result<TokenId> {
        let key = PositionKey { owner, token_id: TokenId(U256::zero()) };

        // Validate position size
        if amount > self.position_limits.max_position_size {
            return Err(PanopticError::ExcessivePositionSize);
        }

        // Calculate required collateral
        let required_collateral = self.calculate_required_collateral(
            option_type,
            strike,
            amount,
            pool.sqrt_price_x96.pow(2.into()) / U256::from(2).pow(96.into()),
            self.pricing_engine.get_volatility(pool, option_type, strike, expiry)?,
        )?;

        if collateral < required_collateral {
            return Err(PanopticError::InsufficientCollateral);
        }

        // Calculate position Greeks
        let greeks = self.calculate_greeks(
            option_type,
            strike,
            amount,
            pool.sqrt_price_x96.pow(2.into()) / U256::from(2).pow(96.into()),
            self.pricing_engine.get_volatility(pool, option_type, strike, expiry)?,
            expiry,
        )?;

        // Check portfolio risk limits
        let portfolio = self.risk_metrics.entry(owner).or_insert(PortfolioRisk {
            total_delta: 0.0,
            total_gamma: 0.0,
            total_vega: 0.0,
            total_theta: 0.0,
            margin_used: U256::zero(),
            margin_available: U256::zero(),
            risk_score: 0.0,
        });

        let new_gamma = portfolio.total_gamma + greeks.gamma;
        let new_vega = portfolio.total_vega + greeks.vega;

        if new_gamma.abs() > self.position_limits.max_portfolio_gamma ||
           new_vega.abs() > self.position_limits.max_portfolio_vega {
            return Err(PanopticError::PortfolioRiskLimitExceeded);
        }

        // Update portfolio metrics
        portfolio.total_delta += greeks.delta;
        portfolio.total_gamma = new_gamma;
        portfolio.total_vega = new_vega;
        portfolio.total_theta += greeks.theta;
        portfolio.margin_used += required_collateral;
        portfolio.risk_score = self.calculate_risk_score(portfolio);

        // Generate token ID
        let token_id = TokenId(U256::from(self.positions.len()));

        // Create and store position
        let position = Position::new(
            owner,
            token_id,
            option_type,
            strike,
            amount,
            collateral,
            expiry,
        );

        self.positions.insert(PositionKey { owner, token_id }, position);
        self.greeks.insert(PositionKey { owner, token_id }, greeks);

        Ok(token_id)
    }

    pub fn close_position(
        &mut self,
        owner: Address,
        token_id: TokenId,
        amount: U256,
    ) -> Result<()> {
        let key = PositionKey { owner, token_id };

        let position = self.positions.get_mut(&key)
            .ok_or(PanopticError::PositionNotFound)?;

        if amount > position.amount {
            return Err(PanopticError::InsufficientPositionSize);
        }

        let greeks = self.greeks.get(&key)
            .ok_or(PanopticError::GreeksNotFound)?;

        // Update portfolio risk metrics
        if let Some(portfolio) = self.risk_metrics.get_mut(&position.owner) {
            let ratio = amount.as_u128() as f64 / position.amount.as_u128() as f64;
            
            portfolio.total_delta -= greeks.delta * ratio;
            portfolio.total_gamma -= greeks.gamma * ratio;
            portfolio.total_vega -= greeks.vega * ratio;
            portfolio.total_theta -= greeks.theta * ratio;
            
            let released_collateral = position.collateral * amount / position.amount;
            portfolio.margin_used -= released_collateral;
            portfolio.risk_score = self.calculate_risk_score(portfolio);
        }

        // Update or remove position
        if amount == position.amount {
            self.positions.remove(&key);
            self.greeks.remove(&key);
        } else {
            position.amount -= amount;
            position.collateral -= position.collateral * amount / position.amount;
        }

        Ok(())
    }

    fn calculate_required_collateral(
        &self,
        option_type: OptionType,
        strike: Strike,
        amount: U256,
        spot_price: U256,
        volatility: f64,
    ) -> Result<U256> {
        let strike_price = U256::from(strike.0);
        
        match option_type {
            OptionType::Call => {
                // For calls, required collateral is max(spot_price - strike_price, 0) * amount
                if spot_price > strike_price {
                    Ok((spot_price - strike_price) * amount)
                } else {
                    Ok(U256::zero())
                }
            },
            OptionType::Put => {
                // For puts, required collateral is strike_price * amount
                Ok(strike_price * amount)
            }
        }
    }

    fn calculate_greeks(
        &self,
        option_type: OptionType,
        strike: Strike,
        amount: U256,
        spot_price: U256,
        volatility: f64,
        expiry: U256,
    ) -> Result<Greeks> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let time_to_expiry = (expiry.as_u64().saturating_sub(now)) as f64 / (365.0 * 24.0 * 3600.0);
        let spot = spot_price.as_u128() as f64;
        let strike = strike.0.as_u128() as f64;
        let size = amount.as_u128() as f64;

        // Black-Scholes Greeks calculation
        let d1 = (f64::ln(spot / strike) + (0.05 + volatility * volatility / 2.0) * time_to_expiry) 
            / (volatility * f64::sqrt(time_to_expiry));
        let d2 = d1 - volatility * f64::sqrt(time_to_expiry);

        let sign = match option_type {
            OptionType::Call => 1.0,
            OptionType::Put => -1.0,
        };

        Ok(Greeks {
            delta: sign * standard_normal_cdf(d1) * size,
            gamma: standard_normal_pdf(d1) / (spot * volatility * f64::sqrt(time_to_expiry)) * size,
            vega: spot * f64::sqrt(time_to_expiry) * standard_normal_pdf(d1) * size / 100.0,
            theta: (-spot * volatility * standard_normal_pdf(d1) / (2.0 * f64::sqrt(time_to_expiry)) 
                   - sign * 0.05 * strike * f64::exp(-0.05 * time_to_expiry) * standard_normal_cdf(sign * d2)) 
                   * size / 365.0,
        })
    }

    fn calculate_risk_score(&self, portfolio: &PortfolioRisk) -> f64 {
        // Weighted risk score based on Greeks and utilization
        let utilization = portfolio.margin_used.as_u128() as f64 / 
            (portfolio.margin_available.as_u128() as f64 + 1.0);
        
        0.4 * portfolio.total_delta.abs() / 100.0 +
        0.2 * portfolio.total_gamma.abs() +
        0.2 * portfolio.total_vega.abs() / 100.0 +
        0.1 * portfolio.total_theta.abs() +
        0.1 * utilization
    }
}

fn standard_normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + libm::erf(x / f64::sqrt(2.0)))
}

fn standard_normal_pdf(x: f64) -> f64 {
    f64::exp(-0.5 * x * x) / f64::sqrt(2.0 * std::f64::consts::PI)
}
