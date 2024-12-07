use crate::types::{Pool, Position, OptionType};
use crate::pricing::{PricingEngine, Greeks};
use crate::errors::{PanopticError, Result};
use ethers::types::U256;
use std::collections::{HashMap, BTreeMap};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct RiskParameters {
    pub max_leverage: f64,
    pub min_collateral_ratio: f64,
    pub liquidation_threshold: f64,
    pub max_position_size: U256,
    pub max_portfolio_notional: U256,
    pub max_portfolio_delta: f64,
    pub max_portfolio_gamma: f64,
    pub max_portfolio_vega: f64,
    pub margin_buffer: f64,
    pub stress_test_scenarios: Vec<StressScenario>,
}

#[derive(Debug, Clone)]
pub struct StressScenario {
    pub price_change: f64,
    pub volatility_change: f64,
    pub interest_rate_change: f64,
    pub time_decay: f64,
}

#[derive(Debug, Clone)]
pub struct RiskMetrics {
    pub value_at_risk: f64,
    pub expected_shortfall: f64,
    pub stress_test_results: Vec<f64>,
    pub portfolio_delta: f64,
    pub portfolio_gamma: f64,
    pub portfolio_vega: f64,
    pub portfolio_theta: f64,
    pub concentration_score: f64,
    pub liquidity_score: f64,
}

#[derive(Debug, Clone)]
pub struct PortfolioMetrics {
    pub total_notional: U256,
    pub total_collateral: U256,
    pub net_delta: f64,
    pub net_gamma: f64,
    pub net_vega: f64,
    pub net_theta: f64,
    pub margin_ratio: f64,
    pub leverage_ratio: f64,
    pub liquidation_price: Option<U256>,
}

pub struct RiskManager {
    parameters: RiskParameters,
    pricing_engine: PricingEngine,
    position_metrics: HashMap<U256, PositionRiskMetrics>,
    pool_metrics: HashMap<Pool, PoolRiskMetrics>,
    portfolio_positions: HashMap<U256, Vec<Position>>,
}

#[derive(Debug, Clone)]
struct PositionRiskMetrics {
    var: f64,
    es: f64,
    greeks: Greeks,
    stress_results: Vec<f64>,
}

#[derive(Debug, Clone)]
struct PoolRiskMetrics {
    utilization: f64,
    concentration: f64,
    volatility: f64,
    liquidity_depth: f64,
}

impl RiskManager {
    pub fn new(parameters: RiskParameters, pricing_engine: PricingEngine) -> Self {
        Self {
            parameters,
            pricing_engine,
            position_metrics: HashMap::new(),
            pool_metrics: HashMap::new(),
            portfolio_positions: HashMap::new(),
        }
    }

    pub fn validate_position(
        &self,
        pool: &Pool,
        position: &Position,
        current_block: U256,
    ) -> Result<()> {
        // Check position size
        if position.amount > self.parameters.max_position_size {
            return Err(PanopticError::Custom("Position size exceeds limit".into()));
        }

        // Check leverage
        let leverage = self.calculate_position_leverage(pool, position)?;
        if leverage > self.parameters.max_leverage {
            return Err(PanopticError::Custom("Leverage exceeds limit".into()));
        }

        // Check concentration
        let concentration = self.calculate_concentration(pool, position)?;
        if concentration > self.parameters.concentration_limit {
            return Err(PanopticError::Custom("Concentration exceeds limit".into()));
        }

        // Validate through stress tests
        self.stress_test_position(pool, position)?;

        Ok(())
    }

    pub fn calculate_portfolio_risk(
        &self,
        pool: &Pool,
        positions: &[Position],
        current_block: U256,
    ) -> Result<RiskMetrics> {
        let mut portfolio_value = 0.0;
        let mut portfolio_delta = 0.0;
        let mut portfolio_gamma = 0.0;
        let mut portfolio_vega = 0.0;
        let mut portfolio_theta = 0.0;
        let mut returns = Vec::new();

        // Calculate portfolio-wide metrics
        for position in positions {
            let time_to_expiry = (position.expiry - current_block).as_u64() as f64 / 31_536_000.0;
            let greeks = self.pricing_engine.calculate_greeks(
                pool,
                position.option_type,
                position.strike,
                time_to_expiry,
            )?;

            let position_value = self.pricing_engine.calculate_option_price(
                pool,
                position.option_type,
                position.strike,
                time_to_expiry,
            )?.as_u128() as f64;

            portfolio_value += position_value;
            portfolio_delta += greeks.delta * position_value;
            portfolio_gamma += greeks.gamma * position_value;
            portfolio_vega += greeks.vega * position_value;
            portfolio_theta += greeks.theta * position_value;

            // Historical returns simulation for VaR
            let daily_returns = self.simulate_returns(pool, position, 252)?;
            returns.extend(daily_returns);
        }

        // Calculate VaR and ES
        returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let var_index = (returns.len() as f64 * 0.01) as usize;
        let value_at_risk = -returns[var_index];

        let expected_shortfall = -returns[..var_index]
            .iter()
            .sum::<f64>() / var_index as f64;

        // Stress testing
        let stress_test_results = self.stress_test_portfolio(pool, positions)?;

        // Calculate concentration and liquidity scores
        let concentration_score = self.calculate_portfolio_concentration(pool, positions)?;
        let liquidity_score = self.calculate_portfolio_liquidity(pool, positions)?;

        Ok(RiskMetrics {
            value_at_risk,
            expected_shortfall,
            stress_test_results,
            portfolio_delta,
            portfolio_gamma,
            portfolio_vega,
            portfolio_theta,
            concentration_score,
            liquidity_score,
        })
    }

    pub fn calculate_portfolio_metrics(
        &self,
        account: U256,
        positions: &[Position],
    ) -> Result<PortfolioMetrics> {
        let mut total_notional = U256::zero();
        let mut total_collateral = U256::zero();
        let mut net_delta = 0.0;
        let mut net_gamma = 0.0;
        let mut net_vega = 0.0;
        let mut net_theta = 0.0;

        // Add existing positions
        if let Some(positions) = self.portfolio_positions.get(&account) {
            for position in positions {
                let metrics = self.position_metrics.get(&position.pool_address)
                    .ok_or(PanopticError::MetricsNotFound)?;

                total_notional = total_notional.saturating_add(position.amount);
                total_collateral = total_collateral.saturating_add(position.collateral);
                
                let size_f64 = position.amount.as_u128() as f64;
                net_delta += metrics.delta * size_f64;
                net_gamma += metrics.gamma * size_f64;
                net_vega += metrics.vega * size_f64;
                net_theta += metrics.theta * size_f64;
            }
        }

        // Add new positions
        for position in positions {
            total_notional = total_notional.saturating_add(position.amount);
            total_collateral = total_collateral.saturating_add(position.collateral);
            
            let size_f64 = position.amount.as_u128() as f64;
            // Note: This assumes we have Greeks for the new position
            if let Some(metrics) = self.position_metrics.get(&position.pool_address) {
                net_delta += metrics.delta * size_f64;
                net_gamma += metrics.gamma * size_f64;
                net_vega += metrics.vega * size_f64;
                net_theta += metrics.theta * size_f64;
            }
        }

        // Calculate ratios
        let margin_ratio = if total_notional == U256::zero() {
            1.0
        } else {
            total_collateral.as_u128() as f64 / total_notional.as_u128() as f64
        };

        let leverage_ratio = if total_collateral == U256::zero() {
            0.0
        } else {
            total_notional.as_u128() as f64 / total_collateral.as_u128() as f64
        };

        // Calculate liquidation price if applicable
        let liquidation_price = self.calculate_liquidation_price(
            account,
            total_collateral,
            net_delta,
            net_gamma,
        )?;

        Ok(PortfolioMetrics {
            total_notional,
            total_collateral,
            net_delta,
            net_gamma,
            net_vega,
            net_theta,
            margin_ratio,
            leverage_ratio,
            liquidation_price,
        })
    }

    pub fn needs_liquidation(
        &self,
        account: U256,
        current_price: U256,
    ) -> Result<bool> {
        let metrics = self.calculate_portfolio_metrics(account, &[])?;

        // Check if margin ratio is below liquidation threshold
        if metrics.margin_ratio < self.parameters.liquidation_threshold {
            return Ok(true);
        }

        // Check if current price is beyond liquidation price
        if let Some(liq_price) = metrics.liquidation_price {
            if current_price >= liq_price {
                return Ok(true);
            }
        }

        Ok(false)
    }

    pub fn calculate_liquidation_price(
        &self,
        account: U256,
        total_collateral: U256,
        net_delta: f64,
        net_gamma: f64,
    ) -> Result<Option<U256>> {
        if net_delta == 0.0 && net_gamma == 0.0 {
            return Ok(None);
        }

        // Simplified quadratic formula for liquidation price
        // Assumes: Collateral = Value + Delta * dP + 0.5 * Gamma * dP^2
        let a = 0.5 * net_gamma;
        let b = net_delta;
        let c = -(total_collateral.as_u128() as f64 * self.parameters.liquidation_threshold);

        // Solve quadratic equation
        let discriminant = b * b - 4.0 * a * c;
        if discriminant < 0.0 {
            return Ok(None);
        }

        let price = (-b + discriminant.sqrt()) / (2.0 * a);
        Ok(Some(U256::from(price as u128)))
    }

    pub fn update_position_metrics(
        &mut self,
        pool_address: U256,
        greeks: Greeks,
    ) {
        self.position_metrics.insert(pool_address, greeks);
    }

    pub fn add_position(
        &mut self,
        account: U256,
        position: Position,
    ) {
        self.portfolio_positions
            .entry(account)
            .or_insert_with(Vec::new)
            .push(position);
    }

    pub fn remove_position(
        &mut self,
        account: U256,
        pool_address: U256,
    ) -> Result<()> {
        if let Some(positions) = self.portfolio_positions.get_mut(&account) {
            positions.retain(|p| p.pool_address != pool_address);
            Ok(())
        } else {
            Err(PanopticError::PositionNotFound)
        }
    }

    fn calculate_position_leverage(&self, pool: &Pool, position: &Position) -> Result<f64> {
        let notional_value = self.calculate_notional_value(pool, position)?;
        let collateral = position.collateral.as_u128() as f64;
        
        if collateral == 0.0 {
            return Ok(f64::INFINITY);
        }
        
        Ok(notional_value / collateral)
    }

    fn calculate_concentration(&self, pool: &Pool, position: &Position) -> Result<f64> {
        let position_notional = self.calculate_notional_value(pool, position)?;
        let pool_total_notional = pool.total_notional().as_u128() as f64;
        
        if pool_total_notional == 0.0 {
            return Ok(1.0);
        }
        
        Ok(position_notional / pool_total_notional)
    }

    fn stress_test_position(&self, pool: &Pool, position: &Position) -> Result<Vec<f64>> {
        let mut results = Vec::new();
        
        for scenario in &self.parameters.stress_test_scenarios {
            // Calculate stressed price
            let base_price = pool.current_price()?;
            let stressed_price = base_price.as_u128() as f64 * (1.0 + scenario.price_change);
            
            // Calculate stressed volatility
            let base_vol = self.pricing_engine.get_implied_volatility(pool)?;
            let stressed_vol = base_vol * (1.0 + scenario.volatility_change);
            
            // Calculate stressed option price
            let stressed_value = self.pricing_engine.calculate_option_price_with_params(
                pool,
                position.option_type,
                position.strike,
                position.time_to_expiry()?,
                U256::from_f64_lossy(stressed_price),
                stressed_vol,
                self.pricing_engine.risk_free_rate + scenario.interest_rate_change,
            )?;
            
            // Calculate PnL under stress
            let base_value = self.pricing_engine.calculate_option_price(
                pool,
                position.option_type,
                position.strike,
                position.time_to_expiry()?,
            )?;
            
            let pnl = (stressed_value.as_u128() as f64 - base_value.as_u128() as f64) 
                     * position.amount.as_u128() as f64;
            
            results.push(pnl);
        }
        
        Ok(results)
    }

    fn simulate_returns(&self, pool: &Pool, position: &Position, days: usize) -> Result<Vec<f64>> {
        let mut returns = Vec::with_capacity(days);
        let mut current_price = pool.current_price()?.as_u128() as f64;
        let volatility = self.pricing_engine.get_implied_volatility(pool)?;
        let dt = 1.0 / 252.0; // Daily timestep
        
        let normal = statrs::distribution::Normal::new(0.0, 1.0)?;
        let mut rng = rand::thread_rng();
        
        for _ in 0..days {
            // Simulate price movement using geometric Brownian motion
            let z: f64 = normal.sample(&mut rng);
            let drift = (self.pricing_engine.risk_free_rate - 0.5 * volatility * volatility) * dt;
            let diffusion = volatility * z.sqrt() * dt.sqrt();
            
            let price_return = (drift + diffusion).exp();
            let new_price = current_price * price_return;
            
            // Calculate option price at new market price
            let new_option_price = self.pricing_engine.calculate_option_price_with_params(
                pool,
                position.option_type,
                position.strike,
                position.time_to_expiry()?,
                U256::from_f64_lossy(new_price),
                volatility,
                self.pricing_engine.risk_free_rate,
            )?;
            
            let old_option_price = self.pricing_engine.calculate_option_price_with_params(
                pool,
                position.option_type,
                position.strike,
                position.time_to_expiry()?,
                U256::from_f64_lossy(current_price),
                volatility,
                self.pricing_engine.risk_free_rate,
            )?;
            
            let return_pct = (new_option_price.as_u128() as f64 - old_option_price.as_u128() as f64)
                            / old_option_price.as_u128() as f64;
            
            returns.push(return_pct);
            current_price = new_price;
        }
        
        Ok(returns)
    }

    fn calculate_portfolio_concentration(
        &self,
        pool: &Pool,
        positions: &[Position],
    ) -> Result<f64> {
        let mut position_weights = Vec::new();
        let mut total_notional = 0.0;
        
        // Calculate notional values and total
        for position in positions {
            let notional = self.calculate_notional_value(pool, position)?;
            position_weights.push(notional);
            total_notional += notional;
        }
        
        if total_notional == 0.0 {
            return Ok(0.0);
        }
        
        // Calculate Herfindahl-Hirschman Index (HHI)
        let hhi = position_weights.iter()
            .map(|&w| (w / total_notional).powi(2))
            .sum::<f64>();
        
        // Normalize HHI to [0,1]
        let n = positions.len() as f64;
        let normalized_hhi = if n <= 1.0 {
            1.0
        } else {
            (hhi - 1.0/n) / (1.0 - 1.0/n)
        };
        
        Ok(normalized_hhi)
    }

    fn calculate_portfolio_liquidity(
        &self,
        pool: &Pool,
        positions: &[Position],
    ) -> Result<f64> {
        let mut weighted_liquidity = 0.0;
        let mut total_weight = 0.0;
        
        for position in positions {
            let notional = self.calculate_notional_value(pool, position)?;
            let position_liquidity = self.calculate_position_liquidity(pool, position)?;
            
            weighted_liquidity += notional * position_liquidity;
            total_weight += notional;
        }
        
        if total_weight == 0.0 {
            return Ok(1.0);
        }
        
        Ok(weighted_liquidity / total_weight)
    }

    fn calculate_position_liquidity(&self, pool: &Pool, position: &Position) -> Result<f64> {
        // Get market depth around strike
        let depth = pool.get_depth_at_price(position.strike)?;
        let position_size = position.amount.as_u128() as f64;
        
        // Calculate liquidity score based on position size relative to market depth
        let liquidity_ratio = depth / position_size;
        
        // Apply exponential decay function to normalize score to [0,1]
        Ok(1.0 - (-liquidity_ratio / 2.0).exp())
    }

    fn calculate_notional_value(&self, pool: &Pool, position: &Position) -> Result<f64> {
        let price = self.pricing_engine.calculate_option_price(
            pool,
            position.option_type,
            position.strike,
            position.time_to_expiry()?,
        )?;
        
        Ok(price.as_u128() as f64 * position.amount.as_u128() as f64)
    }
}
