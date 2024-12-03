use crate::types::{Pool, Result, PanopticError, OptionType};
use crate::pricing::models::OptionPricing;
use crate::risk::manager::RiskManager;
use ethers::types::{U256, Address};
use std::collections::HashMap;
use std::sync::Arc;

pub trait Strategy: Send + Sync {
    fn calculate_quotes(&self, pool: &Pool, timestamp: u64) -> Result<Vec<Quote>>;
    fn should_rebalance(&self, pool: &Pool, timestamp: u64) -> Result<bool>;
    fn get_rebalance_action(&self, pool: &Pool) -> Result<RebalanceAction>;
}

pub struct DeltaNeutralStrategy {
    pricing: Arc<dyn OptionPricing>,
    risk_manager: Arc<RiskManager>,
    target_delta: f64,
    delta_threshold: f64,
    rebalance_interval: u64,
    last_rebalances: HashMap<Address, u64>,
}

pub struct VolatilityArbitrageStrategy {
    pricing: Arc<dyn OptionPricing>,
    risk_manager: Arc<RiskManager>,
    min_vol_spread: f64,
    max_position_value: U256,
    min_profit_threshold: f64,
    positions: HashMap<Address, StrategyPosition>,
}

pub struct GammaScalpingStrategy {
    pricing: Arc<dyn OptionPricing>,
    risk_manager: Arc<RiskManager>,
    target_gamma: f64,
    hedge_interval: u64,
    min_edge: f64,
    positions: HashMap<Address, StrategyPosition>,
}

impl DeltaNeutralStrategy {
    pub fn new(
        pricing: Arc<dyn OptionPricing>,
        risk_manager: Arc<RiskManager>,
        target_delta: f64,
        delta_threshold: f64,
        rebalance_interval: u64,
    ) -> Self {
        Self {
            pricing,
            risk_manager,
            target_delta,
            delta_threshold,
            rebalance_interval,
            last_rebalances: HashMap::new(),
        }
    }
}

impl Strategy for DeltaNeutralStrategy {
    fn calculate_quotes(&self, pool: &Pool, timestamp: u64) -> Result<Vec<Quote>> {
        let mut quotes = Vec::new();
        let portfolio_delta = self.calculate_portfolio_delta(pool)?;
        let delta_deviation = (portfolio_delta - self.target_delta).abs();

        // Adjust quotes based on delta deviation
        let base_spread = 0.02; // 2% base spread
        let adjusted_spread = base_spread * (1.0 + delta_deviation);

        for option_type in [OptionType::Call, OptionType::Put] {
            let base_price = self.pricing.calculate_option_price(
                pool,
                option_type,
                self.pricing.get_implied_volatility(pool)?,
            )?;

            let (bid_price, ask_price) = if portfolio_delta > self.target_delta {
                // Short biased quotes
                let bid = base_price * U256::from((1.0 - adjusted_spread * 1.5) as u128);
                let ask = base_price * U256::from((1.0 + adjusted_spread) as u128);
                (bid, ask)
            } else {
                // Long biased quotes
                let bid = base_price * U256::from((1.0 - adjusted_spread) as u128);
                let ask = base_price * U256::from((1.0 + adjusted_spread * 1.5) as u128);
                (bid, ask)
            };

            let position_limit = self.risk_manager.calculate_max_position_size(
                pool,
                option_type,
                base_price,
            )?;

            quotes.push(Quote {
                option_type,
                bid_price,
                ask_price,
                size: position_limit,
                timestamp,
            });
        }

        Ok(quotes)
    }

    fn should_rebalance(&self, pool: &Pool, timestamp: u64) -> Result<bool> {
        let last_rebalance = self.last_rebalances.get(&pool.address).copied().unwrap_or(0);
        if timestamp - last_rebalance < self.rebalance_interval {
            return Ok(false);
        }

        let portfolio_delta = self.calculate_portfolio_delta(pool)?;
        Ok((portfolio_delta - self.target_delta).abs() > self.delta_threshold)
    }

    fn get_rebalance_action(&self, pool: &Pool) -> Result<RebalanceAction> {
        let portfolio_delta = self.calculate_portfolio_delta(pool)?;
        let delta_deviation = portfolio_delta - self.target_delta;

        if delta_deviation.abs() <= self.delta_threshold {
            return Ok(RebalanceAction::None);
        }

        let size = U256::from((delta_deviation.abs() * 1e18) as u128);
        let option_type = if delta_deviation > 0.0 {
            OptionType::Put
        } else {
            OptionType::Call
        };

        Ok(RebalanceAction::Trade {
            option_type,
            size,
            is_buy: delta_deviation < 0.0,
        })
    }
}

impl VolatilityArbitrageStrategy {
    pub fn new(
        pricing: Arc<dyn OptionPricing>,
        risk_manager: Arc<RiskManager>,
        min_vol_spread: f64,
        max_position_value: U256,
        min_profit_threshold: f64,
    ) -> Self {
        Self {
            pricing,
            risk_manager,
            min_vol_spread,
            max_position_value,
            min_profit_threshold,
            positions: HashMap::new(),
        }
    }

    fn find_arbitrage_opportunities(&self, pool: &Pool) -> Result<Vec<ArbitrageOpportunity>> {
        let mut opportunities = Vec::new();
        let market_vol = self.pricing.get_implied_volatility(pool)?;
        let theoretical_vol = self.pricing.calculate_theoretical_volatility(pool)?;

        let vol_spread = (market_vol - theoretical_vol).abs();
        if vol_spread < self.min_vol_spread {
            return Ok(opportunities);
        }

        for option_type in [OptionType::Call, OptionType::Put] {
            let market_price = self.pricing.calculate_option_price(
                pool,
                option_type,
                market_vol,
            )?;

            let theoretical_price = self.pricing.calculate_option_price(
                pool,
                option_type,
                theoretical_vol,
            )?;

            let price_diff = if market_price > theoretical_price {
                (market_price - theoretical_price).as_u128() as f64
            } else {
                (theoretical_price - market_price).as_u128() as f64
            };

            let profit_potential = price_diff / market_price.as_u128() as f64;
            if profit_potential >= self.min_profit_threshold {
                opportunities.push(ArbitrageOpportunity {
                    option_type,
                    market_price,
                    theoretical_price,
                    profit_potential,
                    is_overvalued: market_price > theoretical_price,
                });
            }
        }

        Ok(opportunities)
    }
}

impl GammaScalpingStrategy {
    pub fn new(
        pricing: Arc<dyn OptionPricing>,
        risk_manager: Arc<RiskManager>,
        target_gamma: f64,
        hedge_interval: u64,
        min_edge: f64,
    ) -> Self {
        Self {
            pricing,
            risk_manager,
            target_gamma,
            hedge_interval,
            min_edge,
            positions: HashMap::new(),
        }
    }

    fn calculate_hedge_ratio(&self, pool: &Pool) -> Result<f64> {
        let position = self.positions.get(&pool.address)
            .ok_or(PanopticError::PositionNotFound)?;

        let spot_price = self.pricing.get_spot_price(pool)?;
        let gamma = position.gamma;
        let delta = position.delta;

        let hedge_ratio = -delta / (2.0 * gamma * spot_price.as_u128() as f64);
        Ok(hedge_ratio)
    }
}

#[derive(Debug)]
pub struct Quote {
    pub option_type: OptionType,
    pub bid_price: U256,
    pub ask_price: U256,
    pub size: U256,
    pub timestamp: u64,
}

#[derive(Debug)]
pub struct RebalanceAction {
    pub option_type: OptionType,
    pub size: U256,
    pub is_buy: bool,
}

#[derive(Debug)]
pub struct ArbitrageOpportunity {
    pub option_type: OptionType,
    pub market_price: U256,
    pub theoretical_price: U256,
    pub profit_potential: f64,
    pub is_overvalued: bool,
}

#[derive(Debug)]
pub struct StrategyPosition {
    pub size: U256,
    pub entry_price: U256,
    pub delta: f64,
    pub gamma: f64,
    pub theta: f64,
    pub vega: f64,
}
