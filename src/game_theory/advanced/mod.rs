pub mod tick_manager;
pub mod fee_system;
pub mod liquidity_manager;
pub mod risk_manager;
pub mod oracle_manager;

use ethers::prelude::*;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Debug)]
pub struct GameTheoryManager {
    tick_manager: Arc<tick_manager::DynamicTickManager>,
    fee_system: Arc<fee_system::DynamicFeeSystem>,
    liquidity_manager: Arc<liquidity_manager::AdvancedLiquidityManager>,
    risk_manager: Arc<risk_manager::RiskManager>,
    oracle_manager: Arc<oracle_manager::AdvancedOracleManager>,
}

impl GameTheoryManager {
    pub fn new(
        tick_manager: Arc<tick_manager::DynamicTickManager>,
        fee_system: Arc<fee_system::DynamicFeeSystem>,
        liquidity_manager: Arc<liquidity_manager::AdvancedLiquidityManager>,
        risk_manager: Arc<risk_manager::RiskManager>,
        oracle_manager: Arc<oracle_manager::AdvancedOracleManager>,
    ) -> Self {
        Self {
            tick_manager,
            fee_system,
            liquidity_manager,
            risk_manager,
            oracle_manager,
        }
    }

    pub async fn optimize_parameters(
        &self,
        token0: Address,
        token1: Address,
    ) -> Result<OptimizedParameters, GameError> {
        // Get current market conditions
        let current_price = self.oracle_manager.calculate_twap(token0, token1, None).await?;
        let volatility = self.oracle_manager.calculate_recent_volatility(token0, token1).await?;
        
        // Optimize tick spacing
        let optimal_tick_spacing = self.tick_manager
            .optimize_tick_spacing(token0, token1, 86400)
            .await?;

        // Calculate dynamic fee
        let dynamic_fee = self.fee_system
            .calculate_dynamic_fee(token0, token1, U256::from(1000000), volatility)
            .await?;

        // Get optimal liquidity ranges
        let optimal_ranges = self.liquidity_manager
            .optimize_position_ranges(token0, token1, 0, current_price)
            .await?;

        // Check risk parameters
        let risk_config = self.risk_manager.config.clone();
        let circuit_breaker_active = self.risk_manager
            .check_circuit_breaker(token0, token1, current_price, U256::from(1000000))
            .await?;

        Ok(OptimizedParameters {
            tick_spacing: optimal_tick_spacing,
            fee_rate: dynamic_fee,
            liquidity_ranges: optimal_ranges,
            risk_config,
            circuit_breaker_active,
        })
    }

    pub async fn monitor_market_conditions(
        &self,
        token0: Address,
        token1: Address,
    ) -> Result<MarketConditions, GameError> {
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Get price and volatility metrics
        let current_price = self.oracle_manager
            .calculate_twap(token0, token1, None)
            .await?;
        
        let volatility = self.oracle_manager
            .calculate_recent_volatility(token0, token1)
            .await?;

        // Check for price manipulation
        let manipulation_detected = self.oracle_manager
            .detect_manipulation(token0, token1)
            .await?;

        // Get risk metrics
        let circuit_breaker_active = self.risk_manager
            .check_circuit_breaker(token0, token1, current_price, U256::from(1000000))
            .await?;

        // Detect potential MEV activity
        let mev_risk = self.risk_manager
            .protect_from_mev(risk_manager::Transaction {
                hash: H256::zero(),
                block_number: 0,
                gas_price: U256::zero(),
                method_id: [0; 4],
                timestamp: current_time,
            })
            .await?;

        Ok(MarketConditions {
            current_price,
            volatility,
            manipulation_detected,
            circuit_breaker_active,
            mev_risk,
            timestamp: current_time,
        })
    }

    pub async fn suggest_position_adjustments(
        &self,
        position_id: U256,
        current_price: U256,
    ) -> Result<PositionAdjustments, GameError> {
        // Get current market conditions
        let volatility = self.oracle_manager
            .calculate_recent_volatility(
                Address::zero(),  // Replace with actual token addresses
                Address::zero(),
            )
            .await?;

        // Calculate position risk
        let risk_assessment = self.risk_manager
            .calculate_position_risk(
                position_id,
                current_price,
                volatility,
                U256::from(1000000),  // Replace with actual collateral
            )
            .await?;

        // Check if rebalance is needed
        let needs_rebalance = self.liquidity_manager
            .calculate_rebalance_need(position_id, 0)  // Replace with current tick
            .await?;

        // Calculate IL protection
        let il_protection = self.liquidity_manager
            .calculate_impermanent_loss(
                position_id,
                current_price,
                current_price,  // Replace with token1 price
            )
            .await?;

        Ok(PositionAdjustments {
            risk_assessment,
            needs_rebalance,
            il_protection,
            suggested_actions: vec![
                "Monitor position health".to_string(),
                "Consider rebalancing".to_string(),
            ],
        })
    }
}

#[derive(Debug, Clone)]
pub struct OptimizedParameters {
    pub tick_spacing: i32,
    pub fee_rate: u32,
    pub liquidity_ranges: Vec<(i32, i32)>,
    pub risk_config: risk_manager::RiskConfig,
    pub circuit_breaker_active: bool,
}

#[derive(Debug, Clone)]
pub struct MarketConditions {
    pub current_price: U256,
    pub volatility: f64,
    pub manipulation_detected: bool,
    pub circuit_breaker_active: bool,
    pub mev_risk: bool,
    pub timestamp: u64,
}

#[derive(Debug, Clone)]
pub struct PositionAdjustments {
    pub risk_assessment: risk_manager::PositionRisk,
    pub needs_rebalance: bool,
    pub il_protection: U256,
    pub suggested_actions: Vec<String>,
}

#[derive(Debug)]
pub enum GameError {
    InsufficientData,
    InvalidFeeShare,
    StrategyNotFound,
    PositionNotFound,
    MetricsNotFound,
    NoILClaims,
}
