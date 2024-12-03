use crate::types::{Pool, Result, PanopticError};
use ethers::types::{U256, Address, H256, Transaction};
use std::collections::{HashMap, BTreeMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;

pub struct ExecutionEngine {
    orders: RwLock<OrderBook>,
    strategies: Vec<Arc<dyn ExecutionStrategy>>,
    router: Arc<dyn ExecutionRouter>,
    risk_manager: Arc<dyn RiskManager>,
    state: RwLock<ExecutionState>,
    config: ExecutionConfig,
}

#[async_trait::async_trait]
pub trait ExecutionStrategy: Send + Sync {
    fn get_name(&self) -> &str;
    async fn execute_order(&self, order: &Order) -> Result<ExecutionPlan>;
    fn estimate_cost(&self, plan: &ExecutionPlan) -> Result<ExecutionCost>;
}

#[async_trait::async_trait]
pub trait ExecutionRouter: Send + Sync {
    async fn route_order(&self, order: &Order) -> Result<Vec<SubOrder>>;
    async fn execute_sub_order(&self, sub_order: &SubOrder) -> Result<H256>;
    fn estimate_gas(&self, sub_order: &SubOrder) -> Result<U256>;
}

#[async_trait::async_trait]
pub trait RiskManager: Send + Sync {
    fn validate_order(&self, order: &Order) -> Result<bool>;
    fn check_limits(&self, account: Address) -> Result<AccountLimits>;
    fn update_risk_metrics(&self, execution: &ExecutionResult) -> Result<()>;
    fn get_collateral(&self, account: Address) -> Result<U256>;
    fn get_balance(&self, account: Address) -> Result<U256>;
    fn get_token_balance(&self, account: Address, token: Address) -> Result<U256>;
}

#[derive(Clone, Debug)]
pub struct Order {
    pub id: H256,
    pub trader: Address,
    pub side: OrderSide,
    pub order_type: OrderType,
    pub asset: Address,
    pub amount: U256,
    pub price: Option<U256>,
    pub stop_price: Option<U256>,
    pub time_in_force: TimeInForce,
    pub execution_options: ExecutionOptions,
    pub created_at: u64,
}

#[derive(Clone, Debug)]
pub struct SubOrder {
    pub parent_id: H256,
    pub venue: Address,
    pub amount: U256,
    pub price: U256,
    pub route: Vec<Address>,
    pub deadline: u64,
}

#[derive(Clone, Debug)]
pub struct ExecutionPlan {
    pub order_id: H256,
    pub sub_orders: Vec<SubOrder>,
    pub estimated_cost: ExecutionCost,
    pub execution_path: Vec<ExecutionStep>,
}

#[derive(Clone, Debug)]
pub struct ExecutionStep {
    pub venue: Address,
    pub action: ExecutionAction,
    pub params: Vec<String>,
    pub dependencies: Vec<usize>,
}

#[derive(Clone, Debug)]
pub enum ExecutionAction {
    Swap(SwapParams),
    Provide(ProvideParams),
    Remove(RemoveParams),
    Flash(FlashParams),
    Custom(String, Vec<String>),
}

#[derive(Clone, Debug)]
pub struct SwapParams {
    pub token_in: Address,
    pub token_out: Address,
    pub amount_in: U256,
    pub min_amount_out: U256,
}

#[derive(Clone, Debug)]
pub struct ProvideParams {
    pub pool: Address,
    pub amounts: Vec<U256>,
    pub min_lp: U256,
}

#[derive(Clone, Debug)]
pub struct RemoveParams {
    pub pool: Address,
    pub lp_amount: U256,
    pub min_amounts: Vec<U256>,
    pub recipient: Address,
}

#[derive(Clone, Debug)]
pub struct FlashParams {
    pub token: Address,
    pub amount: U256,
    pub callback_data: Vec<u8>,
    pub callback_target: Address,
}

#[derive(Clone, Debug)]
pub struct ExecutionCost {
    pub gas_estimate: U256,
    pub max_priority_fee: U256,
    pub max_fee: U256,
    pub value: U256,
    pub token_costs: HashMap<Address, U256>,
}

#[derive(Clone, Debug)]
pub struct ExecutionResult {
    pub order_id: H256,
    pub status: ExecutionStatus,
    pub filled_amount: U256,
    pub remaining_amount: U256,
    pub average_price: U256,
    pub transactions: Vec<H256>,
    pub gas_used: U256,
    pub total_cost: ExecutionCost,
}

#[derive(Clone, Debug)]
pub struct AccountLimits {
    pub max_positions: u32,
    pub max_notional: U256,
    pub max_leverage: U256,
    pub min_collateral: U256,
    pub max_slippage: u32,
}

#[derive(Clone, Debug)]
pub enum OrderSide {
    Buy,
    Sell,
}

#[derive(Clone, Debug)]
pub enum OrderType {
    Market,
    Limit,
    StopLoss,
    StopLimit,
    TrailingStop,
}

#[derive(Clone, Debug)]
pub enum TimeInForce {
    GoodTilCancelled,
    ImmediateOrCancel,
    FillOrKill,
    GoodTilTime(u64),
}

#[derive(Clone, Debug)]
pub struct ExecutionOptions {
    pub max_slippage: f64,
    pub min_fill_ratio: f64,
    pub allowed_venues: Option<Vec<Address>>,
    pub referral_code: Option<H256>,
}

#[derive(Clone, Debug)]
pub enum ExecutionStatus {
    Pending,
    PartiallyFilled(U256),
    Filled,
    Cancelled,
    Failed(String),
}

#[derive(Clone, Debug)]
pub struct ExecutionConfig {
    pub max_sub_orders: usize,
    pub max_slippage: f64,
    pub min_execution_interval: u64,
    pub gas_price_threshold: U256,
    pub max_retry_attempts: u32,
}

#[derive(Default)]
pub struct OrderBook {
    active_orders: BTreeMap<H256, Order>,
    order_history: VecDeque<(H256, ExecutionResult)>,
    user_orders: HashMap<Address, Vec<H256>>,
    fills: HashMap<H256, Vec<(U256, U256)>>,
    cancellations: HashMap<H256, u64>,
}

impl OrderBook {
    pub fn add_order(&mut self, order: Order) -> Result<()> {
        let order_id = order.id;
        let trader = order.trader;
        
        self.active_orders.insert(order_id, order);
        self.user_orders.entry(trader)
            .or_default()
            .push(order_id);
        Ok(())
    }

    pub fn remove_order(&mut self, order_id: H256) -> Result<Option<Order>> {
        if let Some(order) = self.active_orders.remove(&order_id) {
            if let Some(orders) = self.user_orders.get_mut(&order.trader) {
                orders.retain(|id| *id != order_id);
            }
            Ok(Some(order))
        } else {
            Ok(None)
        }
    }

    pub fn get_order(&self, order_id: &H256) -> Option<&Order> {
        self.active_orders.get(order_id)
    }

    pub fn get_user_orders(&self, trader: &Address) -> Vec<H256> {
        self.user_orders.get(trader)
            .cloned()
            .unwrap_or_default()
    }

    pub fn add_fill(&mut self, order_id: H256, amount: U256, price: U256) {
        self.fills.entry(order_id)
            .or_default()
            .push((amount, price));
    }

    pub fn get_fills(&self, order_id: &H256) -> Vec<(U256, U256)> {
        self.fills.get(order_id)
            .cloned()
            .unwrap_or_default()
    }

    pub fn add_execution_result(&mut self, order_id: H256, result: ExecutionResult) {
        while self.order_history.len() >= 1000 {
            self.order_history.pop_front();
        }
        self.order_history.push_back((order_id, result));
    }
}

#[derive(Default)]
pub struct ExecutionState {
    pub active_executions: HashMap<H256, ExecutionStatus>,
    pub execution_metrics: HashMap<Address, ExecutionMetrics>,
    pub last_prices: HashMap<Address, U256>,
    pub gas_prices: RwLock<GasPriceState>,
}

#[derive(Clone, Debug, Default)]
pub struct ExecutionMetrics {
    pub total_orders: u64,
    pub successful_orders: u64,
    pub failed_orders: u64,
    pub total_volume: U256,
    pub total_gas_used: U256,
}

#[derive(Clone, Debug, Default)]
pub struct GasPriceState {
    pub base_fee: U256,
    pub priority_fee: U256,
    pub last_update: u64,
}

impl ExecutionState {
    pub fn update_execution_status(&mut self, order_id: H256, status: ExecutionStatus) {
        self.active_executions.insert(order_id, status);
    }

    pub fn remove_execution(&mut self, order_id: &H256) -> Option<ExecutionStatus> {
        self.active_executions.remove(order_id)
    }

    pub fn update_metrics(&mut self, trader: Address, success: bool, volume: U256, gas_used: U256) {
        let metrics = self.execution_metrics.entry(trader).or_default();
        metrics.total_orders += 1;
        if success {
            metrics.successful_orders += 1;
        } else {
            metrics.failed_orders += 1;
        }
        metrics.total_volume += volume;
        metrics.total_gas_used += gas_used;
    }

    pub fn update_price(&mut self, token: Address, price: U256) {
        self.last_prices.insert(token, price);
    }
}

impl ExecutionEngine {
    pub fn new(
        router: Arc<dyn ExecutionRouter>,
        risk_manager: Arc<dyn RiskManager>,
        config: ExecutionConfig,
    ) -> Self {
        Self {
            orders: RwLock::new(OrderBook::default()),
            strategies: Vec::new(),
            router,
            risk_manager,
            state: RwLock::new(ExecutionState::default()),
            config,
        }
    }

    pub fn add_strategy(&mut self, strategy: Arc<dyn ExecutionStrategy>) {
        self.strategies.push(strategy);
    }

    pub async fn submit_order(&self, order: Order) -> Result<H256> {
        // Validate order
        if !self.risk_manager.validate_order(&order)? {
            return Err(PanopticError::OrderValidationFailed);
        }

        // Check account limits
        let limits = self.risk_manager.check_limits(order.trader)?;
        self.validate_order_against_limits(&order, &limits)?;

        // Store order
        let mut orders = self.orders.write().await;
        orders.add_order(order.clone())?;
        drop(orders);

        // Start execution
        tokio::spawn(self.execute_order_async(order));

        Ok(order.id)
    }

    async fn execute_order_async(&self, order: Order) {
        let mut state = self.state.write().await;
        state.update_execution_status(order.id, ExecutionStatus::Pending);
        drop(state);

        let result = self.execute_order_internal(order).await;
        
        let mut state = self.state.write().await;
        match result {
            Ok(execution_result) => {
                state.update_metrics(order.trader, true, execution_result.filled_amount, execution_result.gas_used);
                state.update_execution_status(order.id, execution_result.status);
            },
            Err(e) => {
                state.update_metrics(order.trader, false, U256::zero(), U256::zero());
                state.update_execution_status(order.id, ExecutionStatus::Failed(e.to_string()));
            }
        }
    }

    async fn execute_order_internal(&self, order: Order) -> Result<ExecutionResult> {
        // Validate order against account limits
        let limits = self.risk_manager.check_limits(order.trader)?;
        self.validate_order_against_limits(&order, &limits)?;

        // Select best execution strategy
        let strategy = self.select_best_strategy(&order)?;

        // Get execution plan
        let plan = strategy.execute_order(&order).await?;
        let cost = strategy.estimate_cost(&plan)?;

        // Validate execution cost
        self.validate_execution_cost(&order, &cost)?;

        let mut filled_amount = U256::zero();
        let mut transactions = Vec::new();
        let mut total_gas_used = U256::zero();

        // Execute each sub-order
        for sub_order in plan.sub_orders {
            // Check if we should continue execution
            if !self.should_continue_execution(&order, filled_amount)? {
                break;
            }

            // Route and execute sub-order
            match self.router.execute_sub_order(&sub_order).await {
                Ok(tx_hash) => {
                    transactions.push(tx_hash);
                    filled_amount += sub_order.amount;
                    total_gas_used += self.router.estimate_gas(&sub_order)?;

                    // Update order book
                    let mut orders = self.orders.write().await;
                    orders.add_fill(order.id, sub_order.amount, sub_order.price);
                }
                Err(e) => {
                    log::error!("Failed to execute sub-order: {}", e);
                    continue;
                }
            }
        }

        // Calculate final execution result
        let status = self.determine_execution_status(&order, filled_amount);
        let remaining = order.amount.saturating_sub(filled_amount);
        let average_price = self.calculate_average_price(&order, filled_amount, total_gas_used)?;

        let result = ExecutionResult {
            order_id: order.id,
            status,
            filled_amount,
            remaining_amount: remaining,
            average_price,
            transactions,
            gas_used: total_gas_used,
            total_cost: cost,
        };

        // Update risk metrics
        self.risk_manager.update_risk_metrics(&result)?;

        Ok(result)
    }

    fn select_best_strategy(&self, order: &Order) -> Result<Arc<dyn ExecutionStrategy>> {
        let mut best_strategy = None;
        let mut lowest_cost = U256::max_value();

        for strategy in &self.strategies {
            match strategy.execute_order(order).await {
                Ok(plan) => {
                    if let Ok(cost) = strategy.estimate_cost(&plan) {
                        let total_cost = cost.gas_estimate
                            .saturating_add(cost.max_priority_fee)
                            .saturating_add(cost.value);

                        if total_cost < lowest_cost {
                            lowest_cost = total_cost;
                            best_strategy = Some(Arc::clone(strategy));
                        }
                    }
                }
                Err(_) => continue,
            }
        }

        best_strategy.ok_or(PanopticError::NoViableStrategy)
    }

    fn validate_order_against_limits(&self, order: &Order, limits: &AccountLimits) -> Result<()> {
        // Check position count
        let orders = self.orders.read().await;
        let user_orders = orders.get_user_orders(&order.trader);
        if user_orders.len() >= limits.max_positions as usize {
            return Err(PanopticError::PositionLimitExceeded);
        }

        // Check notional value
        let notional = order.amount.saturating_mul(
            order.price.unwrap_or_else(|| {
                let state = self.state.read().await;
                state.last_prices.get(&order.asset)
                    .cloned()
                    .unwrap_or_default()
            })
        );
        if notional > limits.max_notional {
            return Err(PanopticError::NotionalLimitExceeded);
        }

        // Check leverage
        let collateral = self.risk_manager.get_collateral(order.trader)?;
        if collateral < limits.min_collateral {
            return Err(PanopticError::InsufficientCollateral);
        }

        let leverage = notional.saturating_div(collateral);
        if leverage > limits.max_leverage {
            return Err(PanopticError::LeverageLimitExceeded);
        }

        Ok(())
    }

    fn validate_execution_cost(&self, order: &Order, cost: &ExecutionCost) -> Result<()> {
        // Check if user has enough balance for execution
        let balance = self.risk_manager.get_balance(order.trader)?;
        
        let total_cost = cost.gas_estimate
            .saturating_mul(cost.max_fee)
            .saturating_add(cost.value);

        if balance < total_cost {
            return Err(PanopticError::InsufficientBalance);
        }

        // Check token costs
        for (token, amount) in &cost.token_costs {
            let token_balance = self.risk_manager.get_token_balance(order.trader, *token)?;
            if token_balance < *amount {
                return Err(PanopticError::InsufficientTokenBalance);
            }
        }

        Ok(())
    }

    fn should_continue_execution(&self, order: &Order, filled_amount: U256) -> Result<bool> {
        // Check if order is still active
        let orders = self.orders.read().await;
        if orders.get_order(&order.id).is_none() {
            return Ok(false);
        }

        // Check if order is filled
        if filled_amount >= order.amount {
            return Ok(false);
        }

        // Check time in force
        match order.time_in_force {
            TimeInForce::GoodTilCancelled => Ok(true),
            TimeInForce::ImmediateOrCancel => Ok(filled_amount == U256::zero()),
            TimeInForce::FillOrKill => Ok(filled_amount == U256::zero()),
            TimeInForce::GoodTilTime(expiry) => {
                let current_time = self.get_current_block_timestamp()?;
                Ok(current_time <= expiry)
            }
        }
    }

    fn determine_execution_status(&self, order: &Order, filled_amount: U256) -> ExecutionStatus {
        if filled_amount == U256::zero() {
            ExecutionStatus::Failed("No fills executed".to_string())
        } else if filled_amount < order.amount {
            ExecutionStatus::PartiallyFilled(filled_amount)
        } else {
            ExecutionStatus::Filled
        }
    }

    fn calculate_average_price(&self, order: &Order, filled_amount: U256, total_cost: U256) -> Result<U256> {
        if filled_amount == U256::zero() {
            return Ok(U256::zero());
        }

        let orders = self.orders.read().await;
        let fills = orders.get_fills(&order.id);

        let total_notional = fills.iter()
            .fold(U256::zero(), |acc, (amount, price)| {
                acc.saturating_add(amount.saturating_mul(*price))
            });

        Ok(total_notional.saturating_div(filled_amount))
    }

    async fn get_current_block_timestamp(&self) -> Result<u64> {
        // Implementation would get current block timestamp from chain
        Ok(0)
    }
}
