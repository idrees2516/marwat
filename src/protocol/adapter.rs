use crate::types::{Pool, Result, PanopticError};
use ethers::types::{U256, Address, H256, Transaction};
use std::collections::HashMap;
use std::sync::Arc;

pub struct ProtocolAdapter {
    adapters: HashMap<Address, Box<dyn ProtocolInterface>>,
    conversions: HashMap<(Address, Address), ConversionConfig>,
    max_conversion_amount: U256,
    default_slippage: f64,
}

#[async_trait::async_trait]
pub trait ProtocolInterface: Send + Sync {
    fn get_protocol_id(&self) -> Address;
    async fn get_balance(&self, token: Address, owner: Address) -> Result<U256>;
    async fn get_allowance(&self, token: Address, owner: Address, spender: Address) -> Result<U256>;
    async fn approve(&self, token: Address, spender: Address, amount: U256) -> Result<H256>;
    async fn transfer(&self, token: Address, recipient: Address, amount: U256) -> Result<H256>;
    async fn execute_swap(&self, params: SwapParams) -> Result<SwapResult>;
    async fn get_quote(&self, params: QuoteParams) -> Result<QuoteResult>;
}

pub struct ConversionConfig {
    source_protocol: Address,
    target_protocol: Address,
    conversion_path: Vec<Address>,
    fee_tiers: Vec<u32>,
    min_amount: U256,
    max_amount: U256,
    max_slippage: f64,
}

pub struct SwapParams {
    pub token_in: Address,
    pub token_out: Address,
    pub amount_in: U256,
    pub min_amount_out: U256,
    pub recipient: Address,
    pub deadline: u64,
}

pub struct SwapResult {
    pub transaction_hash: H256,
    pub amount_in: U256,
    pub amount_out: U256,
    pub fee_amount: U256,
    pub path: Vec<Address>,
}

pub struct QuoteParams {
    pub token_in: Address,
    pub token_out: Address,
    pub amount: U256,
    pub side: QuoteSide,
}

pub struct QuoteResult {
    pub price: U256,
    pub amount: U256,
    pub fee: U256,
    pub price_impact: f64,
    pub path: Vec<Address>,
}

#[derive(Debug, Clone, Copy)]
pub enum QuoteSide {
    Buy,
    Sell,
}

impl ProtocolAdapter {
    pub fn new(max_conversion_amount: U256, default_slippage: f64) -> Self {
        Self {
            adapters: HashMap::new(),
            conversions: HashMap::new(),
            max_conversion_amount,
            default_slippage,
        }
    }

    pub fn register_adapter(
        &mut self,
        adapter: Box<dyn ProtocolInterface>,
    ) -> Result<()> {
        let protocol_id = adapter.get_protocol_id();
        if self.adapters.contains_key(&protocol_id) {
            return Err(PanopticError::AdapterAlreadyRegistered);
        }
        self.adapters.insert(protocol_id, adapter);
        Ok(())
    }

    pub fn configure_conversion(
        &mut self,
        source_protocol: Address,
        target_protocol: Address,
        conversion_path: Vec<Address>,
        fee_tiers: Vec<u32>,
        min_amount: U256,
        max_amount: U256,
        max_slippage: f64,
    ) -> Result<()> {
        if !self.adapters.contains_key(&source_protocol) {
            return Err(PanopticError::AdapterNotFound);
        }
        if !self.adapters.contains_key(&target_protocol) {
            return Err(PanopticError::AdapterNotFound);
        }

        let config = ConversionConfig {
            source_protocol,
            target_protocol,
            conversion_path,
            fee_tiers,
            min_amount,
            max_amount,
            max_slippage,
        };

        self.conversions.insert((source_protocol, target_protocol), config);
        Ok(())
    }

    pub async fn convert_between_protocols(
        &self,
        source_protocol: Address,
        target_protocol: Address,
        token_in: Address,
        amount: U256,
        recipient: Address,
    ) -> Result<SwapResult> {
        let config = self.conversions.get(&(source_protocol, target_protocol))
            .ok_or(PanopticError::ConversionNotConfigured)?;

        if amount < config.min_amount || amount > config.max_amount {
            return Err(PanopticError::InvalidConversionAmount);
        }

        let source_adapter = self.adapters.get(&source_protocol)
            .ok_or(PanopticError::AdapterNotFound)?;

        // Get quote for the conversion
        let quote = source_adapter.get_quote(QuoteParams {
            token_in,
            token_out: config.conversion_path[0],
            amount,
            side: QuoteSide::Sell,
        }).await?;

        let min_amount_out = quote.amount * U256::from((1.0 - config.max_slippage) as u128);

        // Execute the conversion
        let swap_result = source_adapter.execute_swap(SwapParams {
            token_in,
            token_out: config.conversion_path[0],
            amount_in: amount,
            min_amount_out,
            recipient,
            deadline: self.get_deadline()?,
        }).await?;

        Ok(swap_result)
    }

    pub async fn get_conversion_quote(
        &self,
        source_protocol: Address,
        target_protocol: Address,
        token_in: Address,
        amount: U256,
    ) -> Result<QuoteResult> {
        let config = self.conversions.get(&(source_protocol, target_protocol))
            .ok_or(PanopticError::ConversionNotConfigured)?;

        let source_adapter = self.adapters.get(&source_protocol)
            .ok_or(PanopticError::AdapterNotFound)?;

        source_adapter.get_quote(QuoteParams {
            token_in,
            token_out: config.conversion_path[0],
            amount,
            side: QuoteSide::Sell,
        }).await
    }

    pub async fn check_allowance(
        &self,
        protocol: Address,
        token: Address,
        owner: Address,
        spender: Address,
    ) -> Result<bool> {
        let adapter = self.adapters.get(&protocol)
            .ok_or(PanopticError::AdapterNotFound)?;

        let allowance = adapter.get_allowance(token, owner, spender).await?;
        Ok(allowance > U256::zero())
    }

    pub async fn ensure_approval(
        &self,
        protocol: Address,
        token: Address,
        spender: Address,
        amount: U256,
    ) -> Result<Option<H256>> {
        let adapter = self.adapters.get(&protocol)
            .ok_or(PanopticError::AdapterNotFound)?;

        let allowance = adapter.get_allowance(token, spender, spender).await?;
        if allowance < amount {
            Ok(Some(adapter.approve(token, spender, amount).await?))
        } else {
            Ok(None)
        }
    }

    fn get_deadline(&self) -> Result<u64> {
        // Implementation would calculate deadline based on current timestamp
        // This is a placeholder that returns a fixed deadline
        Ok(u64::MAX)
    }
}
