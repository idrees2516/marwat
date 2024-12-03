use crate::types::{Pool, Result, PanopticError, Settlement};
use crate::pricing::models::OptionPricing;
use ethers::types::{U256, Address, H256, Transaction, TransactionReceipt};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;

pub struct SettlementProcessor {
    pricing: Arc<dyn OptionPricing>,
    pending_settlements: RwLock<VecDeque<PendingSettlement>>,
    processed_settlements: HashMap<H256, ProcessedSettlement>,
    batch_size: usize,
    max_retries: u32,
    gas_limit: U256,
    retry_delay: u64,
}

struct PendingSettlement {
    id: H256,
    pool: Address,
    settlement: Settlement,
    attempts: u32,
    last_attempt: u64,
    priority: u8,
}

struct ProcessedSettlement {
    settlement: Settlement,
    transaction: H256,
    block_number: u64,
    gas_used: U256,
    effective_gas_price: U256,
}

impl SettlementProcessor {
    pub fn new(
        pricing: Arc<dyn OptionPricing>,
        batch_size: usize,
        max_retries: u32,
        gas_limit: U256,
        retry_delay: u64,
    ) -> Self {
        Self {
            pricing,
            pending_settlements: RwLock::new(VecDeque::new()),
            processed_settlements: HashMap::new(),
            batch_size,
            max_retries,
            gas_limit,
            retry_delay,
        }
    }

    pub async fn queue_settlement(
        &self,
        settlement: Settlement,
        priority: u8,
    ) -> Result<H256> {
        let pending = PendingSettlement {
            id: settlement.id,
            pool: settlement.pool,
            settlement,
            attempts: 0,
            last_attempt: 0,
            priority,
        };

        let mut queue = self.pending_settlements.write().await;
        queue.push_back(pending);
        
        // Sort by priority
        let mut vec: Vec<_> = queue.drain(..).collect();
        vec.sort_by_key(|s| (s.priority, s.attempts));
        queue.extend(vec);

        Ok(pending.id)
    }

    pub async fn process_batch(&mut self, timestamp: u64) -> Result<Vec<H256>> {
        let mut processed = Vec::new();
        let mut batch_transactions = Vec::new();
        let mut batch_size = 0;

        let mut queue = self.pending_settlements.write().await;
        while let Some(mut settlement) = queue.pop_front() {
            if !self.should_process(&settlement, timestamp) {
                queue.push_back(settlement);
                continue;
            }

            match self.prepare_settlement_transaction(&settlement) {
                Ok(tx) => {
                    batch_transactions.push((settlement.id, tx));
                    batch_size += 1;

                    if batch_size >= self.batch_size {
                        break;
                    }
                },
                Err(e) => {
                    settlement.attempts += 1;
                    settlement.last_attempt = timestamp;

                    if settlement.attempts >= self.max_retries {
                        log::error!("Settlement {} failed after {} attempts: {}", 
                            settlement.id, settlement.attempts, e);
                    } else {
                        queue.push_back(settlement);
                    }
                }
            }
        }

        if !batch_transactions.is_empty() {
            match self.submit_batch_transactions(batch_transactions).await {
                Ok(results) => {
                    for (id, result) in results {
                        if result.success {
                            processed.push(id);
                            self.processed_settlements.insert(id, ProcessedSettlement {
                                settlement: queue.iter()
                                    .find(|s| s.id == id)
                                    .map(|s| s.settlement.clone())
                                    .ok_or(PanopticError::SettlementNotFound)?,
                                transaction: result.transaction_hash,
                                block_number: result.block_number,
                                gas_used: result.gas_used,
                                effective_gas_price: result.effective_gas_price,
                            });
                        } else {
                            // Return failed settlements to queue
                            if let Some(mut settlement) = queue.iter_mut().find(|s| s.id == id) {
                                settlement.attempts += 1;
                                settlement.last_attempt = timestamp;
                                if settlement.attempts < self.max_retries {
                                    queue.push_back(settlement.clone());
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    log::error!("Batch transaction submission failed: {}", e);
                    // Return all settlements to queue
                    for (id, _) in batch_transactions {
                        if let Some(mut settlement) = queue.iter_mut().find(|s| s.id == id) {
                            settlement.attempts += 1;
                            settlement.last_attempt = timestamp;
                            if settlement.attempts < self.max_retries {
                                queue.push_back(settlement.clone());
                            }
                        }
                    }
                }
            }
        }

        Ok(processed)
    }

    pub async fn get_settlement_status(&self, id: H256) -> Result<SettlementStatus> {
        if let Some(processed) = self.processed_settlements.get(&id) {
            return Ok(SettlementStatus::Completed {
                transaction_hash: processed.transaction,
                block_number: processed.block_number,
                gas_used: processed.gas_used,
                effective_gas_price: processed.effective_gas_price,
            });
        }

        let queue = self.pending_settlements.read().await;
        if let Some(pending) = queue.iter().find(|s| s.id == id) {
            return Ok(SettlementStatus::Pending {
                attempts: pending.attempts,
                last_attempt: pending.last_attempt,
                priority: pending.priority,
            });
        }

        Ok(SettlementStatus::Failed {
            reason: "Settlement not found".to_string(),
            attempts: 0,
        })
    }

    fn should_process(&self, settlement: &PendingSettlement, timestamp: u64) -> bool {
        // Skip if too many attempts
        if settlement.attempts >= self.max_retries {
            return false;
        }

        // Check retry delay
        if settlement.last_attempt > 0 {
            let elapsed = timestamp - settlement.last_attempt;
            if elapsed < self.retry_delay {
                return false;
            }
        }

        // High priority settlements are processed immediately
        if settlement.priority <= 1 {
            return true;
        }

        // For other priorities, ensure some delay between attempts
        let min_delay = match settlement.priority {
            2 => self.retry_delay,
            3 => self.retry_delay * 2,
            _ => self.retry_delay * 4,
        };

        settlement.last_attempt == 0 || 
        (timestamp - settlement.last_attempt) >= min_delay
    }

    fn prepare_settlement_transaction(&self, settlement: &PendingSettlement) -> Result<Transaction> {
        let settlement_data = ethers::abi::encode(&[
            ethers::abi::Token::Address(settlement.pool),
            ethers::abi::Token::Uint(settlement.settlement.size),
            ethers::abi::Token::Uint(settlement.settlement.strike_price),
            ethers::abi::Token::Uint(settlement.settlement.settlement_price),
            ethers::abi::Token::Uint(U256::from(settlement.settlement.timestamp)),
        ]);

        Ok(Transaction {
            from: None, // Will be filled by the executor
            to: Some(settlement.pool),
            gas: Some(self.gas_limit),
            gas_price: None, // Will be filled by the executor
            value: None,
            data: Some(settlement_data.into()),
            nonce: None, // Will be filled by the executor
            chain_id: None, // Will be filled by the executor
        })
    }

    async fn submit_batch_transactions(
        &self,
        transactions: Vec<(H256, Transaction)>,
    ) -> Result<Vec<(H256, TransactionResult)>> {
        let mut results = Vec::new();
        let mut current_nonce = self.get_current_nonce().await?;
        let gas_price = self.get_current_gas_price().await?;

        for (id, mut tx) in transactions {
            // Fill in transaction details
            tx.nonce = Some(current_nonce.into());
            tx.gas_price = Some(gas_price);
            
            match self.submit_transaction(tx).await {
                Ok(receipt) => {
                    results.push((id, TransactionResult {
                        success: receipt.status.unwrap_or_default().as_u64() == 1,
                        transaction_hash: receipt.transaction_hash,
                        block_number: receipt.block_number.unwrap_or_default().as_u64(),
                        gas_used: receipt.gas_used.unwrap_or_default(),
                        effective_gas_price: receipt.effective_gas_price.unwrap_or_default(),
                    }));
                    current_nonce += 1;
                }
                Err(e) => {
                    log::error!("Transaction submission failed for settlement {}: {}", id, e);
                    results.push((id, TransactionResult {
                        success: false,
                        transaction_hash: H256::zero(),
                        block_number: 0,
                        gas_used: U256::zero(),
                        effective_gas_price: U256::zero(),
                    }));
                }
            }
        }

        Ok(results)
    }

    async fn get_current_nonce(&self) -> Result<u64> {
        // In a real implementation, this would query the network
        Ok(0)
    }

    async fn get_current_gas_price(&self) -> Result<U256> {
        // In a real implementation, this would query the network
        Ok(U256::from(50000000000u64)) // 50 GWEI
    }

    async fn submit_transaction(&self, transaction: Transaction) -> Result<TransactionReceipt> {
        // In a real implementation, this would submit to the network
        Ok(TransactionReceipt {
            transaction_hash: H256::zero(),
            block_number: Some(0.into()),
            gas_used: Some(U256::zero()),
            effective_gas_price: Some(U256::zero()),
            status: Some(1.into()),
            ..Default::default()
        })
    }
}

#[derive(Debug)]
pub enum SettlementStatus {
    Pending {
        attempts: u32,
        last_attempt: u64,
        priority: u8,
    },
    Processing {
        transaction_hash: H256,
    },
    Completed {
        transaction_hash: H256,
        block_number: u64,
        gas_used: U256,
        effective_gas_price: U256,
    },
    Failed {
        reason: String,
        attempts: u32,
    },
}

#[derive(Debug)]
pub struct TransactionResult {
    pub success: bool,
    pub transaction_hash: H256,
    pub block_number: u64,
    pub gas_used: U256,
    pub effective_gas_price: U256,
}
