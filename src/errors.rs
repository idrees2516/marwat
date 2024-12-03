use thiserror::Error;

#[derive(Error, Debug)]
pub enum PanopticError {
    #[error("Pool error: {0}")]
    Pool(#[from] crate::pool::PoolError),
    
    #[error("Factory error: {0}")]
    Factory(#[from] crate::factory::FactoryError),
    
    #[error("Math error: division by zero")]
    DivisionByZero,
    
    #[error("Math error: multiplication overflow")]
    MultiplicationOverflow,
    
    #[error("Math error: addition overflow")]
    AdditionOverflow,
    
    #[error("Math error: subtraction overflow")]
    SubtractionOverflow,
    
    #[error("Math error: sqrt price out of bounds")]
    SqrtPriceOutOfBounds,
    
    #[error("Invalid token")]
    InvalidToken,
    
    #[error("Invalid amount")]
    InvalidAmount,
    
    #[error("Insufficient liquidity")]
    InsufficientLiquidity,
    
    #[error("Insufficient balance")]
    InsufficientBalance,
    
    #[error("Insufficient allowance")]
    InsufficientAllowance,
    
    #[error("Invalid signature")]
    InvalidSignature,
    
    #[error("Deadline expired")]
    DeadlineExpired,
    
    #[error("Unauthorized")]
    Unauthorized,
    
    #[error("Position locked")]
    PositionLocked,
    
    #[error("Position expired")]
    PositionExpired,
    
    #[error("Invalid strike price")]
    InvalidStrike,
    
    #[error("Invalid expiry")]
    InvalidExpiry,
    
    #[error("Invalid collateral")]
    InvalidCollateral,
    
    #[error("Invalid fee")]
    InvalidFee,
    
    #[error("Invalid tick")]
    InvalidTick,
    
    #[error("Invalid price")]
    InvalidPrice,
    
    #[error("Invalid pool")]
    InvalidPool,
    
    #[error("Invalid option type")]
    InvalidOptionType,
    
    #[error("Invalid token ID")]
    InvalidTokenId,
    
    #[error("Invalid owner")]
    InvalidOwner,
    
    #[error("Invalid parameters")]
    InvalidParameters,
    
    #[error("Contract paused")]
    ContractPaused,
    
    #[error("Reentrancy")]
    Reentrancy,
    
    #[error("Custom error: {0}")]
    Custom(String),
}

pub type Result<T> = std::result::Result<T, PanopticError>;
