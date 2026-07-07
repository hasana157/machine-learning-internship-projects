"""
Security utilities for authentication and authorization.

Implements JWT token management, password hashing, and security configurations.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthCredentials
from passlib.context import CryptContext
from jose import JWTError, jwt

from app.core.config import settings

logger = logging.getLogger(__name__)

# ============================================================================
# PASSWORD HASHING
# ============================================================================

# Configure password context with bcrypt
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
)


def hash_password(password: str) -> str:
    """
    Hash password using bcrypt.
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password
    """
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify password against hash.
    
    Args:
        plain_password: Plain text password
        hashed_password: Hashed password from database
        
    Returns:
        True if passwords match
    """
    return pwd_context.verify(plain_password, hashed_password)


# ============================================================================
# JWT TOKEN MANAGEMENT
# ============================================================================

security = HTTPBearer()


def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create JWT access token.
    
    Args:
        data: Token claims data
        expires_delta: Custom expiration time
        
    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
    
    to_encode.update({"exp": expire})
    
    try:
        encoded_jwt = jwt.encode(
            to_encode,
            settings.SECRET_KEY,
            algorithm=settings.ALGORITHM,
        )
        return encoded_jwt
    except Exception as e:
        logger.error(f"Token creation error: {e}")
        raise


def create_refresh_token(user_id: int) -> str:
    """
    Create JWT refresh token.
    
    Args:
        user_id: User ID
        
    Returns:
        Encoded JWT token
    """
    data = {
        "sub": str(user_id),
        "type": "refresh",
    }
    
    expires_delta = timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    return create_access_token(data, expires_delta)


def verify_token(token: str) -> Dict[str, Any]:
    """
    Verify and decode JWT token.
    
    Args:
        token: JWT token string
        
    Returns:
        Token payload dict
        
    Raises:
        HTTPException: If token is invalid
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
        )
        user_id: str = payload.get("sub")
        
        if user_id is None:
            raise credentials_exception
        
        return payload
        
    except JWTError as e:
        logger.error(f"Token verification error: {e}")
        raise credentials_exception


# ============================================================================
# DEPENDENCY INJECTION
# ============================================================================


async def get_current_user(
    credentials: HTTPAuthCredentials = Depends(security),
) -> Dict[str, Any]:
    """
    Get current authenticated user from token.
    
    Used as FastAPI dependency for protected routes.
    
    Args:
        credentials: HTTP Bearer token credentials
        
    Returns:
        Token payload with user information
        
    Raises:
        HTTPException: If token invalid or missing
    """
    token = credentials.credentials
    payload = verify_token(token)
    return payload


async def get_current_user_id(
    credentials: HTTPAuthCredentials = Depends(security),
) -> int:
    """
    Extract current user ID from token.
    
    Args:
        credentials: HTTP Bearer token credentials
        
    Returns:
        User ID
    """
    payload = verify_token(credentials.credentials)
    user_id = int(payload.get("sub"))
    return user_id


# ============================================================================
# SETUP & VALIDATION
# ============================================================================


def setup_security():
    """
    Validate security configuration on startup.
    """
    # Validate SECRET_KEY
    if settings.ENVIRONMENT == "production":
        if len(settings.SECRET_KEY) < 32:
            raise ValueError(
                "SECRET_KEY must be at least 32 characters in production"
            )
        if settings.SECRET_KEY == "change-this-in-production":
            raise ValueError(
                "SECRET_KEY must be changed from default in production"
            )
    
    logger.info("✓ Security configuration validated")


# ============================================================================
# PERMISSION CHECKS
# ============================================================================


async def require_admin(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Dependency to require admin user.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        User payload if admin
        
    Raises:
        HTTPException: If user is not admin
    """
    if not current_user.get("is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return current_user
