"""
Authentication endpoints for user login, registration, and token refresh.
"""

import logging
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.database import get_db, User
from app.core.security import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    verify_token,
    get_current_user,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# SCHEMAS
# ============================================================================


class LoginRequest(BaseModel):
    """User login request"""
    username: str
    password: str


class RegisterRequest(BaseModel):
    """User registration request"""
    username: str
    email: EmailStr
    password: str
    full_name: str = None


class TokenResponse(BaseModel):
    """Token response with access and refresh tokens"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    """User profile response"""
    id: int
    username: str
    email: str
    full_name: str = None
    is_active: bool
    created_at: str
    
    class Config:
        from_attributes = True


# ============================================================================
# ENDPOINTS
# ============================================================================


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register new user",
)
async def register(
    request: RegisterRequest,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    """
    Register a new user account.
    
    Args:
        request: Registration details
        db: Database session
        
    Returns:
        Access and refresh tokens
        
    Raises:
        HTTPException: If user already exists
    """
    # Check if user already exists
    result = await db.execute(
        select(User).where(
            (User.username == request.username) | (User.email == request.email)
        )
    )
    
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username or email already registered",
        )
    
    # Create new user
    hashed_password = hash_password(request.password)
    
    user = User(
        username=request.username,
        email=request.email,
        full_name=request.full_name,
        hashed_password=hashed_password,
        is_active=True,
    )
    
    db.add(user)
    await db.commit()
    await db.refresh(user)
    
    logger.info(f"New user registered: {user.username}")
    
    # Generate tokens
    access_token = create_access_token(
        {
            "sub": str(user.id),
            "username": user.username,
            "email": user.email,
        }
    )
    refresh_token = create_refresh_token(user.id)
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
    )


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Login user",
)
async def login(
    request: LoginRequest,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    """
    Authenticate user and return tokens.
    
    Args:
        request: Login credentials
        db: Database session
        
    Returns:
        Access and refresh tokens
        
    Raises:
        HTTPException: If credentials invalid
    """
    # Find user
    result = await db.execute(
        select(User).where(User.username == request.username)
    )
    user = result.scalar_one_or_none()
    
    if not user or not verify_password(request.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled",
        )
    
    logger.info(f"User logged in: {user.username}")
    
    # Generate tokens
    access_token = create_access_token(
        {
            "sub": str(user.id),
            "username": user.username,
            "email": user.email,
            "is_admin": user.is_admin,
        }
    )
    refresh_token = create_refresh_token(user.id)
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
    )


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Refresh access token",
)
async def refresh_token(
    request: Dict[str, str],
    db: AsyncSession = Depends(get_db),
):
    """
    Refresh access token using refresh token.
    
    Args:
        request: Dict with "refresh_token"
        db: Database session
        
    Returns:
        New access token
        
    Raises:
        HTTPException: If refresh token invalid
    """
    refresh_token_str = request.get("refresh_token")
    
    if not refresh_token_str:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Refresh token required",
        )
    
    # Verify refresh token
    payload = verify_token(refresh_token_str)
    user_id = int(payload.get("sub"))
    
    # Get user
    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()
    
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )
    
    # Create new access token
    access_token = create_access_token(
        {
            "sub": str(user.id),
            "username": user.username,
            "email": user.email,
            "is_admin": user.is_admin,
        }
    )
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token_str,
    )


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user",
)
async def get_me(
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> UserResponse:
    """
    Get current authenticated user's profile.
    
    Args:
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        User profile
    """
    user_id = int(current_user.get("sub"))
    
    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    
    return UserResponse.model_validate(user)
