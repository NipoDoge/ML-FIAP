from fastapi import APIRouter, Depends
from core.deps import get_current_user
from api.v1.endpoints import users
from api.v1.endpoints import roles
from api.v1.endpoints import authorize
from api.v1.endpoints import health
from platform_ring.domains import domains_router

router = APIRouter()

router.include_router(router=health.router, prefix="/health", tags=["health"])
router.include_router(
    router=users.router, prefix="/users", tags=["users"], dependencies=[Depends(get_current_user)]
)
router.include_router(
    router=roles.router, prefix="/roles", tags=["roles"], dependencies=[Depends(get_current_user)]
)
router.include_router(router=authorize.router, prefix="/auth", tags=["auth"])
router.include_router(domains_router, dependencies=[Depends(get_current_user)])
