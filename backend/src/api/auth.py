"""Users and roles (U14): viewer < analyst < curator.

* viewer: read everything;
* analyst: run pipelines and projections, upload datasets, configure severity weights,
  study regions and alert subscriptions;
* curator: additionally confirm native-range polygons and annotate records (the expert
  review steps CLAUDE.md requires to stay in human hands).

Authentication is delegated to an identity-aware proxy in front of the API
(`SDM_AUTH_MODE=header`): it authenticates the user against the identity provider and passes
the user name and groups in trusted headers; groups map to roles via `SDM_AUTH_ROLE_GROUPS`.
The proxy must strip these headers from incoming client requests and the API must not be
reachable around it. `SDM_AUTH_MODE=none` (default) is a single-user deployment in which every
request acts as a curator.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status

from src.api.deps import settings_dep
from src.config import Settings
from src.domain import Role

ROLE_ORDER: dict[str, int] = {"viewer": 0, "analyst": 1, "curator": 2}


@dataclass(frozen=True)
class User:
    name: str
    role: Role

    def can(self, role: Role) -> bool:
        return ROLE_ORDER[self.role] >= ROLE_ORDER[role]


def current_user(request: Request, settings: Settings = Depends(settings_dep)) -> User:
    if settings.auth_mode == "none":
        return User("local", "curator")
    name = request.headers.get(settings.auth_user_header)
    if not name:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not authenticated")
    groups = [
        g.strip()
        for g in request.headers.get(settings.auth_groups_header, "").replace(";", ",").split(",")
        if g.strip()
    ]
    role: Role = settings.auth_default_role
    for g in groups:
        mapped = settings.auth_role_groups.get(g)
        if mapped in ROLE_ORDER and ROLE_ORDER[mapped] > ROLE_ORDER[role]:
            role = mapped  # type: ignore[assignment]
    return User(name, role)


CurrentUser = Annotated[User, Depends(current_user)]


def require_role(role: Role) -> Callable[[User], User]:
    def check(user: CurrentUser) -> User:
        if not user.can(role):
            raise HTTPException(
                status.HTTP_403_FORBIDDEN, f"This action needs the {role} role (you: {user.role})"
            )
        return user

    return check


Analyst = Depends(require_role("analyst"))
Curator = Depends(require_role("curator"))
