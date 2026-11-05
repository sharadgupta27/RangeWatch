"""Early-detection alerts (U10 / B10): list, RSS feed and subscriptions."""

from __future__ import annotations

import re
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status

from src.api.auth import Analyst, CurrentUser
from src.api.deps import repository_dep, settings_dep
from src.api.schemas import AlertOut, AlertSubscriptionIn, AlertSubscriptionOut
from src.config import Settings
from src.domain import AlertSubscription
from src.monitoring.alerts import WebhookTargetError, rss_feed, validate_webhook_url
from src.persistence.repository import NotFoundError, Repository

router = APIRouter(tags=["alerts"])
RepoDep = Annotated[Repository, Depends(repository_dep)]
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@router.get(
    "/species/{taxon_key}/alerts", response_model=list[AlertOut], operation_id="listSpeciesAlerts"
)
def species_alerts(
    taxon_key: int, repo: RepoDep, limit: Annotated[int, Query(ge=1, le=200)] = 50
) -> list[AlertOut]:
    if repo.get_species(taxon_key) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Species {taxon_key} not in registry")
    return [AlertOut.model_validate(a) for a in repo.list_alerts(taxon_key, limit)]


@router.get("/alerts", response_model=list[AlertOut], operation_id="listAlerts")
def all_alerts(repo: RepoDep, limit: Annotated[int, Query(ge=1, le=200)] = 50) -> list[AlertOut]:
    return [AlertOut.model_validate(a) for a in repo.list_alerts(None, limit)]


@router.get(
    "/alerts/feed.xml",
    response_class=Response,
    operation_id="alertFeed",
    responses={200: {"content": {"application/rss+xml": {}}}},
)
def feed(
    repo: RepoDep,
    taxon_key: int | None = None,
    settings: Settings = Depends(settings_dep),
) -> Response:
    """RSS 2.0 feed of alerts (all species, or one)."""
    alerts = repo.list_alerts(taxon_key, 100)
    names = {s.taxon_key: s.canonical_name or s.scientific_name for s in repo.list_species()}
    return Response(rss_feed(alerts, names, settings), media_type="application/rss+xml")


@router.get(
    "/alerts/subscriptions",
    response_model=list[AlertSubscriptionOut],
    operation_id="listAlertSubscriptions",
)
def list_subscriptions(repo: RepoDep, taxon_key: int | None = None) -> list[AlertSubscriptionOut]:
    """Subscriptions covering one species (its own + all-species), or all."""
    return [AlertSubscriptionOut.model_validate(s) for s in repo.list_subscriptions(taxon_key)]


@router.post(
    "/alerts/subscriptions",
    response_model=AlertSubscriptionOut,
    status_code=status.HTTP_201_CREATED,
    operation_id="createAlertSubscription",
    dependencies=[Analyst],
)
def create_subscription(
    body: AlertSubscriptionIn,
    repo: RepoDep,
    user: CurrentUser,
    settings: Settings = Depends(settings_dep),
) -> AlertSubscriptionOut:
    target = body.target.strip()
    if body.channel == "webhook":
        try:
            validate_webhook_url(target, settings)
        except WebhookTargetError as exc:
            raise HTTPException(422, str(exc)) from exc
    elif not _EMAIL.match(target):
        raise HTTPException(422, "Not an e-mail address")
    if body.taxon_key is not None and repo.get_species(body.taxon_key) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Species {body.taxon_key} not in registry")
    sub = repo.save_subscription(
        AlertSubscription(
            channel=body.channel,
            target=target,
            taxon_key=body.taxon_key,
            kinds=list(body.kinds),
            created_by=user.name,
        )
    )
    return AlertSubscriptionOut.model_validate(sub)


@router.delete(
    "/alerts/subscriptions/{subscription_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    operation_id="deleteAlertSubscription",
    dependencies=[Analyst],
)
def delete_subscription(subscription_id: uuid.UUID, repo: RepoDep) -> None:
    try:
        repo.delete_subscription(subscription_id)
    except NotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Subscription not found") from exc
