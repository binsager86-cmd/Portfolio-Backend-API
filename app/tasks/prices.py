from app.core.celery_app import celery_app
from app.services.price_service import update_all_prices


@celery_app.task(name="prices.update_all")
def update_all_prices_task(user_id: int = 1, only_with_holdings: bool = True) -> dict:
    """Run the existing price updater in a Celery worker."""
    result = update_all_prices(user_id=user_id, only_with_holdings=only_with_holdings)
    return result.to_dict()
