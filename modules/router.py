from fastapi import APIRouter, status


modules_router = APIRouter()


@modules_router.post(
    '/',
    summary='SBS fit',
    status_code=status.HTTP_200_OK
)
async def model_fit():
    pass


@modules_router.post(
    '/predict',
    summary='SBS predict',
    status_code=status.HTTP_200_OK
)
async def model_predict():
    pass