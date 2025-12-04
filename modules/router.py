from fastapi import APIRouter, status
from sklearn.linear_model import LogisticRegression
from .scheme import SBSIn, SBSOut
from SBS import SBS

sbs = SBS(
    LogisticRegression,
    5
)

modules_router = APIRouter()


@modules_router.post(
    '/',
    summary='SBS fit',
    status_code=status.HTTP_200_OK,
    response_model=SBSOut
)
async def model_fit(sbs_scheme: SBSIn) -> SBSOut:
    indices = sbs.fit(sbs_scheme.X, sbs_scheme.y).indices
    sbs_scheme = SBSOut(
        X=sbs_scheme.X[:, indices].tolist(),
        y=sbs_scheme.y.tolist(),
        indices=indices.tolist()
    )
    return sbs_scheme