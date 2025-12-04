from pydantic import BaseModel, field_validator, model_validator
from fastapi import HTTPException, status
from numpy import array, nanmean, isnan, where, take


class SBSIn(BaseModel):
    model_config = {
        'extra': 'forbid'
    }

    X: list[list]
    y: list


    @field_validator('X')
    def verify_X(cls, value):
        X = array(value)
        if X.ndim != 2:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail='X must be a 2D matrix'
            )
        column_mean = nanmean(X, axis=0)
        indices = where(isnan(X))
        X[indices] = take(column_mean, indices[1])
        return X


    @field_validator('y')
    def verify_y(cls, value):
        y = array(value)
        if y.ndim != 1:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail='y must be a 1D vector'
            )
        mean = nanmean(y)
        indices = where(isnan(y))
        y[indices] = mean
        return y

    @model_validator(mode='after')
    def verify_object(self):
        if self.X.shape[0] != self.y.shape[0]:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail='The number of target values '
                       'must be the same as the number'
                       ' of samples X'
            )
        return self


class SBSOut(BaseModel):

    X: list[list]
    y: list
    indices: list