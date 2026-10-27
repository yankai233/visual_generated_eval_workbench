from pydantic import BaseModel


class SaveConfigRequest(BaseModel):
    content: str

