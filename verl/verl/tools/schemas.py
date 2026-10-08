from typing import Literal

from pydantic import BaseModel

class OpenAIFunctionPropertySchema(BaseModel):
    pass

    type: str
    description: str | None = None
    enum: list[str] | None = None

class OpenAIFunctionParametersSchema(BaseModel):
    pass

    type: str
    properties: dict[str, OpenAIFunctionPropertySchema]
    required: list[str]

class OpenAIFunctionSchema(BaseModel):
    pass

    name: str
    description: str
    parameters: OpenAIFunctionParametersSchema
    strict: bool = False

class OpenAIFunctionToolSchema(BaseModel):
    pass

    type: str
    function: OpenAIFunctionSchema

class OpenAIFunctionParsedSchema(BaseModel):
    pass

    name: str
    arguments: str  

class OpenAIFunctionToolCall(BaseModel):
    pass

    id: str
    type: Literal["function"] = "function"
    function: OpenAIFunctionParsedSchema
