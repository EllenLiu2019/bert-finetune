from fastapi import FastAPI
from transformers import pipeline, BertTokenizer
from typing import Literal
from pydantic import BaseModel

app = FastAPI()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
classifier = pipeline("text-classification", model="XiaojingEllen/bert-finetuned-claim-detection", tokenizer=tokenizer)

class PredictionRequest(BaseModel):
    text: str
    threshold: float = 0.5

class PredictionResponse(BaseModel):
    text: str
    label: Literal["check worthy", "uncheck worthy", "unrelated"]
    score: float

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """
    对输入文本进行声明检测分类

    Args:
        request: 包含待分类文本和阈值的请求对象

    Returns:
        PredictionResponse: 分类结果，包括文本、标签和置信度
    """
    result = classifier(request.text)[0]

    # 根据置信度阈值确定最终标签
    if result["score"] > request.threshold:
        # 映射模型标签到业务标签
        label = "check worthy" if result["label"] == "LABEL_1" else "uncheck worthy"
    else:
        label = "unrelated"

    # 构造并返回响应
    return PredictionResponse(
        text=request.text,
        label=label,
        score=result["score"]
    )
