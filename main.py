from fastapi import FastAPI, File, UploadFile
import uvicorn

import sys
sys.path.append("/")
from models.llm_client import llm_client
from evaluation.evaluation_client import evaluation_client
from storage.storage_client import storage_client
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.get("/")
def health_check():
    return {"health check": "passed"}

# data
@app.post("/upload_documents")
async def upload_documents(item_id: str = None, file: UploadFile = File(...)):
    try:
        data = await file.read()
        storage_client.store_documents(data, item_id)
        return {"message": "Documents successfully uploaded"}
    except Exception as e: 
        return {"message": str(e)}
    
@app.post("/delete_documents")
def delete_documents(item_id: str):
    try:
        storage_client.delete_documents(item_id)
        return {"message": "Documents successfully deleted"}
    except Exception as e:
        return {"message": str(e)}

# generation & retrieval
@app.post("/retrieve_and_generate")
def retrieve_and_generate(query: str, item_id: str, retrieval_method: str, generation_method: str):
    try:
        response = llm_client.generate(query, item_id, retrieval_method, generation_method)
        return {"response": response}
    except Exception as e:
        return {"message": str(e)}

# evaluation
@app.post("/create_testset")
async def create_testset(test_size: int, file: UploadFile = File(...)):
    try:
        data = await file.read()
        testset_bytes = await evaluation_client.create_testset(data=data, test_size=test_size)
        return StreamingResponse(
            iter([testset_bytes]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=testset.csv"}
        )
    except Exception as e:
        return {"message": str(e)}

@app.post("/evaluate_one")
async def evaluate_one(retrieval_method: str, generation_method: str, item_id: str, file: UploadFile = File(...)):
    try:
        data = await file.read()
        results = evaluation_client.evaluate_one(data, retrieval_method, generation_method, item_id)
        return results
    except Exception as e:
        return {"message": str(e)}

@app.post("/evaluate_all")
async def evaluate_all(item_id: str, file: UploadFile = File(...)):
    try:
        data = await file.read()
        results = evaluation_client.evaluate_all(data, item_id)
        return results
    except Exception as e:
        return {"message": str(e)}

if __name__=="__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)