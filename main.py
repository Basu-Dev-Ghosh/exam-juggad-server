from pdf import get_questions_chain,answers_chain,get_suggestion_chain
from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List,Optional
from fastapi.middleware.cors import CORSMiddleware
import PyPDF2
app = FastAPI()

origins = [
    "http://localhost:3000",  # Update with your Next.js app's URL
    # Add any additional origins as needed
]

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/uploadpdf/")
async def upload_pdf(files: List[UploadFile] = File(...)):
    pdf_texts = ''
 
    for uploaded_file in files:
        if uploaded_file.content_type != 'application/pdf':
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        try:
            pdf_reader = PyPDF2.PdfReader(uploaded_file.file)
            for page in pdf_reader.pages:
                pdf_texts += page.extract_text()
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing PDF file: {str(e)}")
    
    questions_chain=get_questions_chain(pdf_texts)
    output=questions_chain.invoke({"extracted_text":pdf_texts})
    return output

@app.get("/ai-answer")
async def get_ai_answer(question: str, marks: Optional[int] = None):
    try:
        # Use the `answers_chain` function to get the AI's answer
        answer = answers_chain.invoke({"question": question, "marks": marks})
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/ai-suggestion/")
async def get_ai_suggestion(files: List[UploadFile] = File(...)):
    pdf_texts = '--------------------------------------------'
    for uploaded_file in files:
        if uploaded_file.content_type != 'application/pdf':
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        try:

            pdf_reader = PyPDF2.PdfReader(uploaded_file.file)
            for page in pdf_reader.pages:
                pdf_texts += page.extract_text()
            pdf_texts +='\n-----------------------------------------------------------------------------'
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing PDF file: {str(e)}")
    suggestion_chain=get_suggestion_chain(pdf_texts)
    output=suggestion_chain.invoke({"extracted_text":pdf_texts})
    return output
    