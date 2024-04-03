import os
import PyPDF2
from dotenv import load_dotenv
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser,StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


load_dotenv()


def process_pdf_file(pdf_filename):
    try:
        # Directory where PDF files are stored
        pdf_directory = "./pdf"


        # Get the full path of the PDF file
        pdf_path = os.path.join(pdf_directory, pdf_filename)

        # Check if the file exists
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"File '{pdf_filename}' not found in '{pdf_directory}'")

        # Open the PDF file
        with open(pdf_path, "rb") as pdf_file:
            # Use PyPDF2 to read the PDF file
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            num_pages = len(pdf_reader.pages)
            # You can perform additional processing here using PyPDF2


             # Initialize a list to store text from all pages
            all_pages_text = ""


            if num_pages > 0:
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    all_pages_text+=page_text
           
            
            # Return information about the PDF
            return all_pages_text
    except Exception as e:
        raise RuntimeError(f"Failed to process PDF file '{pdf_filename}': {e}")



llm=ChatOpenAI(temperature=0)

output_format ="""
MCQ:[
    {
        question: '',
        options: ['','','',''],
        marks:1
    },
    {
        question: '',
        options: ['','','',''],
        marks:1
    },
    {
        question: '',
        options: ['','','',''],
        marks:1
    },
    {
        question: '',
        options: ['','','',''],
        marks:1
    },

],
GroupB:[
    {
      question: '',
      marks: 5,
    
    },
    {
      question: '',
      marks: 5,

    },
    {
      question: '',
      marks: 10,

    },
    {
      question: '',
      marks: 2,

    },
  ],
GroupC:[
    {
      question: '',
      marks: 5,
    
    },
    {
      question: '',
      marks: 5,

    },
    {
      question: '',
      marks: 10,

    },
    {
      question: '',
      marks: 2,

    },
  ],
GroupD:[
    {
      question: '',
      marks: 5,
    
    },
    {
      question: '',
      marks: 5,

    },
    {
      question: '',
      marks: 10,

    },
    {
      question: '',
      marks: 2,

    },
  ],
GroupE:[
    {
      question: '',
      marks: 5,
    
    },
    {
      question: '',
      marks: 5,

    },
    {
      question: '',
      marks: 10,

    },
    {
      question: '',
      marks: 2,

    },
  ]
"""


template="""
{format_instructions}
Here is the extracted text from a question(PDF file):
extracted text:{extracted_text}
Follow this steps
Step 1: extract all mcqs/Group-A as it is from the above extracted text with their marks assigned.
step 2: extract all the Group-B questions as it is from the above extracted text with their marks assigned.
step 3: extract all the Group-C questions as it is from the above extracted text with their marks assigned.
step 4: extract all the Group-D questions as it is from the above extracted text with their marks assigned.
step 5: extract all the Group-E questions as it is from the above extracted text with their marks assigned.

**note: If any table or image found then tell in output that refer the pdf for table/image

Now prepare a json objects for the output looks like below:
{output_format}

Here is a sample or example output given below
samele output:{output_format}
"""





def get_questions_chain(pdf_text):
    class Output(BaseModel):
        mcq: str = Field(description="Extracted mcq/group A questions with their marks assigned")
        GroupB: str = Field(description="Extracted all group B questions with their marks assigned")
        GroupC: str = Field(description="Extracted all group C questions with their marks assigned")
        GroupD: str = Field(description="Extracted all group D questions with their marks assigned")
        GroupE: str = Field(description="Extracted all group E questions with their marks assigned")
    parser = JsonOutputParser(pydantic_object=Output)
    prompt = PromptTemplate(
    template=template,
    input_variables=["extracted_text","output_format"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    questions_chain=(
      RunnablePassthrough.assign(
        extracted_text=lambda _:pdf_text,
        output_format=lambda _:output_format,
        )
      | prompt
     | llm
      | parser
    )
    return questions_chain


template2 = """
You are an AI assistant. You have been asked to answer a question that has appeared in a student's semester exam of their Engineering Course. The question carries a certain number of marks, which will be specified below. Please provide a detailed and accurate answer, keeping in mind the marks allocated for the question. 

Question: {question}
Marks: {marks}
"""
prompt = PromptTemplate(
    template=template2,
    input_variables=["question","marks"],
)

answers_chain=(prompt| llm| StrOutputParser())


template3="""
{format_instructions}
You have given below some question paper's extracted text which are seperated by '-----------------------------------------------------------------':
You have to analyze all given question paper's extracted text and give all mcq,group-B,group-C,group-D,group-E questions which maximum times appeared in the given year question papers below.
extracted text:{extracted_text}
Follow this steps
Step 1: extract all mcqs/Group-A from the above previous years question paper's extracted text which are maximum times appeared in exam's papers and also with their marks assigned.
step 2: extract all the Group-B questions from the above previous years question paper's extracted text which are maximum times appeared in exam's papers and also with their marks assigned.
step 3: extract all the Group-C questions from the above previous years question paper's extracted text which are maximum times appeared in exam's papers and also with their marks assigned.
step 4: extract all the Group-D questions from the above previous years question paper's extracted text which are maximum times appeared in exam's papers and also with their marks assigned.
step 5: extract all the Group-E questions from the above previous years question paper's extracted text which are maximum times appeared in exam's papers and also with their marks assigned.

**note: If any table or image found then tell in output that refer the pdf for table/image

Now prepare a json objects for the output looks like below:
{output_format}

Here is a sample or example output given below
samele output:{output_format}
"""


def get_suggestion_chain(pdf_text):
    class Output(BaseModel):
        mcq: str = Field(description="All mcq questions which maximum times appeared in the previous year question papers")
        GroupB: str = Field(description="All group b questions which maximum times appeared in the previous year question papers")
        GroupC: str = Field(description="All group c questions which maximum times appeared in the previous year question papers")
        GroupD: str = Field(description="All group d questions which maximum times appeared in the previous year question papers")
        GroupE: str = Field(description="All group e questions which maximum times appeared in the previous year question papers")
    parser = JsonOutputParser(pydantic_object=Output)
    prompt = PromptTemplate(
    template=template3,
    input_variables=["extracted_text","output_format"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    suggestion_chain=(
      RunnablePassthrough.assign(
        extracted_text=lambda _:pdf_text,
        output_format=lambda _:output_format,
        )
      | prompt
     | llm
      | parser
    )
    return suggestion_chain




# Example usage:





