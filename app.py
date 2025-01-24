import openai
import PyPDF2
import io
import json
import hashlib
import re
from datetime import datetime
import pandas as pd
import streamlit as st
from PIL import Image
import base64
import docx
import docx2txt
import logging
import os
from dateutil import parser
from dateutil.relativedelta import relativedelta
import tempfile
from spire.doc import Document
# from dotenv import load_dotenv
# load_dotenv()
#from langchain.prompts import PromptTemplate
logging.basicConfig(filename='app.log', level=logging.ERROR)

# Set Streamlit to use wide mode for the full width of the page
st.set_page_config(page_title="Resume Scanner", page_icon=":page_facing_up:",layout="wide")

# Load the logo image
logo_image = Image.open('assests/HD_Human_Resources_Banner.jpg')

# Optionally, you can resize only if the original size is too large
# For high-definition display, consider not resizing if the image is already suitable
resized_logo = logo_image.resize((1500, 300), Image.LANCZOS)  # Maintain quality during resizing

# Display the logo
st.image(resized_logo, use_container_width=True)

# Function to add background image from a local file
def add_bg_from_local(image_file,opacity=0.7):
    with open(image_file, "rb") as image:
        encoded_image = base64.b64encode(image.read()).decode()

    # Inject custom CSS for background
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(255, 255, 255, {opacity}), rgba(255, 255, 255, {opacity})),url("data:assests/logo.jfif;base64,{encoded_image}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the function with your image path
#add_bg_from_local('assests/OIP.jfif')  # Adjust path to your image file

# Add styled container for the title and description with a maximum width
st.markdown("""
    <div style="background-color: lightblue; padding: 20px; border-radius: 10px; text-align: center; 
                 max-width: 450px; margin: auto;">
        <p style="color: black; margin-left: 20px; margin-top: 20px; font-weight: bold;font-size: 40px;">CV Screening Portal</p>
        <p style="color: black; margin-left: 15px; margin-top: 20px; font-size: 25px;">AI based CV screening</p>
    </div>
""", unsafe_allow_html=True)

# Change background color using custom CSS
st.markdown(
    """
    <style>
    body {
               background-color: green;

    }
    </style>
    """,
    unsafe_allow_html=True
)

# Hide Streamlit's default footer and customize H1 style
hide_streamlit_style = """
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

with open('style.css') as f:
 st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True) 
# Example: Add your image handling or other logic here
images = ['6MarkQ']


openai.api_key = st.secrets["secret_section"]["OPENAI_API_KEY"]

# Function to extract text from PDF uploaded via Streamlit


def extract_text_from_doc(file_path):
    """
    Extract text from a .doc file using Spire.Doc
    
    Args:
        file_path: Path to the .doc file
    
    Returns:
        str: Extracted text from the file
    """
    # Create a Document object
    document = Document()
    
    # Load the Word document
    document.LoadFromFile(file_path)

    # Extract the text of the document
    document_text = document.GetText()
    document.Close()  # Close the document
    return document_text

def extract_text_from_uploaded_pdf(uploaded_file):
    """
    Extract text from an uploaded PDF, DOCX, or DOC file.

    Args:
        uploaded_file: A file-like object containing the document

    Returns:
        str: Extracted text from the file or an empty string if an error occurs
    """
    try:
        # Determine the file type
        file_type = uploaded_file.name.split('.')[-1].lower()

        if file_type == "pdf":
            # Extract text from PDF
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
            text = "\n".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
            return text.strip()

        elif file_type == "docx":
            # Extract text from DOCX
            doc = docx2txt.process(uploaded_file)
            print(doc)
            return doc.strip()

        elif file_type == "doc":
            # For .doc files, we need to save it temporarily and use Spire.Doc
            with tempfile.NamedTemporaryFile(delete=False, suffix='.doc') as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            try:
                # Extract text using Spire.Doc
                text = extract_text_from_doc(tmp_file_path)
                return text
            finally:
                # Clean up the temporary file
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)

        else:
            # Show error message for unsupported file types
            st.error("This file type is not supported. Please upload only PDF, DOCX, or DOC files.", icon="🚫")
            return ""

    except Exception as e:
        st.error(f"Error reading file: {e}", icon="🚫")
        logging.error(f"Error reading file: {e}")
        return ""
# Function to use GenAI to extract criteria from job description
def use_genai_to_extract_criteria(jd_text):
    prompt = (
        "Extract and structure the following details from the job description: "
        "1. Education requirements "
        "2. Required experience "
        "3. Mandatory skills "
        "4. Certifications "
        "5. Desired skills (for brownie points). "
        "The job description is as follows:\n\n"
        f"{jd_text}\n\n"
        "Please provide the response as a JSON object. For example:\n"
        "{\"education\": \"Bachelor's Degree, Master's Degree\", "
        "\"experience\": \"5 years experience in data science\", "
        "\"skills\": \"Python, SQL, Machine Learning\", "
        "\"Certifications\": \"AWS Certified, PMP\", "
        "\"desired_skills\": \"Deep Learning, NLP\"}"
    )
    
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0
        )
        
        content = response.choices[0].message.content.strip()
        
        try:
            return content
        except json.JSONDecodeError:
            st.error("Failed to parse JSON from AI response. Here's the raw response:")
            st.write(content)
            return ""
    except Exception as e:
        st.error(f"Error with OpenAI API: {e}")
        return ""
    
def calculate_skill_score(skill_scores):
    # Sum up all the skill scores
    total_score = sum(skill_scores.values())    
    return total_score

# Function to extract total years of experience using OpenAI's GPT model
@st.cache_data
def extract_experience_from_cv(cv_text):
    # Get the current year dynamically
    x = datetime.now()
    current_year = x.strftime("%m-%Y")
    
    print("CV_Text : ",cv_text)
    # Enhanced prompt template specifically for handling overlapping dates
    prompt_template = f"""
    Please analyze this  {cv_text} carefully to calculate the total years of professional experience. Follow these steps:
    
    1. First, list out all date ranges found in chronological order:
       - Replace 'Current' or 'till date' with {current_year}
       - Include all years mentioned with positions Formats from following 
       - Format as YYYY-YYYY for each position
       - Format as DD-MM-YYYY - DD-MM-YYYY(For example:10-Jul-12 to 31-Jan-21)
       - Format as YYYY-MM-DD - YYYY-MM-DD(For example:12-Jul-10 to 21-Jan-31)
       - Format as YYYY-DD-MM - YYYY-DD-MM(For example:12-10-Jul to 21-31-Jan)
       - Format as MM-YYYY-MM-YYYY
       - Format as YYYY-MM-YYYY-MM
       - Format as YYYY-YY.
       - if in the cv_text 'start year' and 'present year' or 'end year' are not mentioned,then extract experience from 'cvtext' , if also not mentioned experience in cvtext then return 0.
    2. Then, merge overlapping periods :
       - Identify any overlapping years
       - Only count overlapping periods once
       - Create a timeline of non-overlapping periods

    3. Calculate total experience:
       - Sum up all non-overlapping periods.
       - Round to one decimal place Return in Sngle Value.
       -     
    """

    try:
        # Query GPT-4 API with enhanced system message
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system", 
                    "content": """You are an expert in analyzing cv_text and calculating professional experience 
                     Pay special attention to overlapping date ranges(If two or more projects within the same company 
                     overlap, merge them into a single continuous range.) and ensure no double-counting of experience.
                     Always show your work step by step."""
                },
                {"role": "user", "content": prompt_template}
            ],
            max_tokens=1000,
            temperature=0
        )
        
        # Extract GPT response
        gpt_response = response.choices[0].message.content.strip()
        print("gpt_response:",gpt_response)
        # Handle "present" or "current year" in the response
        gpt_response = gpt_response.replace("present", str(current_year)).replace("current", str(current_year))
        print("gpt_response:",gpt_response)
        # Extract experience and start year using improved regex
        experience_match = re.findall(r'(\d+(?:\.\d+)?)\s*years?', gpt_response, re.IGNORECASE)
        print("experience_match:",experience_match)
        start_year_match = re.search(r'Start Year:\s*(\d{4})', gpt_response, re.IGNORECASE)
        
        # Extract and convert values
        if experience_match:
    # Choose the most relevant value by looking at the context or largest value
            total_experience = max(map(float, experience_match))
            print("total_experience:", total_experience)
            total_experience = str(round(total_experience, 1))  # Round to one decimal place
            print("total_experience2:", total_experience)
        else:
            total_experience = "Not found"

            
        start_year = start_year_match.group(1) if start_year_match else "Not found"
        
        # Debugging output
        print("\nFull GPT Response:", gpt_response)
        print("\nExtracted Total Experience:", total_experience)
        print("Extracted Start Year:", start_year)
        
        return {
            "total_years": total_experience,
            "start_year": start_year,
            "gpt_response": gpt_response
        }
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return {
            "total_years": "Not found",
            "start_year": "Not found",
            "error": str(e)
        }

cv_cache = {}

def normalize_cv_text(cv_text):
    """ Normalize CV text for consistent hashing. """
    return ' '.join(cv_text.strip().split()).lower()

def generate_cv_hash(cv_text):
    """ Generate a consistent hash for the normalized CV text. """
    normalized_text = normalize_cv_text(cv_text)
    return hashlib.sha256(normalized_text.encode()).hexdigest()

def calculate_skill_score(skill_scores):
    """
    Calculate the total skill score from the skill stratification.
    Returns the sum of all skill scores.
    
    Args:
        skill_scores (dict): Dictionary of skills and their scores
    
    Returns:
        float: Total sum of skill scores (not averaged)
    """
    if not skill_scores:
        return 0
    
    # Calculate the total sum of scores
    total_score = sum(skill_scores.values())
    
    return total_score
def extract_candidate_name_from_cv(cv_text):
    """
    Extracts the candidate's name from the {cv_text} content.

    Args:
        cv_text (str): The {cv_text} content.

    Returns:
        str: Extracted candidate name or 'Unknown Candidate' if not found.
    """
    try:
        # Extract top few lines of the CV for name extraction
        # cv_header = "\n".join(cv_text.splitlines()[:50])  # Top 5 lines for context
        # print("cvheader:",cv_header)
        # Prompt for GPT
        prompt = (
            "Extract the candidate's full name from this cv_text. The name is likely at the top "
            "'Name:', 'Resume of:', 'Name of Staff','name of candidate','first name last name', or similar. Ignore job titles, contact details, and other information.\n\n"
            f"Name:\n{cv_text}"
        )

        # GPT call
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a name extraction specialist. Extract candidate name only from a {cv_text}."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=30,
            temperature=0
        )

        # Extracted name
        candidate_name = response.choices[0].message.content.strip()
        print('candidate_name:',candidate_name)
        # Validate and return
        if candidate_name and len(candidate_name.split()) <= 4:
            return candidate_name
        return "Unknown Candidate"

    except Exception as e:
        logging.error(f"Error extracting candidate name: {e}")
        return "Unknown Candidate"
cv_cache = {}

def match_cv_with_criteria(cv_text, criteria_json):
    # Generate hash for the CV
    cv_hash = generate_cv_hash(cv_text)

    # Check if the CV has already been processed and return cached result if available
    if cv_hash in cv_cache:
        #st.success("Returning cached result for this CV.")
        return cv_cache[cv_hash]

    # Check if criteria_json is empty or None
    if not criteria_json:
        st.error("Criteria JSON is empty or invalid.")
        return {'cv_text': cv_text}

    try:
        # Extract candidate name from CV text
        candidate_name = extract_candidate_name_from_cv(cv_text)

        # Load criteria JSON
        criteria = json.loads(criteria_json)

        # Extract total years of experience from the CV using the provided function
        experience_info = extract_experience_from_cv(cv_text)
        print("experience:",experience_info)
        total_years = experience_info.get("total_years", 0)  # Default to 0 if not found
        print("total : ", total_years)  # Should print the original value (e.g., "15")

        # Ensure total_years is properly converted to a float
        try:
            total_years = float(total_years)  # Convert to float if possible
        except ValueError:
            total_years = 0  # Default to 0 if conversion fails

        print("total1 : ", total_years)
        # Extract required years of experience from the criteria
        required_experience = extract_required_experience(criteria.get("experience", "0"))

        # Prepare the GPT prompt for matching
        prompt = (
            "Given the job description criteria {criteria_json} and the candidate's {cv_text}, "
            "please match the following: "
            "1. Which education qualifications from the job description are present in the CV? "
            "2. Which experiences from the job description are present in the CV? "
            "3. Which mandatory skills from the job description are present in the CV? "
            "4. 'Certifications' or 'Professional Certification' (mandatory if they are listed as part of the Essential Criteria in the job description). "
            "5. Which desired skills from the job description are present in the CV? "
            "6. Identify missing qualifications, experiences, skills, or certifications from the {cv_text}. "
            "Also, assign a score (0 to 10) for each desired skill based on the extent of match."
            "The job description criteria are as follows:\n\n"
            f"{criteria_json}\n\n"
            "The CV text is as follows:\n\n"
            f"{cv_text}\n\n"
            f"Total Years of Experience: {total_years} Years.\n\n"
            "Please provide the response in the following format: "
            "{\"Matching Education\": [...], \n"
            "\"Matching Experience\": [...], \n"
            "\"Matching Skills\": [...], \n"
            "\"Matching Certifications\": [...], \n"
            "\"Missing Requirements\": [...], \n"
            "\"Stratification\": {\"skill_1\": 8, \"skill_2\": 6, ...}}"
        )

        # Fetching the result from GPT-3.5
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0
        )

        # Parse the GPT response
        matching_results = response.choices[0].message.content.strip()
        results = json.loads(matching_results)

        # Filter out None or empty values
        for key in ["Matching Education", "Matching Experience", "Matching Skills", "Matching Certifications"]:
            if key in results and results[key] is None:
                results[key] = []  # Replace None with an empty list

        # Determine pass/fail status
        pass_fail = "Pass"  # Default to Pass

        # Check experience requirement
        if total_years < required_experience:
            pass_fail = "Fail"
            if "Missing Requirements" not in results:
                results["Missing Requirements"] = []
            results["Missing Requirements"].append(f"Required {required_experience} years of experience, has {total_years}")

        # Check mandatory requirements
        mandatory_skills = criteria.get("mandatory_skills", [])
        matching_skills = set(results.get("Matching Skills", []))
        
        missing_mandatory = [skill for skill in mandatory_skills if skill not in matching_skills]
        if missing_mandatory:
            pass_fail = "Fail"
            if "Missing Requirements" not in results:
                results["Missing Requirements"] = []
            results["Missing Requirements"].extend(missing_mandatory)

        # Calculate overall skill score
        skill_scores = results.get("Stratification", {})
        overall_skill_score = calculate_skill_score(skill_scores)

        # Add final results
        results.update({
            "Candidate Name": candidate_name,
            "Status": pass_fail,
            "Skill Score": overall_skill_score,
            "Total Years of Experience": total_years
        })

        # Cache results
        cv_cache[cv_hash] = results

        return results

    except json.JSONDecodeError as e:
        st.error(f"Error parsing criteria JSON: {e}")
        return {}
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return {}


def extract_required_experience(experience_str):
    """
    Extract numeric value(s) for required experience from a given string.
    
    Args:
    - experience_str (str): The string containing required experience information.

    Returns:
    - float: The extracted numeric value for required experience.
    """
    try:
        # Use regex to find all numbers in the string and convert them to floats
        numbers = re.findall(r'\d+', experience_str)
        
        # Convert found numbers to floats and return the maximum value found (assuming multiple values)
        return max(float(num) for num in numbers) if numbers else 0.0
        
    except Exception as e:
        st.warning(f"Couldn't extract a numeric value from required experience: {experience_str}. Error: {e}")
        return 0.0  # Default to 0 if extraction fails
    
def get_skill_score_justification(criteria_json, skill, score, cv_text):
    """
    Generate a justification for the skill score based on the candidate's resume text
    and evaluation criteria.
    
    :param criteria_json: JSON object with evaluation criteria
    :param skill: Skill being evaluated
    :param score: Score assigned to the skill
    :param cv_text: Candidate's resume text
    :return: Justification for the skill score
    """
    # Prepare the prompt
    if score == 0:
        prompt = (
            f"Based on the evaluation criteria provided in {criteria_json}, explain in 2 bullet points only,explain every bullent points just 20 words only why the candidate's "
            f"resume text '{cv_text}' does not demonstrate the necessary skills for '{skill}', resulting in a score of {score}/10. "
            "Focus on specific, unique shortcomings in the candidate's resume text that justify this score. "
            "Avoid generic or repetitive explanations across different resumes."
            "*Strictly Every Bullent Points are started from next line.*"
        )
    else:
        prompt = (
            f"Based on the evaluation criteria provided in {criteria_json}, provide a concise justification in 2 bullet points only,explain every bullent points just 20 words only "
            f"explaining why the candidate's resume text '{cv_text}' demonstrates a match for the skill '{skill}' with a score of {score}/10. "
            "Focus on specific, unique strengths in the candidate's resume text that align with the criteria and justify the score. "
            "Avoid reusing the same points for different resumes and ensure each explanation is unique."
            "*Strictly Every Bullent Points are started from next line.*"
        )

    # Using the 'openai.chat.completions.create' method
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
        temperature=0.1
    )

    # Access the explanation from the response
    explanation = response.choices[0].message.content.strip()

    # Return the explanation
    return explanation

def display_pass_fail_verdict(results, cv_text):
    candidate_name = results['Candidate Name']
    skill_scores = results.get("Stratification", {})

    # Debugging: Print skill_scores to verify the data
    logging.debug(f"Skill scores for {candidate_name}: {skill_scores}")

    with st.container():
        # Pass/Fail verdict
        pass_fail = results.get("Status", "Fail")
        if pass_fail == 'Pass':
            st.markdown(f"<h2 style='color: green; font-size: 1em;'>🟢 Profile Eligibility: PASS ✔️</h2>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h2 style='color: red; font-size: 1em;'>🔴 Profile Eligibility: FAIL ❌</h2>", unsafe_allow_html=True)

        # Check for skill scores
        if skill_scores:
            # Create an expander for candidate selection
            expander = st.expander(f"**Click to view {candidate_name}'s detailed skill assessment**")

            with expander:
                # Prepare a table with candidate name, skill, score, and justification
                table_data = []

                for skill, score in skill_scores.items():
                    # Generate justification dynamically for each skill
                    explanation = get_skill_score_justification(criteria_json, skill, score, cv_text)

                    # Append the row to the table data
                    table_data.append({
                        "Candidate Name": candidate_name,
                        "Skill": skill,
                        "Score": f"{score}/10",
                        "Justification": explanation
                    })
                df = pd.DataFrame(table_data)
                blankIndex=[''] * len(df)
                df.index=blankIndex
                #st.markdown(df.to_html(index=False), unsafe_allow_html=True)
                st.table(df)

        # Display overall skill score
        Additional_Skill_Score = round(results.get("Skill Score", 0), 1)
        st.markdown(f"""
    <h3 style='font-size:20px;'>Additional_Skill_Score: <strong>{Additional_Skill_Score:.1f}</strong> out of 50</h3>
    """, unsafe_allow_html=True)

        # Pass/Fail message based on the final status
        if pass_fail == "Pass":
            st.success("The candidate has passed based on the job description criteria.")
        else:
            st.error("The candidate has failed to meet the job description criteria.")

        st.markdown("</div>", unsafe_allow_html=True)


# Function to display and rank candidates in a table
def display_candidates_table(candidates):
    if not candidates:
        st.info("No candidates to display.")
        return

    # Create DataFrame from candidates
    df = pd.DataFrame(candidates)
    # df.to_string(index=False)
    # Ensure 'pass_or_fail' and 'skill_score' columns exist
    if 'Status' not in df.columns or 'Skill Score' not in df.columns or 'Total Years of Experience' not in df.columns or 'Candidate Name' not in df.columns:
        st.error("Missing required columns in candidates data.")
        return

    # Format Total Years of Experience to one decimal place
    df['Total Years of Experience'] = df['Total Years of Experience'].apply(lambda x: f"{float(x):.1f}").astype(float)

    # Sort by pass/fail first, then by skill score
    df['rank'] = df.apply(lambda row: (0 if row['Status'] == 'Pass' else 1, -row['Skill Score'], -row['Total Years of Experience']), axis=1)
    df = df.sort_values(by='rank').drop(columns=['rank'])  # Dropping 'rank' for display

    # Reorder columns to place 'skill_score' second-to-last
    other_columns = [col for col in df.columns if col not in ['Candidate Name', 'Status', 'Skill Score', 'Total Years of Experience']]
    columns_order = ['Candidate Name', 'Status', 'Total Years of Experience'] + other_columns + ['Skill Score']  # Move 'skill_score' second-to-last
    df = df[columns_order]

    # Display the table with beautification
    st.markdown("## :trophy: Candidate Rankings")
    
    # Apply coloring function for pass/fail column
    def color_pass_fail(val):
        color = 'lightgreen' if val == 'Pass' else 'lightcoral'
        return f'background-color: {color}'

    # Use Styler for table display with custom number formatting
    styled_df = df.style.applymap(color_pass_fail, subset=['Status'])
    
    # Format the Total Years of Experience column to show exactly one decimal place
    styled_df = styled_df.format({
        'Total Years of Experience': '{:.1f}',
        'Skill Score': '{:.0f}'  # Format skill score as whole number
    })
    
    st.dataframe(styled_df,hide_index=True)  # Display the DataFrame

# Add a styled box for the file uploaders
st.markdown("""
    <div style="background-color: lightblue; padding: 4px; border-radius: 5px; text-align: left; 
                 max-width: 380px;">
        <p style="color: black; margin-left: 60px; margin-top: 20px; font-weight: bold;font-size: 20px;">Upload Job Description (PDF)</p>
    </div>
""", unsafe_allow_html=True)

# File uploader for Job Description
jd_file = st.file_uploader(" ", type=["pdf", "docx","doc"])

# Header for Candidate Resumes Upload
st.markdown("""
    <div style="background-color: lightblue; padding: 2px; border-radius: 5px; text-align: left; 
                 max-width: 400px;">
        <p style="color: black; margin-left: 55px; margin-top: 20px; font-weight: bold;font-size: 20px;">Upload Candidate Resumes (PDF)</p>
    </div>
""", unsafe_allow_html=True)

# File uploader for Candidate Resumes
cv_files = st.file_uploader("", type=["pdf", "docx","doc"], accept_multiple_files=True)

# Ensure criteria_json is initialized in session state
if 'criteria_json' not in st.session_state:
    st.session_state['criteria_json'] = None

# Button to extract criteria and match candidates
if st.button("Extract Criteria and Match Candidates"):
    if jd_file:
        jd_text = extract_text_from_uploaded_pdf(jd_file)
        #print("jdtext:",jd_text)
        if jd_text:
            criteria_json = use_genai_to_extract_criteria(jd_text)
            #print("criteria jd text",criteria_json)
            if criteria_json:
                # Save criteria in session state
                st.session_state.criteria_json = criteria_json
                st.success("Job description criteria extracted successfully.")

                # Proceed to match candidates
                if cv_files:
                    candidates_results = []
                    for cv_file in cv_files:
                        cv_text = extract_text_from_uploaded_pdf(cv_file)
                        if cv_text:
                            results = match_cv_with_criteria(cv_text, st.session_state.criteria_json)
                            if results:
                                candidates_results.append(results)
                        else:
                            st.error(f"Failed to extract text from CV file.")
                    
                    # Display candidates table first
                    if candidates_results:
                        display_candidates_table(candidates_results)
                        
                    # Then display individual matching details
                    for result in candidates_results:
                        st.markdown(f"### Results for {result['Candidate Name']}:")
                        display_pass_fail_verdict(result, cv_text)
                else:
                    st.error("Please upload at least one CV PDF.")
            else:
                st.error("Failed to extract job description criteria.")
        else:
            st.error("The uploaded JD file appears to be empty.")
    else:
        st.error("Please upload a Job Description PDF.")


disclaimer_text = """
<div style="color: grey; margin-top: 50px;">
    <strong>Disclaimer:</strong> Certification validity must be verified manually. 
    These results are AI-generated, and candidates should be thoroughly evaluated 
    through technical interviews For the information provided in their resumes. 
    
</div>
"""

st.markdown(disclaimer_text, unsafe_allow_html=True)

     
        
footer = """
    <style>
    body {
        margin: 0;
        padding-top: 70px;  /* Add padding to prevent content from being hidden behind the footer */
    }
    .footer {
        position: absolute;
        top: 80px;
        left: 0;
        width: 100%;
        background-color: #002F74;
        color: white;
        text-align: center;
        padding: 5px;
        font-weight: bold;
        z-index: 1000;  /* Ensure it is on top of other elements */
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
    }
    .footer p {
        font-style: italic;
        font-size: 14px;
        margin: 0;
        flex: 1 1 50%;  /* Flex-grow, flex-shrink, flex-basis */
    }
    @media (max-width: 600px) {
        .footer p {
            flex-basis: 100%;
            text-align: center;
            padding-top: 10px;
        }
    }
    </style>
    <div class="footer">
        <p style="text-align: left;">Copyright © 2024 MPSeDC. All rights reserved.</p>
        <p style="text-align: right;">The responses provided on this website are AI-generated. User discretion is advised.</p>
    </div>
"""
