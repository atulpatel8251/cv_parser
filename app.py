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
import logging
import os
from dateutil import parser
from dateutil.relativedelta import relativedelta
import tempfile
from spire.doc import Document
from langchain.prompts import PromptTemplate
logging.basicConfig(filename='app.log', level=logging.ERROR)

# Set Streamlit to use wide mode for the full width of the page
st.set_page_config(page_title="Resume Scanner", page_icon=":page_facing_up:",layout="wide")

# Load the logo image
logo_image = Image.open('assests/HD_Human_Resources_Banner.jpg')

# Optionally, you can resize only if the original size is too large
# For high-definition display, consider not resizing if the image is already suitable
resized_logo = logo_image.resize((1500, 300), Image.LANCZOS)  # Maintain quality during resizing

# Display the logo
st.image(resized_logo, use_column_width=True)

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
            doc = docx.Document(io.BytesIO(uploaded_file.read()))
            text = "\n".join([para.text for para in doc.paragraphs])
            return text.strip()

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
    # Define the prompt for extracting years of experience
    prompt_template = f"""
    You are an expert in analyzing CVs. Given the following CV text, please extract the total years of experience and any specific start year mentioned. 
    If the years of experience or start year are not explicitly stated, return "Not found".

    CV Text: {cv_text}

    Return the total years of experience in the following format:
    - Total Experience: [Total Years of Experience] years
    - Start Year: [Start Year] (or "Not found" if not specified)
    """

    try:
        # Query OpenAI API for response
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert at extracting work experience from resumes."},
                {"role": "user", "content": prompt_template}
            ],
            max_tokens=200,
            temperature=0.7,
        )
        
        # Extract the relevant response
        gpt_response = response.choices[0].message.content.strip()

        # Use regex to find a pattern indicating years of experience, including decimals
        experience_match = re.search(r'Total Experience:\s*([0-9]+(?:\.[0-9]+)?(\+)?|([0-9]+\s*-\s*[0-9]+))\s*years?', gpt_response, re.IGNORECASE)
        start_year_match = re.search(r'Start Year:\s*(\d{4})', gpt_response, re.IGNORECASE)

        # Extract total experience
        total_experience = experience_match.group(1) if experience_match else None

        # Handle cases for "+" and ranges
        if total_experience is not None:
            if '+' in total_experience:
                total_experience = total_experience.replace('+', '')  # Extract only the number
                total_experience = float(total_experience)  # Convert to float
            elif '-' in total_experience:
                # If a range is given, extract the lower end of the range
                range_values = total_experience.split('-')
                total_experience = float(range_values[0].strip())
            else:
                total_experience = float(total_experience)  # Convert to float
        else:
            #st.warning("Couldn't extract the total years of experience. Defaulting to 0.")
            total_experience = 0
        
        # Extract start year or default to "Not found"
        start_year = start_year_match.group(1) if start_year_match else "Not found"

        # Calculate experience from the start year to the present if a valid start year is found
        if start_year != "Not found":
            current_year = datetime.now().year
            total_experience = current_year - int(start_year)
        
        # Return experience info as a dictionary
        return {
            "total_years": total_experience,
            "start_year": start_year,
            "extracted_from": cv_text[:100] + "..."  # First 100 chars of CV for reference
        }

    except Exception as e:
        st.error(f"An error occurred while extracting experience: {e}")
        return {"total_years": 0, "start_year": "Not found", "error": str(e)}
    
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
    Extracts candidate name from the CV text content using GPT with enhanced accuracy.
    
    Args:
        cv_text (str): The CV text content
    
    Returns:
        str: Extracted candidate name
    """
    try:
        # More specific prompt for better name extraction
        prompt = (
            "Extract ONLY the candidate's full name from this CV text. "
            "Guidelines:\n"
            "1. Look for the name at the top of the CV\n"
            "2. Look for common CV header patterns like 'Name:', 'Resume of:', 'Curriculum Vitae of:'\n"
            "3. Ignore any other names that might appear in experience or reference sections\n"
            "4. Only return the full name, no titles (Mr., Ms., Dr., etc)\n"
            "5. No additional text or punctuation\n"
            "6. If multiple possible names are found, return the one that appears to be the CV owner\n\n"
            "CV text:\n"
            f"{cv_text[:1000]}"  # Increased to 1000 characters for better context
        )

        # Get first response from GPT
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a name extraction specialist. Extract only the candidate's name exactly as it appears in their CV header."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,
            temperature=0
        )

        # Extract name from first response
        candidate_name = response.choices[0].message.content.strip()

        # If we got a name, validate it with a second prompt
        if candidate_name and len(candidate_name.split()) <= 4:  # Allow up to 4 name parts
            validation_prompt = (
                f"Verify if this extracted name '{candidate_name}' is correct by checking the CV text again.\n"
                "If it's correct, return ONLY the name.\n"
                "If it's incorrect, extract the correct name from this CV text:\n\n"
                f"{cv_text[:1000]}"
            )

            # Get validation response
            validation_response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a name verification specialist. Verify the extracted name against the CV text."},
                    {"role": "user", "content": validation_prompt}
                ],
                max_tokens=50,
                temperature=0
            )

            validated_name = validation_response.choices[0].message.content.strip()

            # Compare original and validated names
            if validated_name and len(validated_name.split()) <= 4:
                # If names match or validated name looks more legitimate
                if validated_name == candidate_name or (
                    len(validated_name.split()) >= 2 and  # Ensure it's at least a first and last name
                    not any(word.lower() in ['resume', 'cv', 'name', 'unknown'] for word in validated_name.split())
                ):
                    return validated_name

        # If validation failed or original extraction was problematic
        # Try one more time with a different approach
        final_prompt = (
            "Find the candidate's name from this CV text. The name should be:\n"
            "1. Located near the top of the CV\n"
            "2. Not be part of an email address or contact details\n"
            "3. Not be a company name or reference name\n"
            "Return ONLY the name, nothing else.\n\n"
            f"{cv_text[:1000]}"
        )

        final_response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a CV analyzer. Find the candidate's name from the CV header."},
                {"role": "user", "content": final_prompt}
            ],
            max_tokens=50,
            temperature=0
        )

        final_name = final_response.choices[0].message.content.strip()

        # Final validation checks
        if final_name and len(final_name.split()) <= 4:
            if not any(word.lower() in ['resume', 'cv', 'name', 'unknown'] for word in final_name.split()):
                return final_name

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
        total_years = experience_info.get("total_years", 0)  # Default to 0 if not found
        
        # Ensure total_years is an integer or float
        total_years = float(total_years) if isinstance(total_years, (int, float)) else 0

        # Extract required years of experience from the criteria
        required_experience = extract_required_experience(criteria.get("experience", "0"))

        # Prepare the GPT prompt for matching
        prompt = (
            "Given the job description criteria and the candidate's CV text, "
            "please match the following: "
            "1. Which education qualifications from the job description are present in the CV? "
            "2. Which experiences from the job description are present in the CV? "
            "3. Which mandatory skills from the job description are present in the CV? "
            "Certifications are only mandatory if they are listed as part of the Essential Criteria in the job description. "
            "4. Which desired skills from the job description are present in the CV? "
            "5. Identify missing qualifications, experiences, skills, or certifications from the CV. "
            "Also, assign a score (0 to 10) for each desired skill based on the extent of match. "
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
    
if "justifications" not in st.session_state:
    st.session_state.justifications = {}

# Function to justify skill scoring based on the candidate's resume
# Cache for skill justifications
cv_skill_scores_cache = {}

def get_skill_score_justification(candidate_name, skill, score, cv_text):
    # Generate a unique hash for the CV to ensure the same CV gets the same skill score
    cv_hash = generate_cv_hash(cv_text)
    
    # Check if the skill score for this skill and CV is already cached
    if cv_hash in cv_skill_scores_cache and skill in cv_skill_scores_cache[cv_hash]:
        #st.success(f"Returning cached skill justification for '{skill}' skill.")
        return cv_skill_scores_cache[cv_hash][skill]

    # Prepare the prompt
    if score == 0:
        prompt = (
            f"Briefly explain in 2-3 bullet points why the candidate's resume text '{cv_text}' does not demonstrate the "
            f"necessary skills for '{skill}', resulting in a score of {score}/10. This score remains the same if the same "
            "resume is uploaded again, as it accurately reflects the candidate's current skill level. Focus on clear, simple "
            "reasons why the candidate lacks this skill."
        )
    else:
        prompt = (
            f"Provide a concise justification in 2-3 bullet points explaining why the candidate's resume text '{cv_text}' "
            f"demonstrates a match for the skill '{skill}' with a score of {score}/10. This score will remain consistent if the same "
            "resume is uploaded again, as it accurately reflects the candidate's skills. Keep each bullet simple, focusing on "
            "clear reasoning in 7-8 words per line. Avoid overly technical terms."
        )

    # Using the 'openai.chat.completions.create' method
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
        temperature=0
    )

    # Access the explanation from the response
    explanation = response.choices[0].message.content.strip()

    # Cache the skill score and explanation for this CV and skill
    if cv_hash not in cv_skill_scores_cache:
        cv_skill_scores_cache[cv_hash] = {}
    
    # Store the explanation and score in the cache
    cv_skill_scores_cache[cv_hash][skill] = explanation

    # Return the explanation
    return explanation


def display_pass_fail_verdict(results, cv_text):
    candidate_name = results['Candidate Name']
    skill_scores = results.get("Stratification", {})

    # Debugging: Print skill_scores to verify the data
    logging.debug(f"Skill scores for {candidate_name}: {skill_scores}")

    with st.container():
        #st.markdown("<div style='padding: 10px; background-color: #f0f8ff; border-radius: 10px;'>", unsafe_allow_html=True)

        # Pass/Fail verdict
        pass_fail = results.get("Status", "Fail")
        if pass_fail == 'Pass':
            st.markdown(f"<h2 style='color: green; font-size: 1em;'>🟢 Final Result: PASS ✔️</h2>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h2 style='color: red; font-size: 1em;'>🔴 Final Result: FAIL ❌</h2>", unsafe_allow_html=True)

        # Check for skill scores
        if skill_scores:
            # Create an expander for candidate selection
            expander = st.expander(f"**Click to view {candidate_name}'s detailed skill assessment**")

            with expander:
                # Prepare a table with candidate name, skill, score, and justification
                table_data = []

                for skill, score in skill_scores.items():
                    # Create a unique session key for each skill justification
                    justification_key = f"justification_{candidate_name}_{skill}"
                    if justification_key not in st.session_state:
                        explanation = get_skill_score_justification(candidate_name, skill, score, cv_text)
                        st.session_state[justification_key] = explanation

                    # Append the row to the table data
                    table_data.append({
                        "Candidate Name": candidate_name,
                        "Skill": skill,
                        "Score": f"{score}/10",
                        "Justification": st.session_state[justification_key]
                    })

                # Display the table with skill assessment
                st.markdown("<h3>Skill Assessment Table</h3>", unsafe_allow_html=True)
                st.table(pd.DataFrame(table_data))  # Displaying table using pandas DataFrame

        # Display overall skill score
        overall_skill_score = round(results.get("Skill Score", 0), 1)
        st.markdown(f"""
    <h3 style='font-size:20px;'>Overall Skill Score: <strong>{overall_skill_score:.1f}</strong> out of 50</h3>
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
    
    st.dataframe(styled_df)  # Display the DataFrame

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
        if jd_text:
            criteria_json = use_genai_to_extract_criteria(jd_text)
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
                        st.markdown(f"### Matching Results for {result['Candidate Name']}:")
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
    through technical interviews based on the information provided in their resumes. 
    Any gaps in the candidate's profile should be addressed and justified.
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

st.markdown(footer, unsafe_allow_html=True)
